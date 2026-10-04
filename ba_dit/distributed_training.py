"""Two-rank adapter training: global sample cursor, averaged gradients, atomic resume."""
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ba_dit import adapters
from ba_dit.checkpoint import load_adapters, restore_training, save_training, trainable_parameters
from ba_dit.config import digest
from ba_dit.data.conditioning import TrainingConditioner
from ba_dit.data.manifest import assert_disjoint, read_manifest
from ba_dit.logging import connect, log_metrics
from ba_dit.runtime import backend_module


def rank_rng():
    return {'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
            'cuda_rng': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state['torch_rng'])
    random.setstate(state['python_rng'])
    if state['cuda_rng'] is not None:
        torch.cuda.set_rng_state(state['cuda_rng'])


def sample_offsets(cursor, rank, world, accumulation):
    """A global shuffled stream, interleaved across ranks without padding duplicates."""
    return range(cursor + rank, cursor + world * accumulation, world)


def all_true(value, device):
    flag = torch.tensor(int(value), device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def train_segment(config, mode, run_dir, until, resume=None, init_adapter=None, limit=None):
    from ba_dit.nn.masked_face_attention import training_mask
    from ba_dit.training import sample_at

    world = config['training']['world_size']
    if int(os.environ.get('WORLD_SIZE', '1')) != world or mode != 'branch_only' or config['training'].get('microbatch_size', 1) != 1:
        raise ValueError('Launch branch_only with torchrun --nproc-per-node=2')
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '8'))))
    if config['training']['deterministic']:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    experiment = None
    try:
        capacity = torch.cuda.get_device_properties(device).total_memory
        if not all_true(capacity >= config['hardware']['min_vram_gb'] * 10**9, device):
            raise RuntimeError('At least one GPU is below the per-device VRAM requirement')
        torch.manual_seed(config['training']['seed'])
        random.seed(config['training']['seed'])
        rows = read_manifest(config['data']['train_manifest'], training=True, limit=limit)
        validation = read_manifest(config['data']['validation_manifest'])
        assert_disjoint(rows, validation)
        if len(rows) < world * config['training']['grad_accum']:
            raise ValueError('DDP requires at least one global batch of distinct training rows')
        data_digest = digest([{k: v for k, v in row.items() if k not in {'reference', 'target'}} for row in rows])
        backend = backend_module(config)
        model = backend.load_transformer(config)
        inventory = adapters.install(model, config, mode)
        if config['training']['gradient_checkpointing']:
            model.enable_gradient_checkpointing()
        optimizer = torch.optim.AdamW(adapters.groups(model, config['training']['lr']),
                                     weight_decay=config['training']['weight_decay'])
        warmup = config['training']['warmup']
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., (step + 1) / max(1, warmup)))
        conditioner = TrainingConditioner(config, backend)
        scaler = torch.amp.GradScaler('cuda', enabled=config['model'].get('dtype') == 'float16', init_scale=32.)
        parameters = trainable_parameters(model)
        run_dir = Path(run_dir)
        if rank == 0:
            (run_dir/'optimizer_inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
            experiment = connect(config, run_dir)
            if experiment:
                experiment.log_parameters({'training/world_size': world, 'training/microbatch_per_gpu': 1,
                                           'training/effective_batch': world*config['training']['grad_accum']})
        ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
        torch.manual_seed(config['training']['seed'] + rank)
        random.seed(config['training']['seed'] + rank)
        step, cursor = 0, 0
        if resume:
            load_adapters(model, resume, config, mode)
            step, cursor = restore_training(optimizer, schedule, resume, config, data_digest, rank, scaler)
        elif init_adapter:
            load_adapters(model, init_adapter, config, mode)
        if until <= step:
            raise ValueError('Requested end must follow the checkpoint step')
        model.train()
        accumulation = config['training']['grad_accum']
        torch.cuda.reset_peak_memory_stats(device)
        print(json.dumps({'rank': rank, 'device': str(device), 'start_step': step, 'global_cursor': cursor,
                          'microbatch_per_gpu': 1, 'effective_batch': world*accumulation,
                          'dtype': config['model'].get('dtype', 'bfloat16')}), flush=True)
        while step < until:
            started, rng = time.monotonic(), rank_rng()
            for retry in range(5):
                optimizer.zero_grad(set_to_none=True)
                restore_rng(rng)
                losses = []
                for micro, offset in enumerate(sample_offsets(cursor, rank, world, accumulation)):
                    row = sample_at(rows, offset, config['training']['seed'])
                    tensors, metadata = conditioner(row)
                    tensors['target_face_mask'] = training_mask(row, config).to(device)
                    context = ddp.no_sync() if micro + 1 < accumulation else nullcontext()
                    with context:
                        loss = backend.training_loss(ddp, tensors, config, True)
                        if not all_true(torch.isfinite(loss).item(), device):
                            raise RuntimeError(f'Nonfinite forward loss at update {step + 1}; no dtype fallback')
                        scaler.scale(loss / accumulation).backward()
                    losses.append(float(loss.detach()))
                    del tensors, loss
                scaler.unscale_(optimizer)
                finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters.values())
                if all_true(finite, device):
                    break
                if not scaler.is_enabled() or retry == 4:
                    raise RuntimeError('Nonfinite/missing branch gradients after bounded loss-scale retries')
                scaler.update(new_scale=scaler.get_scale()/2)
            if step == 0 and not all_true(all(p.grad.count_nonzero() for n, p in parameters.items() if n.endswith('.b')), device):
                raise RuntimeError('Missing first B gradients')
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError('Frozen backbone accumulated gradients')
            norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), config['training']['gradient_clip'], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            step += 1
            cursor += world * accumulation
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_reserved(device)
            local_stats = [sum(losses)/accumulation, time.monotonic()-started, peak/2**30, peak/capacity]
            stats = [None] * world
            dist.all_gather_object(stats, local_stats)
            memory_failed = max(s[3] for s in stats) >= config['training']['max_reserved_fraction']
            if rank == 0:
                metrics = {'train/loss': sum(s[0] for s in stats)/world, 'train/gradient_norm': float(norm),
                           'train/seconds': max(s[1] for s in stats), 'train/samples_seen': cursor,
                           'train/lr': optimizer.param_groups[0]['lr'], 'train/loss_scale': scaler.get_scale(),
                           'train/overflow_retries': retry, 'training/effective_batch': world*accumulation,
                           'hardware/peak_reserved_gib': max(s[2] for s in stats),
                           'hardware/reserved_fraction': max(s[3] for s in stats)}
                metrics.update({f'hardware/rank{i}_peak_reserved_gib': s[2] for i, s in enumerate(stats)})
                if step == 1:
                    metrics['train/updated_b_matrices'] = sum(int(p.count_nonzero() > 0) for n,p in parameters.items() if n.endswith('.b'))
                log_metrics(experiment, run_dir, metrics, step)
            if step % config['training']['checkpoint_every'] == 0 or step == until or memory_failed:
                states = [None] * world
                dist.all_gather_object(states, rank_rng())
                if rank == 0:
                    checkpoint = save_training(model, optimizer, schedule, config, mode, run_dir, step, cursor,
                                               data_digest, {'world_size': world, 'ranks': states, 'scaler': scaler.state_dict()})
                    (run_dir/'latest_checkpoint.txt').write_text(str(checkpoint)+'\n')
                dist.barrier()
            if memory_failed:
                raise RuntimeError('Per-GPU reserved-memory admission failed; checkpoint saved')
    finally:
        if experiment:
            experiment.end()
        dist.destroy_process_group()
