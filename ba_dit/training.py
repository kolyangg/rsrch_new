"""Adapter-only optimizer loop over native cached or live-encoded inputs."""

import json
import os
import random
import time
from functools import lru_cache
from pathlib import Path

import torch

from ba_dit import adapters
from ba_dit.checkpoint import load_adapters, restore_training, save_training, trainable_parameters
from ba_dit.config import digest
from ba_dit.data.conditioning import TrainingConditioner
from ba_dit.data.manifest import assert_disjoint, read_manifest
from ba_dit.logging import connect, log_metrics
from ba_dit.runtime import backend_module


@lru_cache(maxsize=1)
def epoch_order(length, epoch, seed):
    return torch.randperm(length, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def sample_at(rows, cursor, seed):
    epoch, offset = divmod(cursor, len(rows))
    order = epoch_order(len(rows), epoch, seed)
    return rows[order[offset]]


def train_segment(config, mode, run_dir, until, resume=None, init_adapter=None, allow_small_gpu=False, limit=None):
    if config['training'].get('world_size', 1) > 1:
        from ba_dit.distributed_training import train_segment as distributed_segment
        return distributed_segment(config, mode, run_dir, until, resume, init_adapter, limit)
    if mode == "native":
        raise ValueError("Native mode has no optimizer; use infer")
    run_dir = Path(run_dir)
    if config["training"]["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    if not allow_small_gpu and total_memory < config["hardware"]["min_vram_gb"] * 10**9:
        raise RuntimeError("GPU does not meet this training profile; --allow-small-gpu is for explicit local smoke checks")
    torch.manual_seed(config["training"]["seed"])
    random.seed(config["training"]["seed"])
    rows = read_manifest(config["data"]["train_manifest"], training=True, limit=limit)
    validation = read_manifest(config["data"]["validation_manifest"])
    assert_disjoint(rows, validation)
    data_digest = digest([{key: value for key, value in row.items() if key not in {"reference", "target"}} for row in rows])
    backend = backend_module(config)
    model = backend.load_transformer(config)
    inventory = adapters.install(model, config, mode)
    (run_dir / "optimizer_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    if config["training"]["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(adapters.groups(model, config["training"]["lr"]),
                                 weight_decay=config["training"]["weight_decay"])
    warmup = config["training"]["warmup"]
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(1, warmup)))
    conditioner = TrainingConditioner(config, backend)
    # Adapter modes consume different initialization draws. Match training noise
    # and timestep draws across the LoRA/branch controls after registration.
    torch.manual_seed(config["training"]["seed"])
    random.seed(config["training"]["seed"])
    step, cursor = 0, 0
    if resume:
        load_adapters(model, resume, config, mode)
        step, cursor = restore_training(optimizer, schedule, resume, config, data_digest)
    elif init_adapter:
        load_adapters(model, init_adapter, config, mode)
    experiment = connect(config, run_dir)
    trainable = trainable_parameters(model)
    branch_enabled = mode in {"branch_only", "lora_plus_branch"}
    face_masks = {}
    if config['branch'].get('kind') == 'masked_face_qkvo':
        from ba_dit.nn.masked_face_attention import training_mask
        face_masks = {row['sample_id']: training_mask(row, config) for row in rows}
    print(json.dumps({"cuda_modules": [type(model).__name__], "frozen_encoder_loaded": conditioner.online,
                      "vae_loaded": conditioner.online, "conditioning":config['data'].get('conditioning','cached'),
                      "trainable_parameters": sum(parameter.numel() for parameter in trainable.values()), "mode": mode}), flush=True)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    try:
        while step < until:
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(config["training"]["grad_accum"]):
                row = sample_at(rows, cursor, config["training"]["seed"])
                tensors, metadata = conditioner(row)
                if face_masks:
                    tensors['target_face_mask'] = face_masks[row['sample_id']].to('cuda')
                loss = backend.training_loss(model, tensors, config, branch_enabled)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at sample {row['sample_id']}")
                (loss / config["training"]["grad_accum"]).backward()
                losses.append(float(loss.detach()))
                cursor += 1
                del tensors, loss
            gradients = [parameter.grad for parameter in trainable.values() if parameter.grad is not None]
            if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise RuntimeError("Missing/nonfinite adapter gradients")
            b_gradient = sum(float(parameter.grad.float().square().sum()) for name, parameter in trainable.items()
                             if name.endswith(".b") and parameter.grad is not None) ** 0.5
            if step == 0:
                missing = [name for name, parameter in trainable.items() if name.endswith(".b")
                           and (parameter.grad is None or not parameter.grad.count_nonzero())]
                if missing:
                    raise RuntimeError(f"Missing first B-matrix gradients: {missing}")
            if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
                raise RuntimeError("Frozen backbone accumulated parameter gradients")
            norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), config["training"]["gradient_clip"])
            optimizer.step()
            schedule.step()
            step += 1
            torch.cuda.synchronize()
            fraction = torch.cuda.max_memory_reserved() / total_memory
            metrics = {"train/loss": sum(losses) / len(losses), "train/gradient_norm": float(norm),
                       "train/b_gradient_norm": b_gradient, "train/lr": optimizer.param_groups[0]["lr"],
                       "train/seconds": time.monotonic() - started, "train/samples_seen": cursor,
                       "hardware/peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                       "hardware/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "hardware/reserved_fraction": fraction,
                       "tokens/reference": metadata["vae"]["reference_tokens"],
                       "tokens/text_slots": metadata["encoder"]["text_tokens"]}
            if step == 1:
                metrics["train/updated_b_matrices"] = sum(int(p.count_nonzero() > 0) for n, p in trainable.items() if n.endswith(".b"))
            log_metrics(experiment, run_dir, metrics, step)
            needs_save = step % config["training"]["checkpoint_every"] == 0 or step == until or fraction > config["training"]["max_reserved_fraction"]
            if needs_save:
                checkpoint = save_training(model, optimizer, schedule, config, mode, run_dir, step, cursor, data_digest)
                (run_dir / "latest_checkpoint.txt").write_text(str(checkpoint) + "\n")
            if fraction > config["training"]["max_reserved_fraction"]:
                raise RuntimeError(f"Memory acceptance failed: {fraction:.3f} > {config['training']['max_reserved_fraction']:.3f}; checkpoint saved")
    finally:
        if experiment:
            experiment.end()
