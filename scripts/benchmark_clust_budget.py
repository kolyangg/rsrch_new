"""Bounded V100 timing probes with a dedicated frozen text-encoder GPU.

These 128-row probes estimate a budget, not complete pretrained admission
or image quality. Resolution and mixed-precision cases are separately named.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import yaml

from ba_dit.config import ROOT, load_config


def synchronize():
    for device in range(2):
        torch.cuda.synchronize(device)


def measure(function):
    synchronize()
    start = time.monotonic()
    result = function()
    synchronize()
    return result, time.monotonic() - start


def fp32_residual_inputs(module, args, kwargs):
    """Probe only: retain residual/modulation range while autocasting linears."""
    def cast(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.float()
        if isinstance(value, (tuple, list)):
            return type(value)(cast(item) for item in value)
        return value
    return tuple(cast(value) for value in args), {key:cast(value) for key,value in kwargs.items()}


def require_finite(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'Nonfinite {name}')


def fp32_branch_route(route):
    def wrapped(*args, **kwargs):
        args, kwargs = fp32_residual_inputs(None, args, kwargs)
        with torch.autocast('cuda', enabled=False):
            return route(*args, **kwargs)
    return wrapped


def worker(args):
    from PIL import Image
    from ba_dit import adapters
    from ba_dit.data.geometry import reference_geometry
    from ba_dit.data.manifest import assert_disjoint, read_manifest
    from ba_dit.nn.masked_face_attention import training_mask
    from ba_dit.runtime import backend_module
    from ba_dit.training import sample_at

    assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count() == 2
    devices = [torch.cuda.get_device_properties(i) for i in range(2)]
    assert all('V100' in d.name and d.total_memory > 30*10**9 for d in devices)
    for i in range(2):
        torch.cuda.set_per_process_memory_fraction(.9, i)
    torch.cuda.set_device(0)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(142)
    config = load_config(args.config)
    config['name'] = args.output.name
    config['data'].update(target_size=[args.resolution]*2, encoder_device='cuda')
    config['training'].update(world_size=1, grad_accum=1, steps=args.updates,
                              gradient_checkpointing=args.checkpointing)
    config['logging']['enabled'] = False
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    rows = read_manifest(config['data']['train_manifest'], training=True, limit=128)
    validation = read_manifest(config['data']['validation_manifest'])
    assert_disjoint(rows, validation)
    backend = backend_module(config)
    model = backend.load_transformer(config)
    if args.amp_fp16:
        for block in [*model.double_blocks, *model.single_blocks]:
            block.register_forward_pre_hook(fp32_residual_inputs, with_kwargs=True)
    def compute(function, *values, **kwargs):
        # No persistent half-weight cache beside the FP32 master weights.
        with torch.autocast('cuda', dtype=torch.float16, enabled=args.amp_fp16, cache_enabled=False):
            return function(*values, **kwargs)
    (args.output/'probe_settings.json').write_text(json.dumps({
        'amp_fp16':args.amp_fp16, 'fp32_residual_inputs':args.amp_fp16, 'fp32_branch_math':args.amp_fp16,
        'autocast_cache':False, 'gradient_scaler_initial_scale':32 if args.amp_fp16 else None}, indent=2)+'\n')
    with torch.cuda.device(1):
        encoder = backend.load_encoder(config)
    vae = backend.load_vae(config)
    assert encoder.text_encoder.device == torch.device('cuda:1')
    print('Loaded FP32 denoiser/VAE on GPU0 and frozen FP32 encoder on GPU1', flush=True)

    def condition(row):
        with torch.no_grad(), torch.random.fork_rng(devices=[0, 1]):
            (text, _), text_seconds = measure(lambda: backend.encode_text(encoder, config, row))
            (images, _), vae_seconds = measure(lambda: backend.encode_images(vae, config, row))
        tensors = {k:v.to('cuda:0') for k,v in {**text, **images}.items()}
        return tensors, {'encoder_seconds':text_seconds, 'vae_seconds':vae_seconds}

    tensors, _ = condition(rows[0])
    noisy = torch.zeros_like(tensors['target_latent'])
    sigma = torch.tensor([.5], device='cuda:0')
    precision_difference = None
    with torch.no_grad():
        native = compute(backend.predict, model, tensors, noisy, sigma, config, False)
        require_finite(native, 'native prediction')
        if args.amp_fp16:
            full = backend.predict(model, tensors, noisy, sigma, config, False)
            precision_difference = float((native.float()-full).square().mean().sqrt()/full.square().mean().sqrt())
            del full
    adapters.install(model, config, 'branch_only')
    if args.amp_fp16:
        for block in [*model.double_blocks, *model.single_blocks]:
            if hasattr(block, 'reference_branch'):
                branch = block.reference_branch
                branch.route_attention = fp32_branch_route(branch.route_attention)
    with torch.no_grad():
        assert torch.equal(native, compute(backend.predict, model, tensors, noisy, sigma, config, False))
    del tensors, noisy, native
    model.gradient_checkpointing = args.checkpointing
    model.train()
    params = {n:p for n,p in model.named_parameters() if p.requires_grad}
    assert len(params) == 64 and sum(p.numel() for p in params.values()) == 25165824
    before = {n:p.detach().cpu().clone() for n,p in params.items()}
    optimizer = torch.optim.AdamW(adapters.groups(model, config['training']['lr']), weight_decay=0.)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp_fp16, init_scale=32.)
    for i in range(2):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(config['training']['seed'])
    timings = []
    for step in range(args.updates):
        synchronize(); start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        row = sample_at(rows, step, config['training']['seed'])
        tensors, timing = condition(row)
        tensors['target_face_mask'] = training_mask(row, config).to('cuda:0')
        loss, timing['forward_seconds'] = measure(lambda: compute(backend.training_loss, model, tensors, config, True))
        require_finite(loss, 'training loss')
        _, timing['backward_seconds'] = measure(lambda: scaler.scale(loss).backward())
        scaler.unscale_(optimizer)
        assert all(p.grad is not None for p in params.values())
        for name,p in params.items():
            require_finite(p.grad, name+' gradient')
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1., error_if_nonfinite=True)
        for group in optimizer.param_groups:
            group['lr'] = config['training']['lr'] * min(1., (step+1)/config['training']['warmup'])
        def update():
            scaler.step(optimizer)
            scaler.update()
        _, timing['optimizer_seconds'] = measure(update)
        timing.update(step=step+1, seconds=time.monotonic()-start, loss=float(loss), gradient_norm=float(norm))
        timing['peak_reserved_gib'] = [torch.cuda.max_memory_reserved(i)/2**30 for i in range(2)]
        timing['reserved_fraction'] = [torch.cuda.max_memory_reserved(i)/devices[i].total_memory for i in range(2)]
        assert max(timing['reserved_fraction']) < .9
        timings.append(timing)
        with (args.output/'timings.jsonl').open('a') as f:
            f.write(json.dumps(timing)+'\n')
        print(json.dumps(timing), flush=True)
        del tensors, loss
    changes = {n:not torch.equal(p.detach().cpu(), before[n]) for n,p in params.items()}
    (args.output/'adapter_updates.json').write_text(json.dumps(changes, indent=2)+'\n')
    if not all(changes.values()):
        raise FloatingPointError('Unchanged adapters: '+str([n for n,changed in changes.items() if not changed]))
    del optimizer, before
    for p in params.values():
        p.grad = None
    gc.collect(); torch.cuda.empty_cache()
    model.eval()

    def reference_tokens(row):
        with Image.open(row['reference']) as image:
            return reference_geometry(image.convert('RGB'), row['reference_box'], 'flux',
                                      config['data']['reference_size'])[1].size
    row = max(validation, key=reference_tokens)
    tensors, _ = condition(row)
    with torch.no_grad(), torch.random.fork_rng(devices=[0, 1]):
        negative, _ = backend.encode_text(encoder, config, {**row, 'prompt':''})
        tensors.update({'negative_'+k:v.to('cuda:0') for k,v in negative.items()})
        tensors['target_face_mask'] = torch.ones((1, args.resolution**2//256), device='cuda:0')
        noisy = torch.zeros((1,128,args.resolution//16,args.resolution//16), device='cuda:0')
        generation = {}
        for branch, name in ((False, 'native'), (True, 'branch_full_mask')):
            def cfg_pair():
                for negative_prompt in (False, True):
                    prediction = compute(backend.predict, model, tensors, noisy, sigma, config, branch, negative=negative_prompt)
                    require_finite(prediction, 'validation prediction')
            cfg_pair()  # Warmup.
            values = [measure(cfg_pair)[1] for _ in range(3)]
            generation[name+'_seconds_per_image'] = statistics.mean(values)*config['validation']['steps']
    warm = timings[1:]
    step_seconds = statistics.mean(t['seconds'] for t in warm)
    generation_seconds = 96*(generation['native_seconds_per_image']+11*generation['branch_full_mask_seconds_per_image'])
    result = {'accepted':True, 'resolution':args.resolution, 'checkpointing':args.checkpointing,
              'dtype':'float32', 'effective_batch':1, 'gpu_roles':['denoiser_and_vae','frozen_text_encoder'],
              'compute_precision':'amp_fp16_fp32_residuals_and_branch' if args.amp_fp16 else 'float32',
              'measured_updates':args.updates,
              'native_amp_vs_fp32_relative_rmse':precision_difference,
              'mean_warm_step_seconds':step_seconds, 'warm_stage_seconds':{
                  k:statistics.mean(t[k] for t in warm) for k in warm[0] if k.endswith('_seconds')},
              'training_hours_20k':step_seconds*20000/3600, **generation,
              'serial_generation_hours_all_panels':generation_seconds/3600,
              'ideal_4way_generation_hours_not_implemented':generation_seconds/4/3600,
              'peak_reserved_gib':[max(t['peak_reserved_gib'][i] for t in timings) for i in range(2)],
              'native_off_exact':True, 'finite_gradients':True, 'all_64_adapters_updated':True,
              'limitations':'128-row timing probe; no save/resume or scored panels; excludes startup/checkpoint/scoring; full-mask inference timing is conservative, not a hard bound.'}
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


def main(args):
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    cases = ((768,True,False),(768,False,False),(512,True,False),(512,False,False))
    if args.followup:
        cases = ((768,True,True),(768,False,True),(384,False,False),(384,True,False))
    for resolution, checkpointing, amp in cases:
        precision = 'amp_fp16' if amp else 'fp32'
        folder = args.output/f'{precision}_{resolution}_checkpoint{int(checkpointing)}_encoder_gpu1'
        command = [sys.executable, '-m', 'scripts.benchmark_clust_budget', '--worker',
                   '--config', str(args.config), '--output', str(folder), '--resolution', str(resolution)]
        if checkpointing:
            command += ['--checkpointing']
        if amp:
            command += ['--amp-fp16']
        command += ['--updates', str(100 if amp and checkpointing else 6)]
        with (args.output/(folder.name+'.log')).open('w') as log:
            code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        result_path = folder/'result.json'
        if code:
            error = (args.output/(folder.name+'.log')).read_text()
            reason = 'memory' if 'OutOfMemoryError' in error else 'nonfinite' if 'FloatingPointError' in error else 'worker_error'
            result = {'accepted':False,'resolution':resolution,'checkpointing':checkpointing,
                      'compute_precision':precision, 'reason':reason, 'exit_code':code}
            result_path.write_text(json.dumps(result,indent=2)+'\n')
            if result['reason'] == 'worker_error':
                raise RuntimeError(f'Inspect {folder.name}.log')
        else:
            result = json.loads(result_path.read_text())
        results.append(result)
        (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=ROOT/'configs/clust/FLUX1_cluster_4b_fp32.yaml')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--resolution', type=int, default=768)
    p.add_argument('--checkpointing', action='store_true')
    p.add_argument('--amp-fp16', action='store_true')
    p.add_argument('--updates', type=int, default=6)
    p.add_argument('--followup', action='store_true', help='Probe FP16 autocast with FP32 residuals and FP32 384px fallback')
    main(p.parse_args())
