"""Pretrained masked Q/K/V/O checks and exact fresh-process optimizer replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from contextlib import nullcontext

import torch
import yaml
from safetensors.torch import load_file

from ba_dit import adapters
from ba_dit.config import ROOT, load_config
from ba_dit.data.cache import load_pair
from ba_dit.data.conditioning import TrainingConditioner
from ba_dit.data.manifest import read_manifest
from ba_dit.nn.masked_face_attention import training_mask, parameter_count
from ba_dit.nn.batched_face_attention import collate, reference_batch, training_loss as batch_loss
from ba_dit.runtime import backend_module
from ba_dit.precision import make_scaler, device_memory, mixed


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def frozen_hash(model):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            h.update(name.encode())
            h.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def parity(config, out):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.getenv('OMP_NUM_THREADS', '8')))
    torch.manual_seed(config['training']['seed'])
    backend = backend_module(config)
    model = backend.load_transformer(config)
    rows = read_manifest(config['data']['train_manifest'], training=True)
    # Include the largest face support to qualify the worst routing workload.
    row = max(rows, key=lambda r: float(training_mask(r, config).sum()))
    conditioner = TrainingConditioner(config, backend)
    identity_objective = None
    identity_checks = {}
    if config['training'].get('identity_loss', {}).get('weight', 0) > 0:
        from ba_dit.nn.online_identity_loss import OnlineIdentityObjective
        identity_objective = OnlineIdentityObjective(config, conditioner.vae)
        eligible = [r for r in rows if identity_objective.rows.get(r['sample_id'],{}).get('accepted',False)]
        row = max(eligible, key=lambda r: float(training_mask(r, config).sum()))
        import numpy as np
        probe = np.load(Path(config['data']['identity_supervision'])/'arcface_parity.npz')
        with torch.no_grad():
            expected = torch.from_numpy(probe['embedding']).cuda()
            actual = identity_objective.recognizer(torch.from_numpy(probe['input']).cuda())
            difference = float((actual-expected).abs().max())
            assert difference < .01, 'Differentiable ArcFace differs from frozen ONNX evaluator'
        identity_checks['arcface_onnx_parity_max_abs'] = difference
    live_check = {}
    if conditioner.online:
        # Compare against an independently prepared validation cache, not a
        # serialization of this call. No training cache is created for this test.
        probe = read_manifest(config['data']['validation_manifest'])[0]
        cached, _ = load_pair(config, probe, 'cuda')
        cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state_all()
        live, _ = conditioner(probe)
        assert cached.keys() == live.keys() and all(torch.equal(value,live[name]) for name,value in cached.items())
        assert torch.equal(cpu_rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
        live_check = {'live_cached_conditioning_exact':True,'conditioning_preserves_training_rng':True}
        del cached, live
    tensors, _ = conditioner(row)
    tensors['target_face_mask'] = training_mask(row, config).cuda()
    batch = config['training'].get('microbatch_size', 1)
    if batch > 1:
        pairs = [tensors]
        other = [r for r in rows if r['sample_id'] != row['sample_id']][:batch-1]
        if len(other) != batch-1:
            raise ValueError('Admission needs distinct examples for the full microbatch')
        for item in other:
            pair, _ = conditioner(item)
            pair['target_face_mask'] = training_mask(item, config).cuda()
            pairs.append(pair)
        tensors = collate(pairs)
        del pairs, pair
    def read_context():
        return reference_batch(tensors['reference_masks'],config['branch']['max_reference_keys']) if batch > 1 else nullcontext()
    noisy = torch.randn_like(tensors['target_latent'])
    sigma = torch.tensor([.5], device='cuda', dtype=noisy.dtype)
    with torch.no_grad():
        native = backend.predict(model, tensors, noisy, sigma, config, False)
    inventory = adapters.install(model, config, 'branch_only')
    params = {n:p for n,p in model.named_parameters() if p.requires_grad}
    assert len(params) == 64 and all('.reference_branch.' in n for n in params)
    assert sum(p.numel() for p in params.values()) == parameter_count(config)
    before = {n:p.detach().cpu().clone() for n,p in params.items()}
    base_hash = frozen_hash(model)
    with torch.no_grad(), read_context():
        off = backend.predict(model, tensors, noisy, sigma, config, False)
        assert torch.equal(native, off)
        zero_mask = {**tensors, 'target_face_mask': torch.zeros_like(tensors['target_face_mask'])}
        empty = backend.predict(model, zero_mask, noisy, sigma, config, True)
        assert torch.equal(native, empty)
        initial = backend.predict(model, tensors, noisy, sigma, config, True)
        assert torch.isfinite(initial).all() and not torch.equal(native, initial)
    if config['training']['gradient_checkpointing']:
        model.enable_gradient_checkpointing()
    model.train()
    optimizer = torch.optim.AdamW(adapters.groups(model, config['training']['lr']), weight_decay=0.)
    scaler = make_scaler(config)
    torch.cuda.reset_peak_memory_stats()
    if identity_objective is not None:
        # Force one decoded auxiliary gradient independently of random sigma
        # admission draws, using valid training-target landmarks only.
        id_sigma = sigma.new_tensor([.4])
        id_noisy = .6*tensors['target_latent'] + .4*noisy
        prediction = backend.predict(model,tensors,id_noisy,id_sigma,config,True)
        id_loss = identity_objective.loss(prediction,id_noisy,id_sigma,row,0,force=True)
        id_loss.backward()
        id_norm = sum(float(p.grad.float().square().sum()) for p in params.values() if p.grad is not None)**.5
        assert id_norm > 0 and all(torch.isfinite(p.grad).all() for p in params.values() if p.grad is not None)
        assert all(p.grad is None for p in conditioner.vae.parameters())
        identity_checks.update(identity_gradient_norm=id_norm, weighted_identity_loss=float(id_loss.detach()))
        optimizer.zero_grad(set_to_none=True)
        del prediction, id_loss
        prediction = backend.predict(model,tensors,id_noisy,id_sigma,config,True)
        mask = tensors['target_face_mask'].reshape(tensors['target_latent'].shape[0],1,*id_noisy.shape[-2:])
        flow_probe = ((prediction.float()-(noisy-tensors['target_latent']).float()).square()*mask).sum()/(mask.sum()*id_noisy.shape[1])
        flow_probe.backward()
        flow_norm = sum(float(p.grad.float().square().sum()) for p in params.values() if p.grad is not None)**.5
        assert flow_norm > 0
        identity_checks.update(flow_gradient_norm_at_id_sigma=flow_norm,
                               weighted_identity_to_flow_gradient_ratio=id_norm/flow_norm)
        optimizer.zero_grad(set_to_none=True)
        del prediction, flow_probe, id_noisy
    gradients = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with read_context():
            extra = {'identity_objective':identity_objective, 'row':row, 'step':0} if identity_objective else {}
            loss = batch_loss(backend,model,tensors,config) if batch > 1 else backend.training_loss(model,tensors,config,True,**extra)
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
        gradients.append({n:float(p.grad.norm()) for n,p in params.items()})
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1., error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
    assert all(v > 0 for n,v in gradients[0].items() if n.endswith('.b'))
    assert all(v > 0 for v in gradients[1].values())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert frozen_hash(model) == base_hash
    changes = {n:float((p.detach().cpu()-before[n]).norm()) for n,p in params.items()}
    assert all(changes.values())
    with torch.no_grad(), read_context():
        trained = backend.predict(model, tensors, noisy, sigma, config, True)
        off = backend.predict(model, tensors, noisy, sigma, config, False)
    assert torch.equal(native, off) and not torch.equal(initial, trained)
    stress = None
    if mixed(config) or config['branch'].get('reference_bank') == 'isolated_image':
        # Qualify the largest real reference grid with maximal routing support.
        # This is a memory/numerics stress probe, not a changed training mask.
        import math
        from PIL import Image
        sizes = {}
        def reference_tokens(item):
            path = item['reference']
            if path not in sizes:
                with Image.open(path) as image:
                    width, height = image.size
                scale = min(1., math.sqrt(config['data']['reference_size']**2/(width*height)))
                sizes[path] = (int(width*scale)//16)*(int(height*scale)//16)
            return sizes[path]
        eligible = [r for r in rows if identity_objective.rows.get(r['sample_id'],{}).get('accepted',False)] if identity_objective else rows
        largest = max(eligible, key=reference_tokens)
        full, _ = conditioner(largest)
        full['reference_mask'] = torch.ones_like(full['reference_mask'])
        full['target_face_mask'] = torch.ones_like(training_mask(largest, config)).cuda()
        assert full['reference_tokens'].shape[1] == reference_tokens(largest)
        optimizer.zero_grad(set_to_none=True)
        extra = {'identity_objective':identity_objective, 'row':largest, 'force_identity':True} if identity_objective else {}
        stress_loss = backend.training_loss(model, full, config, True, **extra)
        scaler.scale(stress_loss).backward()
        scaler.unscale_(optimizer)
        assert torch.isfinite(stress_loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        stress = {'sample':largest['sample_id'], 'reference_tokens':reference_tokens(largest),
                  'target_queries':full['target_face_mask'].numel(), 'full_routing_masks':True,
                  'finite_gradients':True, 'optimizer_update':False}
        del full, stress_loss
    per_device = device_memory(config)
    fraction = max(item['reserved_fraction'] for item in per_device.values())
    assert fraction < config['training']['max_reserved_fraction']
    write(out/'native_checks.json', {'sample':row['sample_id'], 'trainable_parameters':sum(p.numel() for p in params.values()),
        'native_off_exact':True, 'zero_mask_exact':True, 'all_frozen_parameters_exact':True,
        'frozen_sha256':base_hash, 'trainable_tensors':len(params), 'first_two_gradients':gradients,
        'parameter_changes':changes, 'branch_on_off_mean_abs':float((initial-native).abs().float().mean()),
        'trained_prediction_change':float((trained-initial).abs().float().mean()),
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30, 'reserved_fraction':fraction,
        'inventory':inventory, 'device_memory':per_device, 'largest_layout_stress':stress, **live_check, **identity_checks})
    print('Pretrained parity, gradients, frozen equality and memory passed', flush=True)


def main(args):
    config = load_config(args.config)
    out = args.output.resolve()
    if args.parity:
        parity(config, out)
        return
    out.mkdir(parents=True, exist_ok=False)
    config['logging']['enabled'] = False
    config['training'].update(steps=2, checkpoint_every=1)
    path = out/'resolved_config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    commands = [[sys.executable, '-m', 'scripts.check_online_face_ba', '--config', str(path), '--output', str(out), '--parity']]
    for name, until, resume in [('continuous',2,None), ('restarted',1,None), ('restarted',2,'checkpoint-000001')]:
        folder = out/name
        folder.mkdir(exist_ok=True)
        (folder/'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        command = [sys.executable, '-m', 'ba_dit.cli', '_train-worker', '--config', str(path),
                   '--mode','branch_only','--output-dir',str(folder),'--until',str(until)]
        if config['training'].get('world_size', 1) > 1:
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       '--nproc-per-node=2', *command[1:]]
        if resume: command += ['--resume',str(folder/resume)]
        commands.append(command)
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    a = load_file(out/'continuous/checkpoint-000002/adapters.safetensors')
    b = load_file(out/'restarted/checkpoint-000002/adapters.safetensors')
    assert a.keys() == b.keys() and all(torch.equal(v,b[n]) for n,v in a.items())
    # Compare Adam moments, scheduler, data cursor and all RNG states as well.
    a = torch.load(out/'continuous/checkpoint-000002/training_state.pt', weights_only=True, map_location='cpu')
    b = torch.load(out/'restarted/checkpoint-000002/training_state.pt', weights_only=True, map_location='cpu')
    def same(x,y):
        if isinstance(x,torch.Tensor): return torch.equal(x,y)
        if isinstance(x,dict): return x.keys()==y.keys() and all(same(v,y[k]) for k,v in x.items())
        if isinstance(x,(tuple,list)): return len(x)==len(y) and all(same(u,v) for u,v in zip(x,y))
        return x==y
    assert same(a,b)
    if mixed(config):
        assert a['grad_scaler']
        if config['training'].get('world_size', 1) > 1:
            assert a['distributed']['scaler'] == a['grad_scaler']
            assert len(a['distributed']['ranks']) == config['training']['world_size']
        else:
            assert len(a['cuda_rng']) == (2 if config['data'].get('encoder_device') == 'cuda:1' else 1)
    write(out/'resume_parity.json', {'exact_parameters':True,'exact_optimizer_scheduler_rng_cursor':True,
                                   'exact_gradient_scaler':mixed(config), 'cuda_rng_devices':len(a['cuda_rng']),
                                   'optimizer_updates':2,'microbatch':config['training'].get('microbatch_size',1),
                                   'world_size':config['training'].get('world_size', 1),
                                   'gradient_accumulation':config['training']['grad_accum']})
    print('Fresh-process optimizer/RNG replay passed', flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parity',action='store_true')
    main(p.parse_args())
