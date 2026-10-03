"""Pretrained masked Q/K/V/O checks and exact fresh-process optimizer replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml
from safetensors.torch import load_file

from ba_dit import adapters
from ba_dit.config import ROOT, load_config
from ba_dit.data.cache import load_pair
from ba_dit.data.manifest import read_manifest
from ba_dit.nn.masked_face_attention import training_mask
from ba_dit.runtime import backend_module


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
    torch.set_num_threads(8)
    torch.manual_seed(config['training']['seed'])
    backend = backend_module(config)
    model = backend.load_transformer(config)
    rows = read_manifest(config['data']['train_manifest'], training=True)
    # Include the largest face support to qualify the worst routing workload.
    row = max(rows, key=lambda r: float(training_mask(r, config).sum()))
    tensors, _ = load_pair(config, row, 'cuda')
    tensors['target_face_mask'] = training_mask(row, config).cuda()
    noisy = torch.randn_like(tensors['target_latent'])
    sigma = torch.tensor([.5], device='cuda', dtype=noisy.dtype)
    with torch.no_grad():
        native = backend.predict(model, tensors, noisy, sigma, config, False)
    inventory = adapters.install(model, config, 'branch_only')
    params = {n:p for n,p in model.named_parameters() if p.requires_grad}
    assert len(params) == 64 and all('.reference_branch.' in n for n in params)
    assert sum(p.numel() for p in params.values()) == 25165824
    before = {n:p.detach().cpu().clone() for n,p in params.items()}
    base_hash = frozen_hash(model)
    with torch.no_grad():
        off = backend.predict(model, tensors, noisy, sigma, config, False)
        assert torch.equal(native, off)
        zero_mask = {**tensors, 'target_face_mask': torch.zeros_like(tensors['target_face_mask'])}
        empty = backend.predict(model, zero_mask, noisy, sigma, config, True)
        assert torch.equal(native, empty)
        initial = backend.predict(model, tensors, noisy, sigma, config, True)
        assert torch.isfinite(initial).all() and not torch.equal(native, initial)
    model.enable_gradient_checkpointing()
    model.train()
    optimizer = torch.optim.AdamW(adapters.groups(model, config['training']['lr']), weight_decay=0.)
    torch.cuda.reset_peak_memory_stats()
    gradients = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = backend.training_loss(model, tensors, config, True)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
        gradients.append({n:float(p.grad.norm()) for n,p in params.items()})
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1., error_if_nonfinite=True)
        optimizer.step()
    assert all(v > 0 for n,v in gradients[0].items() if n.endswith('.b'))
    assert all(v > 0 for v in gradients[1].values())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert frozen_hash(model) == base_hash
    changes = {n:float((p.detach().cpu()-before[n]).norm()) for n,p in params.items()}
    assert all(changes.values())
    with torch.no_grad():
        trained = backend.predict(model, tensors, noisy, sigma, config, True)
        off = backend.predict(model, tensors, noisy, sigma, config, False)
    assert torch.equal(native, off) and not torch.equal(initial, trained)
    fraction = torch.cuda.max_memory_reserved()/torch.cuda.get_device_properties(0).total_memory
    assert fraction < config['training']['max_reserved_fraction']
    write(out/'native_checks.json', {'sample':row['sample_id'], 'trainable_parameters':sum(p.numel() for p in params.values()),
        'native_off_exact':True, 'zero_mask_exact':True, 'all_frozen_parameters_exact':True,
        'frozen_sha256':base_hash, 'trainable_tensors':len(params), 'first_two_gradients':gradients,
        'parameter_changes':changes, 'branch_on_off_mean_abs':float((initial-native).abs().float().mean()),
        'trained_prediction_change':float((trained-initial).abs().float().mean()),
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30, 'reserved_fraction':fraction,
        'inventory':inventory})
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
    write(out/'resume_parity.json', {'exact_parameters':True,'exact_optimizer_scheduler_rng_cursor':True,
                                   'optimizer_updates':2,'microbatch':1,'gradient_accumulation':4})
    print('Fresh-process optimizer/RNG replay passed', flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parity',action='store_true')
    main(p.parse_args())
