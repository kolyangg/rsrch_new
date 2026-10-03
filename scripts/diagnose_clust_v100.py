"""Locate nonfinite native activations on V100 without taking optimizer steps."""
import argparse
import json
import os
from pathlib import Path
import torch

from ba_dit import adapters
from ba_dit.config import load_config
from ba_dit.data.conditioning import TrainingConditioner
from ba_dit.data.manifest import read_manifest
from ba_dit.nn.masked_face_attention import training_mask
from ba_dit.runtime import backend_module


def stats(value):
    value = value.detach()
    finite = torch.isfinite(value)
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'finite': bool(finite.all()), 'nonfinite': int((~finite).sum()),
            'finite_absmax': float(value[finite].abs().max()) if finite.any() else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dtype', choices=['float16', 'float32'], required=True)
    args = p.parse_args()
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count() == 1
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    config = load_config(args.config)
    config['model']['dtype'] = args.dtype
    torch.manual_seed(config['training']['seed'])
    backend = backend_module(config)
    model = backend.load_transformer(config)
    rows = read_manifest(config['data']['train_manifest'], training=True)
    row = max(rows, key=lambda r: float(training_mask(r, config).sum()))
    conditioner = TrainingConditioner(config, backend)
    tensors, _ = conditioner(row)
    tensors['target_face_mask'] = training_mask(row, config).cuda()
    noisy = torch.randn_like(tensors['target_latent'])
    sigma = torch.tensor([.5], device='cuda', dtype=noisy.dtype)
    report = {'dtype': args.dtype, 'sample': row['sample_id'], 'optimizer_updates': 0,
              'inputs': {k: stats(v) for k, v in tensors.items() if v.is_floating_point()}}
    first_bad = []
    def hook(name):
        def inspect(module, inputs, output):
            values = [output] if isinstance(output, torch.Tensor) else list(output) if isinstance(output, tuple) else []
            if not first_bad:
                for value in values:
                    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                        first_bad.append({'module': name, 'type': type(module).__name__,
                                          'inputs': [stats(v) for v in inputs if isinstance(v, torch.Tensor)],
                                          'output': stats(value)})
                        print(json.dumps({'first_nonfinite': first_bad[-1]}), flush=True)
        return inspect
    handles = [module.register_forward_hook(hook(name)) for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) or name.startswith(('double_blocks.', 'single_blocks.')) and name.count('.') == 1]
    try:
        with torch.no_grad():
            native = backend.predict(model, tensors, noisy, sigma, config, False)
            report['native'] = stats(native)
            for handle in handles:
                handle.remove()
            repeated = backend.predict(model, tensors, noisy, sigma, config, False)
            report['repeat_exact'] = torch.equal(native, repeated)
            adapters.install(model, config, 'branch_only')
            off = backend.predict(model, tensors, noisy, sigma, config, False)
            report['off'] = stats(off)
            report['native_off_exact'] = torch.equal(native, off)
            report['finite_difference_absmax'] = float((native-off).abs().max()) if torch.isfinite(native).all() and torch.isfinite(off).all() else None
    finally:
        report['first_nonfinite'] = first_bad
        report['peak_reserved_gib'] = torch.cuda.max_memory_reserved()/2**30
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
