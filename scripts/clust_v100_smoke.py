"""Named single-V100 two-update probe; never substitutes for DDP admission."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from ba_dit.config import ROOT, load_config
from ba_dit.logging import connect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', choices=['float16', 'float32'])
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID') or torch.cuda.device_count() != 1:
        raise RuntimeError('Requires one Slurm-allocated V100')
    gpu = torch.cuda.get_device_properties(0)
    if 'V100' not in gpu.name or gpu.total_memory < 30*10**9 or 'sm_70' not in torch.cuda.get_arch_list():
        raise RuntimeError('Unexpected GPU or unsupported CUDA build')
    run = ROOT/'runs'/f'clust_v100_two_update_smoke_{os.environ["SLURM_JOB_ID"]}'
    run.mkdir(exist_ok=False)
    config = load_config(ROOT/'configs/clust/flux4b_2v100.yaml')
    if args.dtype:
        config['model']['dtype'] = args.dtype
    original = Path(config['data']['train_manifest'])
    lines = original.read_text().splitlines()[:8]
    assert len(lines) == 8
    manifest = run/'train_pairs.jsonl'
    manifest.write_text('\n'.join(lines)+'\n')  # Cluster paths in these rows are absolute.
    config['name'] = f'flux4b_clust_v100_two_update_smoke_{config["model"]["dtype"]}_8pairs'
    config['data']['train_manifest'] = str(manifest)
    config['training'].update(world_size=1, steps=2, grad_accum=1)
    path = run/'resolved_config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    experiment = connect(config, run, name=config['name'])
    experiment.log_parameters({'scope':'two optimizer updates; no quality panel or DDP qualification',
                               'slurm_job_id':os.environ['SLURM_JOB_ID'], 'gpu':gpu.name,
                               'training_pairs':8, 'conditioning_probe':'first original fixed96 item'})
    try:
        subprocess.run([sys.executable, '-m', 'ba_dit.cli', 'precompute', '--config', str(path),
                        '--split','validation','--limit','1'], cwd=ROOT, check=True)
        subprocess.run([sys.executable, '-m', 'scripts.check_online_face_ba', '--config',str(path),
                        '--output',str(run),'--parity'], cwd=ROOT, check=True)
        result = json.loads((run/'native_checks.json').read_text())
        experiment.log_metrics({k:result[k] for k in ('peak_reserved_gib','reserved_fraction',
                                                     'trained_prediction_change')}, step=2)
        experiment.log_asset(str(run/'native_checks.json'))
        print(json.dumps({'status':'passed','optimizer_updates':2,'run':str(run),
                          'peak_reserved_gib':result['peak_reserved_gib']}), flush=True)
    finally:
        experiment.end()


if __name__ == '__main__':
    main()
