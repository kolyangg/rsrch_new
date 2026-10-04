"""Serial two-update benchmarks at equal effective batch; preserve every result."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from ba_dit.config import ROOT, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for batch, checkpointing in [(1,True),(1,False),(2,False),(4,False)]:
        folder = args.output/f'batch{batch}_checkpoint{int(checkpointing)}'
        if not (folder/'result.json').exists():
            folder.mkdir(exist_ok=False)
            config = load_config(args.config)
            config['data']['train_manifest'] = str(args.manifest.resolve())
            config['logging']['enabled'] = False
            config['training'].update(steps=2, checkpoint_every=2, microbatch_size=batch,
                grad_accum=8//batch, gradient_checkpointing=checkpointing)
            path = folder/'resolved_config.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            with (folder/'worker.log').open('w') as log:
                worker = ('import torch; torch.cuda.set_per_process_memory_fraction('
                          f'{config["training"]["max_reserved_fraction"]}); '
                          'from ba_dit.cli import main; main()')
                code = subprocess.run([sys.executable,'-c',worker,'_train-worker',
                    '--config',str(path),'--output-dir',str(folder),'--until','2'],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:
                error = (folder/'worker.log').read_text()
                if not any(x in error for x in ('OutOfMemoryError','Memory acceptance failed')):
                    raise RuntimeError(f'Benchmark failed; inspect {folder}/worker.log')
                result = {'microbatch':batch,'checkpointing':checkpointing,'accepted':False,'reason':'memory'}
            else:
                metrics = [json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
                result = {'microbatch':batch,'checkpointing':checkpointing,'accepted':True,
                    'warm_seconds':metrics[0]['train/seconds'], 'seconds':metrics[-1]['train/seconds'],
                    'images_per_second':8/metrics[-1]['train/seconds'],
                    'peak_reserved_gib':max(m['hardware/peak_reserved_gib'] for m in metrics),
                    'gradient_norms':[m['train/gradient_norm'] for m in metrics]}
            (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        result = json.loads((folder/'result.json').read_text())
        results.append(result)
        print(json.dumps(result),flush=True)
    chosen = max((r for r in results if r['accepted']), key=lambda r:r['images_per_second'])
    config = load_config(args.config)
    config['training'].update(microbatch_size=chosen['microbatch'],grad_accum=8//chosen['microbatch'],
                              gradient_checkpointing=chosen['checkpointing'])
    (args.output/'selected_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    (args.output/'summary.json').write_text(json.dumps({'results':results,'selected':chosen},indent=2)+'\n')


if __name__ == '__main__':
    main()
