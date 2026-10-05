"""Plan, prepare training-only ID labels, or launch one isolated FLUX1 follow-up."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from ba_dit.config import ROOT, load_config
from ba_dit.data.manifest import file_hash
from ba_dit.nn.masked_face_attention import parameter_count


def main(args):
    config_path = ROOT / 'configs' / f'{args.experiment}_vast_9b.yaml'
    config = load_config(config_path)
    run = (args.run or ROOT / 'runs' / config['name']).resolve()
    bundle = (args.native_bundle or ROOT / 'runs/flux9b_8000_fixed96_20261004').resolve()
    identity = config['training'].get('identity_loss', {}).get('weight', 0) > 0
    plan = {'experiment': args.experiment, 'config': str(config_path), 'run': str(run),
        'initialization': f"fresh rank{config['branch']['rank']} Q/K/V/O BA; no historical FLUX1 checkpoint resume",
        'priority': {'FLUX1a':1, 'FLUX1b':2, 'FLUX1c':3}[args.experiment],
        'reference_bank': 'frozen reference-image-only pass, no target or prompt tokens, current sigma',
        'token_ownership': config['branch']['token_ownership'], 'identity_loss': config['training'].get('identity_loss'),
        'trainable_parameters': parameter_count(config), 'training_updates': 4000, 'validation_steps': [0,2000,4000],
        'native_bundle': str(bundle), 'native_attention_changed': False,
        'stages': ['CPU identity supervision preparation']*int(identity) +
            ['pretrained parity, gradients, memory and exact resume admission',
             'reuse fixed96 native bundle', 'initial validation', 'serial training and fixed96 scoring'],
        'launch_requested': args.action == 'run'}
    print(json.dumps(plan, indent=2), flush=True)
    if args.action == 'plan':
        return
    if ROOT.as_posix() in {'/workspace/rsrch_9b_8k','/workspace/rsrch_9b'}:
        raise RuntimeError('Use a separate new checkout; historical FLUX1 deployment sources are immutable')
    if args.action == 'prepare':
        if identity:
            metrics = Path(os.getenv('BA_ENVS_DIR', ROOT/'envs'))/'metrics/bin/python'
            subprocess.run([str(metrics), '-m', 'scripts.prepare_flux1_identity', '--config', str(config_path), '--scheduled-only'],
                           cwd=ROOT, check=True)
        return
    if identity:
        from ba_dit.nn.online_identity_loss import supervision_identity
        supervision_identity(config)  # Clear failure before loading any CUDA model.
    record = json.loads((bundle/'routing_masks.json').read_text())
    assert record['source_signature']['panel_sha256'] == file_hash(config['data']['validation_manifest'])
    assert len(record['samples']) == 96
    for key, row in record['samples'].items():
        for suffix, field in (('.png','baseline_image_sha256'),('.safetensors','latent_sha256')):
            assert file_hash(bundle/'native'/(key+suffix)) == row[field]
    command = [sys.executable, '-m', 'scripts.run_multi_id_face_ba', '--config', str(config_path),
               '--run', str(run), '--native-bundle', str(bundle), '--id-clip-only']
    if args.resume:
        command.append('--resume')
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=('FLUX1a','FLUX1b','FLUX1c'), required=True)
    parser.add_argument('--action', choices=('plan','prepare','run'), default='plan')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--native-bundle', type=Path)
    parser.add_argument('--resume', action='store_true')
    main(parser.parse_args())
