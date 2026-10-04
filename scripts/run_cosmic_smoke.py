"""Short online Cosmic hardware proof: real gradients/resume, no image panel."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from ba_dit.config import ROOT, load_config
from ba_dit.data.manifest import file_hash, read_manifest, assert_disjoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/FLUX1_vast_4b_cosmic_smoke.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    if config['data'].get('conditioning') != 'online':
        raise ValueError('This short hardware probe requires online conditioning')
    run = args.run.resolve()
    run.parent.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'runs/face_flow_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = read_manifest(config['data']['train_manifest'], training=True)
    assert_disjoint(rows, read_manifest(config['data']['validation_manifest']))
    run.mkdir(exist_ok=False)
    path = run/'resolved_config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    sources = list((ROOT/'ba_dit').rglob('*.py')) + [Path(__file__), ROOT/'scripts/check_online_face_ba.py']
    hashes = {}
    for source in sources:
        source = source.resolve()
        relative = source.relative_to(ROOT)
        saved = run/'source'/relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, saved)
        hashes[str(relative)] = file_hash(source)
    metadata = {'purpose':'Cosmic hardware smoke; not a full-data or image-quality result',
                'training_pairs':len(rows), 'image_validation':'skipped at user request for fast hardware proof',
                'source_sha256':hashes, 'manifest_sha256':file_hash(config['data']['train_manifest'])}
    (run/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    from ba_dit.logging import connect
    experiment = connect(config, run, name=run.name)
    if experiment:
        experiment.log_parameters({'hardware_smoke':True, 'training_pairs':len(rows), 'image_validation':'none'})
        experiment.end()

    def call(phase, module, *arguments):
        command = [sys.executable, '-m', module, *map(str, arguments)]
        (run/'status.json').write_text(json.dumps({'phase':phase,'command':command})+'\n')
        subprocess.run(command, cwd=ROOT, check=True, pass_fds=(lock.fileno(),))
        with (run/'completed.jsonl').open('a') as stream:
            stream.write(json.dumps({'phase':phase,'command':command})+'\n')

    # Only one cached validation input is needed to check live encoding parity.
    call('one_input_cache', 'ba_dit.cli', 'precompute', '--config',path,'--split','validation','--limit',1)
    admission = run/'admission'
    admission.mkdir()
    call('native_parity_and_two_updates', 'scripts.check_online_face_ba', '--config',path,'--output',admission,'--parity')
    base = ('_train-worker','--config',path,'--mode','branch_only','--output-dir',run)
    call('training_first_two', 'ba_dit.cli', *base,'--until',2)
    checkpoint = (run/'latest_checkpoint.txt').read_text().strip()
    call('resumed_training', 'ba_dit.cli', *base,'--until',config['training']['steps'],'--resume',checkpoint)
    (run/'status.json').write_text(json.dumps({'phase':'completed','steps':config['training']['steps']})+'\n')


if __name__ == '__main__':
    main()
