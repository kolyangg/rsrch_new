"""Serial full96 validation after the 8k pilot and verified local checkpoint copy."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml
from ba_dit.config import ROOT, adapter_identity, digest, load_config
from ba_dit.data.manifest import file_hash, read_manifest
from scripts.online_face_ba import SOURCES, verify, write

PARENT = ROOT/'runs/flux9b_large_qkvo_r128_to8k'
BASELINE = ROOT/'runs/flux9b_native_fixed96_20261003'
RUN = ROOT/'runs/flux9b_8000_fixed96_20261004'


def hashes(directory):
    return {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()}


def prepare():
    from scripts.run_multi_id_face_ba import freeze_native
    config, identity = verify(PARENT)
    config['name'] = RUN.name
    config['data']['validation_manifest'] = str(ROOT/'data/validation/manual_val_96.jsonl')
    config['validation']['limit'] = 96
    rows = read_manifest(config['data']['validation_manifest'])
    assert len(rows) == 96
    if (RUN/'identity.json').exists():
        verify(RUN)
        return
    RUN.mkdir(exist_ok=True)
    (RUN/'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    freeze_native(config, BASELINE, RUN, rows)
    shutil.copyfile(BASELINE/'quality_summary.json', RUN/'native/quality_summary.json')
    checkpoint = RUN/'checkpoint-008000'
    if not checkpoint.is_symlink():
        checkpoint.symlink_to(PARENT/'checkpoint-008000', target_is_directory=True)
    sources = (*SOURCES, 'scripts/finalize_flux9b_remote.py')
    for path in sources:
        dest = RUN/'source_snapshot'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/path, dest)
    write(RUN/'identity.json', {**identity, 'config_sha256':digest(config),
        'base':adapter_identity(config), 'validation_manifest_sha256':file_hash(config['data']['validation_manifest']),
        'routing_masks_sha256':file_hash(RUN/'routing_masks.json'),
        'source_sha256':{p:file_hash(ROOT/p) for p in sources}, 'parent_run':str(PARENT)})
    write(RUN/'status.json', {'status':'waiting_for_8k_and_local_copy'})
    verify(RUN)


def main(prepare_only=False):
    prepare()
    if prepare_only:
        print('Full96 prepared: original panel, native image/mask hashes, frozen runtime verified.')
        return
    while True:
        status = json.loads((PARENT/'status.json').read_text())
        if status.get('status') == 'failed':
            raise RuntimeError('Parent training/validation failed; refusing to proceed')
        if status.get('status') == 'completed' and (RUN/'local_checkpoint_verified.json').exists():
            assert status['training_steps'] == status['last_completed_validation'] == 8000
            break
        time.sleep(60)
    checkpoint = PARENT/'checkpoint-008000'
    ack = json.loads((RUN/'local_checkpoint_verified.json').read_text())
    assert ack['verified'] and ack['sha256'] == hashes(checkpoint)
    assert json.loads((checkpoint/'manifest.json').read_text())['step'] == 8000
    lock = (ROOT/'runs/face_flow_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    os.set_inheritable(lock.fileno(), True)
    stages = [(action, [sys.executable, '-m', 'scripts.online_face_ba', action,
               '--run', str(RUN), '--step', '8000']) for action in ('infer', 'decode')]
    stages.append(('score', [str(ROOT/'envs/metrics/bin/python'), '-m', 'scripts.evaluate_metrics',
        '--validation', str(RUN/'validation-008000'), '--log-dir', str(RUN), '--global-step', '8000',
        '--ownership-boxes', str(RUN/'ownership_boxes.json')]))
    for stage, command in stages:
        receipt = RUN/f'{stage}.done.json'
        if receipt.exists():
            continue
        verify(RUN)
        write(RUN/'status.json', {'status':'running', 'stage':stage})
        with (RUN/f'{stage}.log').open('a') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    pass_fds=(lock.fileno(),))
        if result.returncode:
            write(RUN/'status.json', {'status':'failed', 'stage':stage, 'code':result.returncode})
            raise RuntimeError(f'Full96 {stage} failed')
        write(receipt, {'command':command})
    report = json.loads((RUN/'validation-008000/validation.json').read_text())
    assert len(report['samples']) == 96
    assert report['checkpoint_sha256'] == file_hash(checkpoint/'adapters.safetensors')
    audit = json.loads((RUN/'inference_audit_8000.json').read_text())
    assert audit['exact_latent_exterior'] and audit['samples'] == 96
    ready = {'status':'completed', 'step':8000, 'samples':96,
             'checkpoint_sha256':hashes(checkpoint),
             'files':{str(p.relative_to(RUN)):file_hash(p) for p in RUN.rglob('*')
                      if p.is_file() and 'checkpoint-' not in str(p.relative_to(RUN))
                      and p.name not in ('status.json', 'ready.json')}}
    write(RUN/'ready.json', ready)
    write(RUN/'status.json', {'status':'completed', 'step':8000, 'samples':96})
    print('Full96 complete; local controller must verify download and Comet before stopping.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    main(parser.parse_args().prepare_only)

