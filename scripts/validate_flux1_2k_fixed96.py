"""Validate the saved FLUX1 Vast9B 2k checkpoint against the exact 8k full96 inputs."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import yaml

from ba_dit.config import ROOT, digest
from ba_dit.data.manifest import file_hash, read_manifest
from ba_dit.logging import connect
from scripts.online_face_ba import verify, write

REFERENCE = ROOT / 'runs/flux9b_8000_fixed96_20261004'
CHECKPOINT = ROOT / 'runs/flux9b_large_qkvo_r128_to8k/checkpoint-002000'
RUN = ROOT / 'runs/FLUX1_vast9B_2000_fixed96_20261004'
COMET = '2b3eda1b1f364ecfaa584ffe273102a1'
ADAPTER_SHA = 'eb27ca1c63780a50ae090cc65c746b6a7225ea66e776400f1b5bd131606857fd'


def prepare():
    config, identity = verify(REFERENCE)
    assert json.loads((REFERENCE / 'status.json').read_text())['status'] == 'completed'
    assert json.loads((REFERENCE / 'comet_experiment.json').read_text())['experiment_key'] == COMET
    assert json.loads((CHECKPOINT / 'manifest.json').read_text())['step'] == 2000
    assert file_hash(CHECKPOINT / 'adapters.safetensors') == ADAPTER_SHA
    rows = read_manifest(config['data']['validation_manifest'])
    assert len(rows) == config['validation']['limit'] == 96
    masks = json.loads((REFERENCE / 'routing_masks.json').read_text())['samples']
    old = json.loads((REFERENCE / 'validation-008000/validation.json').read_text())
    assert [r['sample_id'] for r in rows] == [r['sample_id'] for r in old['samples']]
    for row in rows:
        key = row['sample_id']
        for suffix, field in (('.png', 'baseline_image_sha256'), ('.safetensors', 'latent_sha256')):
            assert file_hash(REFERENCE / 'native' / (key + suffix)) == masks[key][field]
    config['name'] = RUN.name
    if (RUN / 'identity.json').exists():
        actual, _ = verify(RUN)
        assert actual == config
        assert json.loads((RUN / 'comet_experiment.json').read_text())['experiment_key'] == COMET
        return config
    RUN.mkdir(exist_ok=True)
    shutil.copytree(REFERENCE / 'native', RUN / 'native', dirs_exist_ok=True)
    for name in ('routing_masks.json', 'ownership_boxes.json', 'mask_overlays.png', 'comet_experiment.json'):
        shutil.copyfile(REFERENCE / name, RUN / name)
    (RUN / 'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    link = RUN / 'checkpoint-002000'
    if not link.is_symlink():
        link.symlink_to(CHECKPOINT, target_is_directory=True)
    sources = {**identity['source_sha256'], 'scripts/validate_flux1_2k_fixed96.py': file_hash(__file__)}
    for relative in sources:
        dest = RUN / 'source_snapshot' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
    write(RUN / 'identity.json', {**identity, 'config_sha256': digest(config),
          'source_sha256': sources, 'comparison_run': str(REFERENCE)})
    write(RUN / 'validation_request.json', {
        'requested_at_utc': datetime.now(timezone.utc).isoformat(),
        'experiment': 'FLUX1 Vast9B', 'step': 2000, 'samples': 96,
        'comet_experiment_key': COMET, 'comparison_step': 8000,
        'adapter_sha256': ADAPTER_SHA, 'comparison_run': str(REFERENCE),
        'config_changes_from_8k': ['name'],
        'frozen_native_images_and_latents_verified': 96,
        'validation_manifest_sha256': identity['validation_manifest_sha256'],
        'routing_masks_sha256': identity['routing_masks_sha256'],
        'ownership_boxes_sha256': file_hash(RUN / 'ownership_boxes.json')})
    verify(RUN)
    write(RUN / 'status.json', {'status': 'prepared', 'step': 2000, 'samples': 96})
    return config


def comet_metrics(remote, step):
    return {r['metricName']: float(r['metricValue']) for r in remote.get_metrics()
            if r.get('step') is not None and int(float(r['step'])) == step}


def finish_comet(config, remote):
    summary = json.loads((RUN / 'validation-002000/quality_summary.json').read_text())
    previous = json.loads((RUN / 'comet_8k_before.json').read_text())
    expected = {'validation/' + k: v for k, v in summary['metrics'].items()}
    report = json.loads((RUN / 'validation-002000/validation.json').read_text())
    names = {'fixed96/' + row['sample_id'] for row in report['samples']}
    exp = connect(config, RUN)
    try:
        for relative in ('validation_request.json', 'inference_audit_2000.json',
                         'background_audit_2000.json', 'validation-002000/quality_summary.json',
                         'validation-002000/quality_per_image.csv'):
            exp.log_asset(str(RUN / relative), file_name='fixed96_2k/' + relative, step=2000)
    finally:
        exp.end()
    for _ in range(10):
        current, old = comet_metrics(remote, 2000), comet_metrics(remote, 8000)
        assert all(k in old and math.isclose(old[k], v, rel_tol=1e-9, abs_tol=1e-12)
                   for k, v in previous.items()), 'Previously logged 8k metrics changed'
        uploaded = {re.sub(r'(?: \(\d+\))?\.png$', '', a['fileName'])
                    for a in remote.get_asset_list() if a.get('type') == 'image'
                    and a.get('step') is not None and int(float(a['step'])) == 2000}
        if all(k in current and math.isclose(current[k], v, rel_tol=1e-7, abs_tol=1e-9)
               for k, v in expected.items()) and names <= uploaded:
            write(RUN / 'comet_verified.json', {'verified': True, 'experiment_key': COMET,
                  'step': 2000, 'images': len(names), 'metrics': expected,
                  'previous_8k_metrics_preserved': True})
            return
        time.sleep(30)
    raise RuntimeError('Comet read-back did not confirm all metrics and 96 images')


def main(prepare_only=False):
    config = prepare()
    if prepare_only:
        print('Exact full96 inputs, checkpoint, frozen runtime and existing Comet key verified.', flush=True)
        return
    lock = (ROOT / 'runs/face_flow_gpu.lock').open('a')
    write(RUN / 'status.json', {'status': 'waiting_for_gpu', 'step': 2000, 'samples': 96})
    fcntl.flock(lock, fcntl.LOCK_EX)
    os.set_inheritable(lock.fileno(), True)
    exp = connect(config, RUN)
    try:
        exp.log_asset(str(RUN / 'validation_request.json'), file_name='fixed96_2k/startup.json', step=2000)
    finally:
        exp.end()
    from comet_ml.api import API
    remote = API(cache=False).get_experiment('nikolay-2104', 'rsrch-new', COMET)
    if not (RUN / 'comet_8k_before.json').exists():
        previous = comet_metrics(remote, 8000)
        assert math.isclose(previous['validation/id_sim'], 0.2762179034431635, rel_tol=1e-7)
        write(RUN / 'comet_8k_before.json', previous)
    stages = [(action, [sys.executable, '-m', 'scripts.online_face_ba', action,
               '--run', str(RUN), '--step', '2000']) for action in ('infer', 'decode')]
    stages.append(('score', [str(ROOT / 'envs/metrics/bin/python'), '-m', 'scripts.evaluate_metrics',
        '--validation', str(RUN / 'validation-002000'), '--log-dir', str(RUN),
        '--global-step', '2000', '--ownership-boxes', str(RUN / 'ownership_boxes.json')]))
    for stage, command in stages:
        receipt = RUN / f'{stage}.done.json'
        if receipt.exists():
            continue
        verify(RUN)
        write(RUN / 'status.json', {'status': 'running', 'stage': stage, 'step': 2000, 'samples': 96})
        with (RUN / f'{stage}.log').open('a') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           pass_fds=(lock.fileno(),), check=True)
        write(receipt, {'command': command, 'completed_at_utc': datetime.now(timezone.utc).isoformat()})
    folder = RUN / 'validation-002000'
    report = json.loads((folder / 'validation.json').read_text())
    assert len(report['samples']) == 96 and report['checkpoint_sha256'] == ADAPTER_SHA
    audit = json.loads((RUN / 'inference_audit_2000.json').read_text())
    assert audit['exact_latent_exterior'] and audit['samples'] == 96
    background = json.loads((RUN / 'background_audit_2000.json').read_text())
    assert len(background) == 96 and all(row['background_max_abs'] == 0 for row in background)
    write(RUN / 'status.json', {'status': 'running', 'stage': 'verify_comet', 'step': 2000, 'samples': 96})
    finish_comet(config, remote)
    ready = {'status': 'completed', 'step': 2000, 'samples': 96, 'comet_experiment_key': COMET,
             'checkpoint_sha256': {p.name: file_hash(p) for p in CHECKPOINT.iterdir() if p.is_file()},
             'files': {str(p.relative_to(RUN)): file_hash(p) for p in RUN.rglob('*') if p.is_file()
                       and 'checkpoint-' not in str(p.relative_to(RUN))
                       and p.name not in ('status.json', 'ready.json') and p.suffix != '.log'}}
    write(RUN / 'ready.json', ready)
    write(RUN / 'status.json', {'status': 'completed', 'step': 2000, 'samples': 96,
                              'comet_verified': True})
    print('Full96 at 2k complete; Comet metrics and all 96 images verified. GB10 remains running.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    try:
        main(parser.parse_args().prepare_only)
    except Exception as error:
        if RUN.exists():
            write(RUN / 'status.json', {'status': 'failed', 'error': str(error), 'step': 2000})
        raise
