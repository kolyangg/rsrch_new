"""Serial operational continuation after the unchanged FLUX1a 4k controller."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/FLUX1a_vast9B_20261004'


def continuation_config(run, config):
    from ba_dit.config import digest
    from ba_dit.continuation import verify_extension
    request = json.loads((run/'continuation_6000.json').read_text())
    assert request['target'] == 6000 and request['original_config_digest'] == digest(config)
    extended = copy.deepcopy(config)
    extended['training']['steps'] = 6000
    checkpoint = run/'checkpoint-004000'
    if not checkpoint.exists():
        checkpoint = run/'checkpoint-002500'  # CPU-only qualification before4k exists.
    manifest = json.loads((checkpoint/'manifest.json').read_text())
    verify_extension(checkpoint, extended, manifest['config_sha256'])
    from scripts.extend_flux1a_labels import apply
    return apply(extended, run)


def main(check_only=False):
    if ROOT != Path('/workspace/rsrch_FLUX1abc'):
        raise RuntimeError('Continuation is scoped to the existing GB10 run')
    if check_only:
        from scripts.online_face_ba import verify
        original, _ = verify(RUN)
        extended = continuation_config(RUN, original)
        assert extended['training']['steps'] == 6000 and original['training']['steps'] == 4000
        print('Qualified: only training.steps extends4000 to6000; frozen source guards pass', flush=True)
        return
    print('Queued6k continuation: waiting for completed4k training and validation', flush=True)
    while True:
        result = subprocess.run(['supervisorctl', 'status', 'rsrch_flux1a'],
                                capture_output=True, text=True, timeout=15).stdout.split()
        state = json.loads((RUN/'status.json').read_text())
        parent_done = (RUN/'summarize_4000.done.json').exists()
        if parent_done and len(result) > 1 and result[1] in {'EXITED', 'STOPPED'}:
            break
        if len(result) > 1 and result[1] in {'FATAL', 'STOPPED', 'EXITED'}:
            raise RuntimeError(f'Parent controller is {result[1]} before completed4k validation: {state}')
        time.sleep(15)
    from scripts.online_face_ba import verify, write
    from scripts.run_multi_id_face_ba import latest_checkpoint, archive_uncheckpointed_metrics
    config, _ = verify(RUN)
    import yaml
    lock = (ROOT/'runs/face_flow_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.set_inheritable(lock.fileno(), True)
    def call(stage, command):
        receipt = RUN/f'{stage}.done.json'
        if receipt.exists():
            return
        verify(RUN)
        print(f'{stage}: starting', flush=True)
        with (RUN/f'{stage}.log').open('a') as log:
            child = subprocess.Popen(list(map(str, command)), cwd=ROOT, stdout=log,
                                     stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
            write(RUN/'status.json', {'status': 'running', 'stage': stage,
                                     'controller_pid': os.getpid(), 'child_pid': child.pid})
            code = child.wait()
        if code:
            write(RUN/'status.json', {'status': 'failed', 'stage': stage, 'exit_code': code})
            raise RuntimeError(f'{stage} failed; inspect its log')
        write(receipt, {'command': list(map(str, command)),
                        'finished': datetime.now(timezone.utc).isoformat()})
    from scripts.extend_flux1a_labels import seed, verify as verify_labels
    destination = seed(config)
    preparation = copy.deepcopy(config)
    preparation['training']['steps'] = 6000
    preparation_path = RUN/'identity_preparation_6000.yaml'
    preparation_path.write_text(yaml.safe_dump(preparation, sort_keys=False))
    call('prepare_identity_6000', [ROOT/'envs/metrics/bin/python', '-m', 'scripts.prepare_flux1_identity',
                                 '--config', preparation_path, '--output', destination,
                                 '--workers', '4', '--threads', '2', '--scheduled-only'])
    labels = verify_labels(RUN, config)
    print(f'Identity extension verified: {labels["original_rows_exact"]} original rows exact; '
          f'{labels["extended_rows"]} rows cover6000 updates', flush=True)
    extended = continuation_config(RUN, config)
    (RUN/'resolved_config_6000.yaml').write_text(yaml.safe_dump(extended, sort_keys=False))
    worker = [sys.executable, '-m', 'scripts.flux1a_live', '--worker', 'scripts.online_face_ba']
    if not (RUN/'train_6000.done.json').exists():
        resume = latest_checkpoint(RUN, 6000)
        if resume < 6000:
            archive_uncheckpointed_metrics(RUN, resume)
            call('train_6000', worker+['train', '--run', RUN, '--step', '6000', '--resume', str(resume)])
    for action in ('infer', 'decode'):
        call(f'{action}_6000', worker+[action, '--run', RUN, '--step', '6000'])
    call('score_6000', [ROOT/'envs/metrics/bin/python', '-m', 'scripts.flux1a_live', '--worker',
                       'scripts.evaluate_metrics', '--validation', RUN/'validation-006000',
                       '--log-dir', RUN, '--global-step', '6000',
                       '--ownership-boxes', RUN/'ownership_boxes.json'])
    call('summarize_6000', worker+['summarize', '--run', RUN, '--step', '6000'])
    write(RUN/'status.json', {'status': 'completed', 'training_steps': 6000,
                              'last_completed_validation': 6000})
    print('Completed6k training and fixed96 scoring; publisher verifies Comet delivery', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    main(parser.parse_args().check)
