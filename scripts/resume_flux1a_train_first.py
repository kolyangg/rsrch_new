"""User-authorized FLUX1a training first; finish deferred step-0 panel at 2k."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from ba_dit.config import ROOT
from scripts.online_face_ba import verify, write
from scripts.run_multi_id_face_ba import archive_uncheckpointed_metrics, latest_checkpoint


def main():
    if ROOT != Path('/workspace/rsrch_FLUX1abc'):
        raise RuntimeError('Controller is scoped to the existing GB10 FLUX1a run')
    run = ROOT/'runs/FLUX1a_vast9B_20261004'
    lock = (ROOT/'runs/face_flow_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config, _ = verify(run)
    assert config['name'] == 'FLUX1a_vast_9b' and config['training']['steps'] == 4000
    resume = latest_checkpoint(run, 2000)
    if resume < 2000:
        archive_uncheckpointed_metrics(run, resume)
        command = [sys.executable, '-m', 'scripts.online_face_ba', 'train',
                   '--run', str(run), '--step', '2000', '--resume', str(resume)]
        with (run/'train_2000.log').open('a') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                     stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
            write(run/'status.json', {'status': 'running', 'stage': 'train_2000',
                  'controller_pid': os.getpid(), 'child_pid': child.pid,
                  'initial_validation': 'deferred until first 2000 updates by user request'})
            code = child.wait()
        if code:
            write(run/'status.json', {'status': 'failed', 'stage': 'train_2000', 'exit_code': code})
            raise RuntimeError(f'Training failed ({code}); inspect train_2000.log')
        write(run/'train_2000.done.json', {'command': command,
              'finished': datetime.now(timezone.utc).isoformat(),
              'initial_validation_deferred_by_user': True})
    lock.close()
    # The original controller first finishes checkpoint-0 validation, then the
    # already saved 2k panel, before continuing its original serial 4k plan.
    command = [sys.executable, '-m', 'scripts.run_multi_id_face_ba',
               '--config', str(run/'resolved_config.yaml'), '--run', str(run),
               '--native-bundle', str(ROOT/'runs/native_fixed96_bundle'),
               '--id-clip-only', '--resume']
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
