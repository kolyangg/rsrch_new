"""Download complete4k/6k checkpoints from the existing GB10 and verify every file."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from scripts.finalize_flux9b_local import Transport

REMOTE = '/workspace/rsrch_FLUX1abc/runs/FLUX1a_vast9B_20261004'
DESTINATION = Path('/mnt/c/Users/ogure/FLUX1a_checkpoints/FLUX1a_vast9B_20261004')
STATE = Path(__file__).resolve().parents[1]/'runs/FLUX1a_checkpoint_transfers'
FILES = {'adapters.safetensors', 'training_state.pt', 'manifest.json',
         'resolved_config.yaml', 'resume_config.yaml'}


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(transport, step):
    receipt_path = STATE/f'checkpoint_{step:06d}_download_verified.json'
    destination = DESTINATION/f'checkpoint-{step:06d}'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('verified') and all((destination/f).is_file() and sha(destination/f) == h
                                          for f, h in receipt['sha256'].items()):
            return True
        raise RuntimeError(f'Existing local checkpoint{step} differs from its verified receipt')
    expected = json.loads(transport.python(f'''
from pathlib import Path
import hashlib,json
p=Path({REMOTE!r})/{destination.name!r}
files={sorted(FILES)!r}
complete=all((p/f).is_file() and (p/f).stat().st_size>0 for f in files)
if complete:
 assert json.loads((p/'manifest.json').read_text())['step']=={step}
print(json.dumps({{'complete':complete,'size':sum((p/f).stat().st_size for f in files) if complete else 0,
 'sha256':{{f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in files}} if complete else {{}}}}))
'''))
    if not expected['complete']:
        return False
    assert set(expected['sha256']) == FILES
    DESTINATION.mkdir(parents=True, exist_ok=True)
    if not destination.exists() and shutil.disk_usage(DESTINATION).free < expected['size']+512*2**20:
        raise RuntimeError('Insufficient local destination space for checkpoint plus512MiB reserve')
    staging = DESTINATION/(destination.name+'.partial')
    if not destination.exists():
        staging.mkdir(exist_ok=True)
        print(f'Downloading checkpoint{step} ({expected["size"]/2**20:.1f}MiB)', flush=True)
        subprocess.run(['rsync', '-rt', '--partial', '--timeout=120', '-e', shlex.join(transport.ssh),
                        f'{transport.host}:{REMOTE}/{destination.name}/', str(staging)+'/'],
                       check=True, timeout=1800, capture_output=True, text=True)
    source = destination if destination.exists() else staging
    actual = {name: sha(source/name) for name in FILES}
    assert actual == expected['sha256'], 'Downloaded checkpoint SHA256 mismatch'
    assert json.loads((source/'manifest.json').read_text())['step'] == step
    if source == staging:
        staging.rename(destination)
    receipt = {'verified': True, 'step': step, 'instance_id': 53994096,
               'remote': REMOTE+'/'+destination.name, 'local': str(destination),
               'sha256': actual, 'verified_at_utc': datetime.now(timezone.utc).isoformat()}
    write(DESTINATION/receipt_path.name, receipt)
    write(receipt_path, receipt)
    print(f'Checkpoint{step}: all five files downloaded and SHA256-verified at {destination}', flush=True)
    return True


def main(once=False):
    STATE.mkdir(parents=True, exist_ok=True)
    lock = (STATE/'mirror.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    completed = set()
    while True:
        try:
            transport = Transport()
            for step in (4000, 6000):
                if step not in completed and download(transport, step):
                    completed.add(step)
            status = {'checked_at': datetime.now(timezone.utc).isoformat(),
                      'verified_steps': sorted(completed), 'pending_steps': sorted({4000, 6000}-completed),
                      'destination': str(DESTINATION)}
            write(STATE/'status.json', status)
            if once or completed == {4000, 6000}:
                print(json.dumps(status), flush=True)
                return
        except Exception as error:
            # Do not print command/environment values that might contain credentials.
            print(f'Checkpoint transfer pending: {type(error).__name__}; retrying in60s', flush=True)
            write(STATE/'status.json', {'error_type': type(error).__name__,
                                      'checked_at': datetime.now(timezone.utc).isoformat(),
                                      'verified_steps': sorted(completed)})
            if once:
                raise
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    main(parser.parse_args().once)
