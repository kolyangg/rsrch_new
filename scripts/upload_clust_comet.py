"""Publish one Slurm run's closed Comet archives from the networked login node."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--job-id', required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    for line in (root/'.env').read_text().splitlines():
        if line.startswith('COMET_API_KEY='):
            os.environ['COMET_API_KEY'] = line.split('=', 1)[1].strip().strip('"\'')
    state = root/'scratch/clust-v100'/f'comet-upload-{args.job_id}.json'
    uploaded = set(json.loads(state.read_text())) if state.exists() else set()
    while True:
        for archive in sorted((args.run/'comet-offline').glob('*.zip')):
            if str(archive) in uploaded:
                continue
            if not zipfile.is_zipfile(archive):
                continue  # The compute process is still writing the archive.
            # These archives use get_or_create, never force-upload (which makes
            # a different key when the original experiment already exists).
            subprocess.run(['comet', 'upload', str(archive)], check=True)
            uploaded.add(str(archive))
            state.parent.mkdir(parents=True, exist_ok=True)
            state.write_text(json.dumps(sorted(uploaded), indent=2)+'\n')
        status = subprocess.run(['squeue', '-h', '-j', args.job_id], text=True, capture_output=True)
        if status.returncode:
            raise RuntimeError(status.stderr)
        if not status.stdout.strip():
            print(f'Job {args.job_id} ended; {len(uploaded)} archives uploaded.', flush=True)
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
