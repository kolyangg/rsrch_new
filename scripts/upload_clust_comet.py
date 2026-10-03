"""Publish one Slurm run's closed Comet archives from the networked login node."""
import argparse
import fcntl
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
    args.run = args.run.resolve()
    setup = args.run.with_name(args.run.name+'_setup')
    root = Path(__file__).resolve().parents[1]
    lock_path = root/'scratch/clust-v100'/f'comet-upload-{args.job_id}.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for line in (root/'.env').read_text().splitlines():
        if line.startswith('COMET_API_KEY='):
            os.environ['COMET_API_KEY'] = line.split('=', 1)[1].strip().strip('"\'')
    state = root/'scratch/clust-v100'/f'comet-upload-{args.job_id}.json'
    uploaded = set(json.loads(state.read_text())) if state.exists() else set()
    experiment = None
    last_status = None
    last_step = -1
    try:
        while True:
            failed_upload = False
            archives = sorted(path for folder in (setup, args.run)
                              for path in (folder/'comet-offline').glob('*.zip'))
            for archive in archives:
                if str(archive) in uploaded:
                    continue
                if not zipfile.is_zipfile(archive):
                    failed_upload = True
                    continue  # The compute process is still writing the archive.
                # get_or_create preserves identity even after an interrupted upload.
                try:
                    subprocess.run(['comet', 'upload', str(archive)], check=True, timeout=180)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    print(f'Upload retry pending for {archive.name}: {type(error).__name__}', flush=True)
                    failed_upload = True
                    continue
                uploaded.add(str(archive))
                state.parent.mkdir(parents=True, exist_ok=True)
                state.write_text(json.dumps(sorted(uploaded), indent=2)+'\n')
            status = subprocess.run(['squeue', '-h', '-j', args.job_id, '-o', '%T'], text=True, capture_output=True)
            active = status.returncode == 0 and bool(status.stdout.strip())
            job_state = status.stdout.strip()
            if not active:
                accounting = subprocess.run(['sacct', '-n', '-X', '-j', args.job_id,
                                             '--format=State', '-P'], text=True, capture_output=True)
                states = [line.strip().split('|')[0] for line in accounting.stdout.splitlines()
                          if line.strip()]
                if accounting.returncode or not states:
                    print('Scheduler/accounting state unavailable; retrying.', flush=True)
                    time.sleep(30)
                    continue
                job_state = states[0].split()[0]
                active = job_state in {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING',
                                       'SUSPENDED', 'REQUEUED', 'RESIZING', 'SIGNALING', 'STAGE_OUT'}
            record = setup/'comet_experiment.json'
            if not record.exists():
                record = args.run/'comet_experiment.json'
            if not record.exists():
                record = root/'scratch/clust-v100'/f'comet-registration-{args.job_id}'/'comet_experiment.json'
            if record.exists():
                try:
                    if experiment is None:
                        from comet_ml import API
                        experiment = API().get_experiment_by_key(json.loads(record.read_text())['experiment_key'])
                    if experiment is None:
                        raise RuntimeError('Startup experiment is not visible yet')
                    experiment.set_state('running' if active else
                                         ('finished' if job_state == 'COMPLETED' else 'crashed'))
                    if active:
                        experiment.update_status()
                    stage_path = args.run/'status.json'
                    if not stage_path.exists():
                        stage_path = setup/'status.json'
                    stage = (json.loads(stage_path.read_text()) if stage_path.exists() else
                             {'stage':'queued' if job_state == 'PENDING' else 'starting'})
                    if job_state == 'PENDING':
                        stage = {'stage':'queued', 'status':'pending'}
                    observed = {'job_id': args.job_id, 'slurm_state':job_state, **stage}
                    if observed != last_status:
                        experiment.log_other('cluster/slurm_state', job_state)
                        experiment.log_other('cluster/stage', stage.get('stage', stage.get('status', 'starting')))
                        experiment.log_other('cluster/job_id', args.job_id)
                        experiment.log_other('cluster/status', json.dumps(observed, sort_keys=True))
                        print(json.dumps(observed, sort_keys=True), flush=True)
                        last_status = observed
                    metrics_path = args.run/'metrics.jsonl'
                    if metrics_path.exists():
                        # Skip a partially written final line. These live curves are
                        # separate from the complete offline metrics uploaded later.
                        lines = metrics_path.read_text().splitlines(keepends=True)
                        complete = [line for line in lines if line.endswith('\n')]
                        latest = json.loads(complete[-1]) if complete else None
                        if latest and latest['step'] > last_step:
                            experiment.log_metrics({'live/optimizer_step': latest['step'],
                                'live/loss': latest['train/loss'],
                                'live/eta_seconds': latest['train/eta_seconds']}, step=latest['step'])
                            last_step = latest['step']
                except Exception as error:
                    print(f'Live status retry pending: {type(error).__name__}', flush=True)
                    failed_upload = True
            if not active and not failed_upload:
                print(f'Job {args.job_id} ended ({job_state}); {len(uploaded)} archives uploaded.', flush=True)
                return
            time.sleep(30)
    finally:
        print(f'Uploader stopped after {len(uploaded)} archives.', flush=True)


if __name__ == '__main__':
    main()
