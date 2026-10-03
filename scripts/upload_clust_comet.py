"""Keep one Slurm run live in Comet independently of slow archive uploads."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import zipfile


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def scheduler_state(job_id):
    result = subprocess.run(['squeue', '-h', '-j', job_id, '-o', '%T'],
                            text=True, capture_output=True, timeout=15)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip(), True
    result = subprocess.run(['sacct', '-n', '-X', '-j', job_id, '--format=State', '-P'],
                            text=True, capture_output=True, timeout=15)
    states = [line.strip().split('|')[0] for line in result.stdout.splitlines() if line.strip()]
    if result.returncode or not states:
        raise RuntimeError('Scheduler/accounting state unavailable')
    state = states[0].split()[0]
    active = state in {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED',
                      'REQUEUED', 'RESIZING', 'SIGNALING', 'STAGE_OUT'}
    return state, active


def keep_alive(key, active, stop, receipt):
    """Use a separate API connection; metadata/asset calls cannot delay heartbeats."""
    from comet_ml import API
    experiment = None
    count = 0
    interval = 5.0
    while not stop.is_set():
        started = time.monotonic()
        if active.is_set():
            try:
                if experiment is None:
                    experiment = API(cache=False).get_experiment_by_key(key)
                if experiment is None:
                    raise RuntimeError('Comet experiment not available')
                experiment.set_state('running')
                response = experiment.update_status()
                if not response:
                    raise RuntimeError('Comet heartbeat was not acknowledged')
                interval = max(1.0, min(5.0, response['isAliveBeatDurationMillis']/2000))
                count += 1
                write_json(receipt, {'experiment_key': key, 'heartbeat_count': count,
                    'acknowledged_unix_seconds': time.time(), 'interval_seconds': interval,
                    'logger_job_id': os.getenv('SLURM_JOB_ID'), 'pid': os.getpid()})
            except Exception as error:
                print(f'Heartbeat retry pending: {type(error).__name__}', flush=True)
                experiment = None
        stop.wait(max(0.1, interval-(time.monotonic()-started)))


class ArchiveUploader:
    """At most one bounded subprocess, polled without blocking live reporting."""
    def __init__(self, folders, state):
        self.folders, self.state = folders, state
        self.uploaded = set(json.loads(state.read_text())) if state.exists() else set()
        self.retry_after = {}
        self.process = None
        self.archive = None
        self.deadline = 0

    def pending(self):
        return sorted(path for folder in self.folders for path in (folder/'comet-offline').glob('*.zip')
                      if str(path) not in self.uploaded)

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=5)

    def poll(self):
        if self.process is not None:
            if self.process.poll() is None and time.monotonic() >= self.deadline:
                self.stop()
            code = self.process.poll()
            if code is None:
                return
            if code == 0:
                self.uploaded.add(str(self.archive))
                write_json(self.state, sorted(self.uploaded))
            else:
                print(f'Archive retry pending: {self.archive.name} (exit {code})', flush=True)
                self.retry_after[str(self.archive)] = time.monotonic()+300
            self.process = None
        for archive in self.pending():
            if time.monotonic() < self.retry_after.get(str(archive), 0) or not zipfile.is_zipfile(archive):
                continue
            self.archive = archive
            self.process = subprocess.Popen(['comet', 'upload', str(archive)], start_new_session=True)
            self.deadline = time.monotonic()+180
            break


def validation_progress(directory, stage):
    name = stage.get('stage', '')
    if not re.fullmatch(r'infer_\d+', name):
        return None
    path = directory/(name+'.log')
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-16384))
        lines = stream.read().decode(errors='replace')
    matches = re.findall(r'Validation (\d+): (\d+)/(\d+);', lines)
    if not matches:
        return None
    step, completed, total = map(int, matches[-1])
    return {'step': step, 'completed': completed, 'total': total}


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
    experiment = None
    last_status = None
    last_step = -1
    active_event = threading.Event()
    stop_event = threading.Event()
    heartbeat = None
    uploader = ArchiveUploader((setup, args.run), state)
    terminal_since = None
    try:
        while True:
            try:
                job_state, active = scheduler_state(args.job_id)
            except (RuntimeError, subprocess.TimeoutExpired):
                active_event.clear()  # Do not claim liveness without scheduler evidence.
                print('Scheduler/accounting state unavailable; retrying.', flush=True)
                time.sleep(10)
                continue
            if active:
                active_event.set()
            else:
                active_event.clear()
                terminal_since = terminal_since or time.monotonic()
            uploader.poll()
            record = setup/'comet_experiment.json'
            if not record.exists():
                record = args.run/'comet_experiment.json'
            if not record.exists():
                record = root/'scratch/clust-v100'/f'comet-registration-{args.job_id}'/'comet_experiment.json'
            if record.exists():
                try:
                    if experiment is None:
                        from comet_ml import API
                        key = json.loads(record.read_text())['experiment_key']
                        experiment = API(cache=False).get_experiment_by_key(key)
                    if experiment is None:
                        raise RuntimeError('Startup experiment is not visible yet')
                    if heartbeat is None and active:
                        heartbeat = threading.Thread(target=keep_alive, args=(key, active_event, stop_event,
                            state.with_name(f'comet-heartbeat-{args.job_id}.json')), daemon=True)
                        heartbeat.start()
                    stage_path = args.run/'status.json'
                    if not stage_path.exists():
                        stage_path = setup/'status.json'
                    stage = (json.loads(stage_path.read_text()) if stage_path.exists() else
                             {'stage':'queued' if job_state == 'PENDING' else 'starting'})
                    if job_state == 'PENDING':
                        stage = {'stage':'queued', 'status':'pending'}
                    observed = {'job_id': args.job_id, 'slurm_state':job_state, **stage}
                    progress = validation_progress(stage_path.parent, stage)
                    if progress:
                        observed['validation'] = progress
                    if observed != last_status:
                        experiment.log_other('cluster/slurm_state', job_state)
                        experiment.log_other('cluster/stage', stage.get('stage', stage.get('status', 'starting')))
                        experiment.log_other('cluster/job_id', args.job_id)
                        experiment.log_other('cluster/status', json.dumps(observed, sort_keys=True))
                        if progress:
                            experiment.log_metrics({'live/validation_images': progress['completed'],
                                'live/validation_total': progress['total']}, step=progress['step'])
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
                    elif last_step < 0:
                        experiment.log_metric('live/optimizer_step', 0, step=0)
                        last_step = 0
                except Exception as error:
                    print(f'Live status retry pending: {type(error).__name__}', flush=True)
                    time.sleep(5)
                    continue
            if not active and (not uploader.pending() or time.monotonic()-terminal_since >= 180):
                stop_event.set()
                if heartbeat is not None:
                    heartbeat.join(timeout=30)
                uploader.stop()
                if experiment is not None:
                    experiment.set_state('finished' if job_state == 'COMPLETED' else 'crashed')
                write_json(state.with_name(f'comet-pending-{args.job_id}.json'),
                           [str(path) for path in uploader.pending()])
                print(f'Job {args.job_id} ended ({job_state}); {len(uploader.uploaded)} archives uploaded.', flush=True)
                return
            time.sleep(5)
    finally:
        stop_event.set()
        uploader.stop()
        print(f'Uploader stopped after {len(uploader.uploaded)} archives.', flush=True)


if __name__ == '__main__':
    main()
