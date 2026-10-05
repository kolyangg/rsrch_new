"""Score a completed fixed96 panel before releasing its inference GPU."""
import json
from datetime import timedelta
import os
from pathlib import Path
import subprocess
import time


def score(root, run, step):
    root, run = Path(root), Path(run)
    folder = run / f'validation-{step:06d}'
    environment = root / 'envs/clust-v100/metrics-gpu'
    if not (environment / 'gpu_verified.json').exists():
        print('GPU scoring admission pending; existing CPU scoring stages retained', flush=True)
        return
    # Called by the supervisor after the inference child exits: its backbone
    # and VAE have been released before either scoring environment starts.
    from scripts.clust_stream_decode import decoded
    from ba_dit.config import load_config
    assert decoded(run, load_config(run / 'resolved_config.yaml'), step)
    for action, module in [('score', 'scripts.evaluate_metrics'),
                           ('face_quality', 'scripts.evaluate_face_quality')]:
        marker = run / f'{action}_{step}.done.json'
        if marker.exists():
            continue
        started = time.monotonic()
        command = [str(root / 'envs/clust-v100/metrics-gpu/bin/python'), '-u', '-m', module,
                   '--validation', str(folder), '--log-dir', str(run), '--global-step', str(step),
                   '--no-comet', '--device', 'cuda']
        if action == 'score':
            command += ['--ownership-boxes', str(run / 'ownership_boxes.json')]
        else:
            command += ['--threads', os.getenv('SLURM_CPUS_PER_TASK', '2')]
        log = run / f'gpu_{action}_{step}.log'
        print(f'GPU METRICS {step} | {action} | details {log}', flush=True)
        with log.open('w') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            while child.poll() is None:
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    progress = os.getenv('BA_STAGE_PROGRESS')
                    if progress:
                        try:
                            value = json.loads(Path(progress).read_text())
                            eta=value['eta_seconds']
                            remaining=str(timedelta(seconds=max(0,int(eta)))) if eta is not None else 'estimating'
                            print(f"GPU METRICS {step} | {value['stage']} {value['completed']}/{value['total']} | ETA {remaining}", flush=True)
                        except (OSError, json.JSONDecodeError):
                            pass
        if child.returncode:
            raise RuntimeError(f'GPU {action} failed ({child.returncode}); see {log}')
        print(f'GPU METRICS {step} | {action} complete in {time.monotonic()-started:.1f}s', flush=True)
        marker.write_text(json.dumps({'job_id': os.getenv('SLURM_JOB_ID'),
                                     'device': 'cuda', 'seconds': time.monotonic()-started})+'\n')
    (run / f'decode_{step}.done.json').write_text(json.dumps({'job_id': os.getenv('SLURM_JOB_ID'),
                                                           'streamed_and_verified': True})+'\n')


def cleanup_cache(run, config, rows, step):
    """Delete consumed conditioning before creating the next window, never during it."""
    from ba_dit.data.cache import cache_path
    run = Path(run)
    receipt = run / f'cache_files_{step}.json'
    if not receipt.exists():
        return
    assert (run / f'decode_{step}.done.json').exists(), 'Validation conditioning still in use'
    keep = {cache_path(config, r, s) for r in rows for s in ('encoder', 'vae')}
    keep.add(cache_path(config, {**rows[0], 'prompt': ''}, 'encoder'))
    for path in map(Path, json.loads(receipt.read_text())):
        assert path.is_relative_to(Path(config['data']['cache_dir']))
        if path not in keep:
            path.unlink(missing_ok=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    score(Path(__file__).resolve().parents[1], args.run, args.step)
