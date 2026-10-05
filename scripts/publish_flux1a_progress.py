"""Publish GB10 FLUX1a controller progress without changing training state."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/FLUX1a_vast9B_20261004'


def snapshot(run):
    setup = run.with_name(run.name + '_setup')
    states = [p for p in (setup/'status.json', run/'status.json') if p.exists()]
    state = json.loads(max(states, key=lambda p: p.stat().st_mtime).read_text())
    last = {}
    metrics = run/'metrics.jsonl'
    if metrics.exists():
        # A writer may currently be appending the last line.
        for line in reversed(metrics.read_text().splitlines()):
            try:
                last = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    folders = sorted(run.glob('validation-*'))
    validation = folders[-1] if folders else None
    return {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': state['status'], 'stage': state.get('stage', state['status']),
        'optimizer_step': last.get('step', 0),
        'validation_checkpoint': int(validation.name.split('-')[-1]) if validation else None,
        'validation_latents': len(list(validation.glob('*.safetensors'))) if validation else 0,
        'validation_images': sum(not p.stem.endswith('_raw') for p in validation.glob('*.png')) if validation else 0,
        'validation_total': 96,
        'last_training_loss': last.get('train/loss'),
    }


def main(once=False):
    if ROOT != Path('/workspace/rsrch_FLUX1abc'):
        raise RuntimeError('Publisher is scoped to the isolated GB10 FLUX1a deployment')
    for line in (ROOT/'.env').read_text().splitlines():
        if line.startswith('COMET_API_KEY='):
            os.environ.setdefault('COMET_API_KEY', line.split('=', 1)[1].strip().strip('\"\''))
    from comet_ml import APIExperiment, ExistingExperiment
    setup = RUN.with_name(RUN.name + '_setup')
    record = json.loads((setup/'comet_experiment.json').read_text())
    experiment = APIExperiment(previous_experiment=record['experiment_key'])
    owner = ExistingExperiment(previous_experiment=record['experiment_key'],
        auto_param_logging=False, auto_metric_logging=False, auto_output_logging='simple',
        log_env_details=False, log_code=False, log_git_metadata=False, log_git_patch=False)
    last_console_state = None
    log_offsets = {}
    last_telemetry = 0.
    while True:
        progress = snapshot(RUN)
        target = 6000 if (RUN/'continuation_6000.json').exists() else 4000
        checkpoints = (2000, 4000, 6000) if target == 6000 else (2000, 4000)
        result = subprocess.run(['supervisorctl', 'status', 'rsrch_flux1a'],
                                capture_output=True, text=True, timeout=15)
        supervisor = result.stdout.split()
        progress['supervisor_state'] = supervisor[1] if len(supervisor) > 1 else 'UNKNOWN'
        terminal = progress['supervisor_state'] in {'EXITED', 'FATAL', 'STOPPED'}
        if terminal and target == 6000:
            queued = subprocess.run(['supervisorctl', 'status', 'rsrch_flux1a_6k'],
                                    capture_output=True, text=True, timeout=15).stdout.split()
            if len(queued) > 1:
                progress['supervisor_state'] = queued[1]
                terminal = queued[1] in {'EXITED', 'FATAL', 'STOPPED'}
        if terminal and progress['status'] != 'completed':
            progress['status'] = 'failed_or_stopped'
        receipt = setup/'live_progress.json'
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(progress, indent=2) + '\n')
        temporary.replace(receipt)
        try:
            if time.monotonic()-last_telemetry >= 30:
                experiment.log_other('launch/stage', progress['stage'])
                experiment.log_other('runtime/live_progress', json.dumps(progress))
                experiment.log_metrics({
                    'runtime/optimizer_step': progress['optimizer_step'],
                    'runtime/validation_latents': progress['validation_latents'],
                    'runtime/validation_images': progress['validation_images'],
                }, step=progress['optimizer_step'])
                last_telemetry = time.monotonic()
            console_state = tuple(progress[k] for k in ('status', 'stage', 'optimizer_step',
                                  'validation_latents', 'validation_images'))
            if console_state != last_console_state:
                print(
                    f"{progress['checked_at_utc']} Controller: {progress['stage']} "
                    f"({progress['status']}); optimizer {progress['optimizer_step']}/{target}; "
                    f"validation checkpoint {progress['validation_checkpoint']}: "
                    f"{progress['validation_latents']}/96 generated, "
                    f"{progress['validation_images']}/96 decoded.", flush=True)
                last_console_state = console_state
            log = RUN/(progress['stage']+'.log')
            if log.exists():
                with log.open(errors='replace') as stream:
                    stream.seek(log_offsets.get(str(log), max(0, log.stat().st_size-4000)))
                    text = stream.read()
                    log_offsets[str(log)] = stream.tell()
                if text:
                    print(text.rstrip(), flush=True)
        except Exception as error:
            print(f'Comet progress upload failed: {type(error).__name__}; retrying', flush=True)
        for step in checkpoints:
            if ((RUN/f'summarize_{step}.done.json').exists()
                    and not (RUN/f'comet_verified_{step:06d}.json').exists()):
                try:
                    subprocess.run([sys.executable, '-m', 'scripts.verify_flux1a_validation',
                                    '--step', str(step), '--repair-missing'],
                                   cwd=ROOT, check=True, timeout=300)
                except (subprocess.SubprocessError, OSError) as error:
                    print(f'Validation {step} publication verification pending: {type(error).__name__}', flush=True)
        publication_complete = all((RUN/f'comet_verified_{step:06d}.json').exists()
                                   for step in checkpoints)
        if once or (terminal and (progress['status'] != 'completed' or publication_complete)):
            owner.end()
            return
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    main(parser.parse_args().once)
