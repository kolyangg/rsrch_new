"""Expose a running native validation's progress without touching its GPU worker."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from ba_dit.config import load_config
from ba_dit.logging import connect


def main(args):
    config = load_config(args.validation / 'resolved_config.yaml')
    total = config['validation']['limit']
    experiment = connect(config, args.log_dir)
    try:
        while True:
            report = json.loads((args.validation / 'validation.json').read_text())
            assert report['mode'] == 'native' and not report['checkpoint'] and not report['adapter_inventory']
            rows = report['samples']
            count = len(rows)
            seconds = sum(row['seconds'] for row in rows) / max(1, count)
            scored = (args.validation / 'quality_summary.json').exists()
            stage = 'completed' if scored else ('decode_and_score' if count == total else 'generation')
            metrics = {'validation/generated_samples': count, 'validation/total_samples': total,
                       'validation/generation_progress_percent': 100 * count / total,
                       'validation/generation_eta_seconds': (total - count) * seconds}
            experiment.log_metrics(metrics, step=count)
            experiment.log_other('validation/live_stage', stage)
            experiment.log_parameter('queue/state', stage)
            print(json.dumps({'stage': stage, **metrics}), flush=True)
            if scored:
                break
            status = subprocess.run(['supervisorctl', 'status', args.supervisor],
                                    capture_output=True, text=True, timeout=10)
            if 'RUNNING' not in status.stdout:
                experiment.log_other('validation/live_stage', 'controller_stopped_before_scoring')
                raise RuntimeError('Validation controller stopped before scoring')
            time.sleep(60)
    finally:
        experiment.end()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--supervisor', required=True)
    main(parser.parse_args())
