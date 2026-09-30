"""Add progress/ETA logging to a training worker already running old code."""

import argparse
import json
import time
from pathlib import Path

from ba_dit.config import load_config
from ba_dit.logging import connect
from ba_dit.progress import progress_line, progress_metrics, recent_seconds, total_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--every", type=int, default=25)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-comet", action="store_true")
    args = parser.parse_args()
    if args.every < 1:
        parser.error("--every must be positive")
    path = args.run_dir / "metrics.jsonl"
    total = total_steps(args.run_dir)
    seconds = recent_seconds(args.run_dir)
    experiment = None if args.no_comet else connect(load_config(args.run_dir / "resolved_config.yaml"), args.run_dir)
    last_logged = -1
    try:
        with path.open() as stream:
            record = None
            for line in stream:
                record = json.loads(line)
            if record is None:
                raise RuntimeError("No completed optimizer steps in metrics.jsonl")
            stream.seek(0, 2)
            while True:
                step = int(record["step"])
                if step != last_logged and (last_logged == -1 or step % args.every == 0 or step == total):
                    progress = progress_metrics(step, total, seconds)
                    if experiment:
                        experiment.log_metrics(progress, step=step)
                    print(progress_line(step, total, progress), flush=True)
                    last_logged = step
                if args.once or step >= total:
                    break
                line = stream.readline()
                if line:
                    record = json.loads(line)
                    seconds.append(float(record["train/seconds"]))
                else:
                    time.sleep(2)
    finally:
        if experiment:
            experiment.end()


if __name__ == "__main__":
    main()
