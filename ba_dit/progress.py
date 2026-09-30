"""Compact optimizer progress shared by training and the live-run logger."""

from collections import deque
from datetime import datetime
from pathlib import Path

import yaml


def total_steps(run_dir):
    return int(yaml.safe_load((Path(run_dir) / "resolved_config.yaml").read_text())["training"]["steps"])


def recent_seconds(run_dir, window=100):
    samples = deque(maxlen=window)
    path = Path(run_dir) / "metrics.jsonl"
    if path.exists():
        import json

        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if "train/seconds" in record:
                    samples.append(float(record["train/seconds"]))
    return samples


def progress_metrics(step, total, seconds):
    mean = sum(seconds) / len(seconds)
    remaining = max(0, total - step)
    return {"train/progress_percent": 100 * step / total,
            "train/steps_remaining": remaining,
            "train/eta_seconds": remaining * mean,
            "train/updates_per_hour": 3600 / mean}


def progress_line(step, total, metrics):
    filled = min(20, int(20 * step / total))
    eta = int(metrics["train/eta_seconds"])
    hours, remainder = divmod(eta, 3600)
    minutes, seconds = divmod(remainder, 60)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    return (f"{stamp} [{('#' * filled):<20}] {step}/{total} "
            f"({metrics['train/progress_percent']:.1f}%) "
            f"{metrics['train/updates_per_hour']:.1f} steps/h ETA {hours:02d}:{minutes:02d}:{seconds:02d}")
