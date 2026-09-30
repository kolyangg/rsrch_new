"""Comet run identity and explicit research metrics; credentials stay local."""

import json
import os
from pathlib import Path

from ba_dit.config import ROOT, adapter_identity, digest
from ba_dit.progress import progress_line, progress_metrics, recent_seconds, total_steps


_progress = {}


def connect(config, run_dir, name=None):
    if not config["logging"]["enabled"]:
        return None
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("COMET_API_KEY="):
                os.environ.setdefault("COMET_API_KEY", line.split("=", 1)[1].strip().strip("\"'"))
    if not os.getenv("COMET_API_KEY"):
        raise RuntimeError("COMET_API_KEY is required; use --no-comet only for an explicitly local smoke")
    from comet_ml import ExistingExperiment, Experiment

    run_dir = Path(run_dir)
    record = run_dir / "comet_experiment.json"
    options = dict(auto_param_logging=False, auto_metric_logging=False, log_env_details=False, log_code=False,
                   log_git_metadata=False, log_git_patch=False)
    if record.exists():
        key = json.loads(record.read_text())["experiment_key"]
        experiment = ExistingExperiment(previous_experiment=key, **options)
    else:
        experiment = Experiment(project_name=config["logging"]["comet_project"], **options)
        record.write_text(json.dumps({"experiment_key": experiment.get_key(), "project_name": config["logging"]["comet_project"]}, indent=2) + "\n")
    if name:
        experiment.set_name(name)
    experiment.log_parameters({f"{section}/{key}": value for section, fields in config.items()
                               if isinstance(fields, dict) for key, value in fields.items()})
    experiment.log_parameters({"config_sha256": digest(config), "model_identity": json.dumps(adapter_identity(config), sort_keys=True)})
    return experiment


def log_metrics(experiment, run_dir, metrics, step):
    run_dir = Path(run_dir)
    key = str(run_dir.resolve())
    if key not in _progress:
        _progress[key] = (total_steps(run_dir), recent_seconds(run_dir))
    total, seconds = _progress[key]
    seconds.append(float(metrics["train/seconds"]))
    progress = progress_metrics(step, total, seconds)
    record = {"step": step, **metrics, **progress}
    with (run_dir / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    if experiment:
        experiment.log_metrics({**metrics, **progress}, step=step)
    if step == 1 or step % 25 == 0 or step == total:
        print(progress_line(step, total, progress), flush=True)
