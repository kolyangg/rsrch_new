#!/usr/bin/env python3
"""Upload a completed native validation run to the rsrch_new Comet project."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    report = run_dir / "validation.json"
    data = json.loads(report.read_text())
    if not data.get("samples") or len(data["samples"]) != len(list(run_dir.glob("[0-9][0-9].png"))):
        raise ValueError("Validation report and generated images do not match")
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("COMET_API_KEY="):
            os.environ["COMET_API_KEY"] = line.split("=", 1)[1].strip().strip('"\'')
    if not os.getenv("COMET_API_KEY"):
        raise RuntimeError("COMET_API_KEY is unavailable")
    from comet_ml import Experiment

    experiment = Experiment(project_name="rsrch_new")
    experiment.set_name(args.run_name)
    panel = ROOT / "data/validation/manual_val_96.jsonl"
    backend = data["backend"]
    is_qwen = "qwen" in backend
    source_key = "diffusers-qwen" if is_qwen else "ai-toolkit-flux"
    weight_lock = ROOT / "locks" / ("weights-qwen.json" if is_qwen else "weights-small.json")
    components = json.loads(weight_lock.read_text())["components"]
    needed = {"qwen21"} if is_qwen else {"flux4b", "flux4b_text", "flux_vae"}
    revisions = {component["key"]: component["revision"] for component in components if component["key"] in needed}
    experiment.log_parameters({
        "backend": data["backend"], "steps": data["steps"], "width": data["width"],
        "height": data["height"], "reference_size": data.get("reference_size"),
        "panel_sha256": hashlib.sha256(panel.read_bytes()).hexdigest(),
        "validation_items": len(data["samples"]),
        "mode": "branch_initialized" if "branch_initialized" in backend else "native",
        "source_commit": (ROOT / "locks" / f"{source_key}.commit").read_text().strip(),
        "weight_lock_sha256": hashlib.sha256(weight_lock.read_bytes()).hexdigest(),
        "weight_revisions": json.dumps(revisions, sort_keys=True),
    })
    for item in data["samples"]:
        experiment.log_metric("validation/seconds", item["seconds"], step=int(item["sample_id"]))
        experiment.log_metric("hardware/peak_cuda_reserved_gib", item["peak_cuda_reserved_gib"], step=int(item["sample_id"]))
        experiment.log_image(str(run_dir / item["image"]), name=item["image"], step=int(item["sample_id"]))
    key = experiment.get_key()
    record = {
        "schema_version": 1, "run_name": args.run_name,
        "comet": {"experiment_key": key, "project_name": "rsrch_new", "mode": "online"},
    }
    with tempfile.NamedTemporaryFile("w", dir=run_dir, prefix=".comet-", delete=False) as temporary:
        json.dump(record, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(run_dir / "comet_experiment.json")
    experiment.end()
    print(f"Comet experiment {key} saved to {run_dir / 'comet_experiment.json'}")


if __name__ == "__main__":
    main()
