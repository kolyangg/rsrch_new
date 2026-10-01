"""Start the one-ID diagnostic only after the existing 2,000-step run and scoring exit."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ba_dit.config import ROOT, load_config


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def require_completed(run, step):
    manifest = json.loads((run / f"checkpoint-{step:06d}/manifest.json").read_text())
    if manifest["step"] != step:
        raise ValueError("Checkpoint step differs from requested handoff")
    config = load_config(run / "resolved_config.yaml")
    if config["training"]["steps"] != step:
        raise ValueError("The previous run is not configured to finish at the requested step")
    directory = run / f"validation-{step:06d}"
    report = json.loads((directory / "validation.json").read_text())
    if len(report["samples"]) != config["validation"]["limit"]:
        raise ValueError("Final validation panel is incomplete")
    for row in report["samples"]:
        if not (directory / row["image"]).is_file():
            raise FileNotFoundError(directory / row["image"])
    for name in ("quality_summary.json", "face_quality_summary.json"):
        if not json.loads((directory / name).read_text())["metrics"]:
            raise ValueError(f"Final scoring is incomplete: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--previous-pid", type=int, required=True)
    parser.add_argument("--run-name", default="flux48_one_id_diagnostic")
    parser.add_argument("--state", type=Path, default=ROOT / "runs/one_id_queue.json")
    args = parser.parse_args()
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"previous_run": str(args.previous_run.resolve()), "previous_pid": args.previous_pid,
                 "run_dir": str(ROOT / "runs" / args.run_name), "queue_pid": os.getpid()}

        def save(phase, **extra):
            state.update(phase=phase, updated_unix=time.time(), **extra)
            temporary = args.state.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(args.state)
            print(json.dumps(state), flush=True)

        try:
            identity = process_identity(args.previous_pid)
            save("waiting_for_previous_training_and_validation")
            while identity is not None and process_identity(args.previous_pid) == identity:
                time.sleep(30)
            require_completed(args.previous_run, 2000)
            subprocess.run([sys.executable, "-m", "scripts.compare_validation_steps", str(args.previous_run)], cwd=ROOT, check=True)
            if (ROOT / "runs" / args.run_name).exists():
                raise FileExistsError("One-ID run already exists; inspect it before resuming")
            save("starting_one_id")
            child = subprocess.Popen(["bash", "scripts/run_profile.sh", "flux48-one-id", "train",
                                      "--mode", "branch_only", "--quality-metrics", "--run-name", args.run_name], cwd=ROOT)
            save("one_id_running", training_pid=child.pid)
            result = child.wait()
            if result:
                raise RuntimeError(f"One-ID training exited with status {result}")
            require_completed(ROOT / "runs" / args.run_name, 2000)
            for step in (500, 1000, 1500, 2000):
                subprocess.run([sys.executable, "-m", "scripts.compare_validation_steps", state["run_dir"],
                                "--later-step", str(step)], cwd=ROOT, check=True)
            save("one_id_complete")
        except Exception as error:
            save("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
