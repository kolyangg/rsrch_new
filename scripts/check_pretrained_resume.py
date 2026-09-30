#!/usr/bin/env python3
"""Bounded pretrained test: two updates must equal one update + process restart."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

from ba_dit.config import ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("branch_only", "lora_only", "lora_plus_branch"), default="branch_only")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    config["data"].update(train_manifest=str(ROOT / "data/train_pairs_smoke.jsonl"), train_limit=2, target_size=[256, 256], reference_size=512)
    config["training"].update(steps=2, grad_accum=1)
    config["logging"]["enabled"] = False
    path = output / "resolved_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))

    def run(command, *options):
        subprocess.run([sys.executable, "-m", "ba_dit.cli", command, "--config", str(path), *map(str, options)], check=True, cwd=ROOT)

    run("precompute", "--split", "train", "--limit", 2)
    continuous, restarted = output / "continuous", output / "restarted"
    for directory in (continuous, restarted):
        directory.mkdir()
    run("_train-worker", "--mode", args.mode, "--output-dir", continuous, "--until", 2, "--allow-small-gpu")
    run("_train-worker", "--mode", args.mode, "--output-dir", restarted, "--until", 1, "--allow-small-gpu")
    run("_train-worker", "--mode", args.mode, "--output-dir", restarted, "--until", 2, "--allow-small-gpu", "--resume", restarted / "checkpoint-000001")
    a = load_file(continuous / "checkpoint-000002/adapters.safetensors")
    b = load_file(restarted / "checkpoint-000002/adapters.safetensors")
    exact = all(torch.equal(a[name], b[name]) for name in a)
    report = {"mode": args.mode, "adapter_tensors": len(a), "all_adapter_tensors_exact": exact,
              "max_abs": max(float((a[name] - b[name]).abs().max()) for name in a), "steps": 2,
              "target_size": [256, 256], "reference_size": 512}
    (output / "resume_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not exact:
        raise RuntimeError("Pretrained resume changed the optimizer trajectory")


if __name__ == "__main__":
    main()
