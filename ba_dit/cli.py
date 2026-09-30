"""Preparation, training, and validation entry points for all four GPU profiles."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ba_dit.config import ROOT, digest, load_config, revisions


def call(command, config_path, *extra):
    subprocess.run([sys.executable, "-m", "ba_dit.cli", command, "--config", str(config_path), *map(str, extra)], cwd=ROOT, check=True)


def resolved_file(config):
    destination = Path(config["data"]["cache_dir"]) / "configs" / f"{digest(config)}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False))
    return destination


def prepare_all(config_path, split, limit=None):
    extra = ["--split", split]
    if limit:
        extra += ["--limit", limit]
    for stage in ("encoder", "vae"):
        call("precompute", config_path, "--stage", stage, *extra)


def preflight(config, split):
    import inspect
    import torch
    from ba_dit.data.manifest import assert_disjoint, read_manifest
    from ba_dit.runtime import ROOT as SOURCE_ROOT, SOURCE_DIRS, backend_module

    locked = revisions(config)
    for field in ("weights", "encoder", "vae"):
        path = Path(config["model"][field])
        if not path.exists():
            raise FileNotFoundError(path)
        snapshot_path = path if path.is_dir() else path.parent
        if not any(revision in snapshot_path.resolve().parts for revision in locked.values()):
            raise ValueError(f"{field} path does not resolve into a locked HF snapshot: {path}")
    validation = read_manifest(config["data"]["validation_manifest"])
    if split == "train":
        train = read_manifest(config["data"]["train_manifest"], training=True)
        assert_disjoint(train, validation)
    backend = backend_module(config)
    model_class = backend.Flux2 if config["model"]["backend"] == "flux" else backend.QwenImage21Transformer2DModel
    origin = Path(inspect.getfile(model_class)).resolve()
    expected = SOURCE_ROOT / "sources" / SOURCE_DIRS[config["model"]["backend"]]
    if not origin.is_relative_to(expected.resolve()):
        raise RuntimeError(f"Model imported from unpinned source: {origin}")
    result = {"profile": config["name"], "model_origin": str(origin), "revisions": locked,
              "validation_items": len(validation), "torch": torch.__version__, "cuda_runtime": torch.version.cuda}
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        result.update(gpu=gpu.name, vram_gib=gpu.total_memory / 2**30,
                      meets_training_profile=gpu.total_memory >= config["hardware"]["min_vram_gb"] * 10**9)
    print(json.dumps(result, indent=2), flush=True)


def infer(config, args, config_path):
    output = Path(args.output_dir or ROOT / "runs" / f"{config['name']}_{args.mode}_validation_{stamp()}").resolve()
    if output.exists() and not args.resume_validation:
        raise FileExistsError(f"Choose a new validation output directory: {output}")
    if args.resume_validation:
        previous = load_config(output / "resolved_config.yaml")
        if digest(previous) != digest(config):
            raise ValueError("Validation resume requires the same resolved config")
    metrics_python = Path(os.environ.get("BA_ENVS_DIR", ROOT / "envs")) / "metrics/bin/python"
    quality_python = metrics_python.parents[2] / "face-quality/bin/python"
    if (not args.skip_output_masks or args.quality_metrics) and not metrics_python.exists():
        raise FileNotFoundError("Run scripts/setup_metrics.sh for backbone-specific validation face masks and quality scoring")
    if args.quality_metrics and not quality_python.exists():
        raise FileNotFoundError("Run scripts/setup_face_quality.sh for the original seven face-quality curves")
    if args.checkpoint and not args.skip_output_masks:
        from ba_dit.data.manifest import read_manifest
        from ba_dit.validation_masks import load_masks
        samples = read_manifest(config["data"]["validation_manifest"])[:args.limit or config["validation"]["limit"]]
        load_masks(config, [row["sample_id"] for row in samples])
    output.mkdir(parents=True, exist_ok=args.resume_validation)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    from ba_dit.logging import connect
    experiment = connect(config, args.log_dir or output, name=None if args.log_dir else output.name)
    if experiment:
        experiment.log_parameters({"validation/mode": args.mode})
        experiment.log_asset(str(output / "resolved_config.yaml"), file_name=f"validation_config_{args.global_step:06d}.yaml")
        experiment.end()
    prepare_all(config_path, "validation", args.limit or config["validation"]["limit"])
    extra = ["--mode", args.mode, "--output-dir", output]
    if args.checkpoint:
        extra += ["--checkpoint", args.checkpoint]
    if args.limit:
        extra += ["--limit", args.limit]
    if args.compare_native:
        extra += ["--compare-native"]
    if args.resume_validation:
        extra += ["--resume-validation"]
    call("_infer-worker", config_path, *extra)
    decode_extra = ["--output-dir", output, "--global-step", args.global_step]
    if args.log_dir:
        decode_extra += ["--log-dir", args.log_dir]
    call("_decode-worker", config_path, *decode_extra)
    evaluate(output)
    if not args.skip_output_masks:
        if args.checkpoint:
            from ba_dit.validation_masks import load_masks
            report = json.loads((output / "validation.json").read_text())
            load_masks(config, [row["sample_id"] for row in report["samples"]])
        else:
            subprocess.run([str(metrics_python), str(ROOT / "scripts/build_validation_output_masks.py"), "--validation", str(output)], check=True, cwd=ROOT)
    if args.quality_metrics:
        metrics_args = [str(metrics_python), str(ROOT / "scripts/evaluate_metrics.py"), "--validation", str(output), "--global-step", str(args.global_step)]
        if args.log_dir:
            metrics_args += ["--log-dir", str(args.log_dir)]
        subprocess.run(metrics_args, check=True, cwd=ROOT)
        quality_args = [str(quality_python), str(ROOT / "scripts/evaluate_face_quality.py"), "--validation", str(output), "--global-step", str(args.global_step)]
        if args.log_dir:
            quality_args += ["--log-dir", str(args.log_dir)]
        subprocess.run(quality_args, check=True, cwd=ROOT)
    print(f"Validation saved to {output}", flush=True)


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def train(config, args, config_path):
    from ba_dit.logging import connect
    import torch

    if args.mode == "native":
        raise ValueError("Use infer for the native mode")
    if not args.allow_small_gpu and torch.cuda.get_device_properties(0).total_memory < config["hardware"]["min_vram_gb"] * 10**9:
        raise RuntimeError("GPU does not meet this training profile; --allow-small-gpu is only for a named local smoke")
    if args.resume:
        run_dir = Path(args.resume).resolve().parent
        start = json.loads((Path(args.resume) / "manifest.json").read_text())["step"]
    else:
        run_dir = Path(args.output_dir or ROOT / "runs" / (args.run_name or f"{config['name']}_{args.mode}_{stamp()}")).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        start = 0
    experiment = connect(config, run_dir, name=run_dir.name)
    if experiment:
        experiment.log_parameters({"mode": args.mode, "smoke": bool(args.smoke_steps)})
        experiment.end()
    prepare_all(config_path, "train", args.limit)
    checkpoint = args.resume
    if not args.smoke_steps and start == 0:
        initial = ["--checkpoint", args.init_adapter] if args.init_adapter else []
        initial += ["--quality-metrics" if args.quality_metrics else "--no-quality-metrics"]
        call("infer", config_path, "--mode", args.mode, "--output-dir", run_dir / "validation-000000", "--log-dir", run_dir, *initial)
    while start < config["training"]["steps"]:
        interval = config["training"]["validation_every"]
        boundary = start + args.smoke_steps if args.smoke_steps else (start // interval + 1) * interval
        until = min(config["training"]["steps"], boundary)
        extra = ["--mode", args.mode, "--output-dir", run_dir, "--until", until]
        if checkpoint:
            extra += ["--resume", checkpoint]
        elif args.init_adapter:
            extra += ["--init-adapter", args.init_adapter]
        if args.allow_small_gpu:
            extra += ["--allow-small-gpu"]
        if args.limit:
            extra += ["--limit", args.limit]
        call("_train-worker", config_path, *extra)
        checkpoint = (run_dir / "latest_checkpoint.txt").read_text().strip()
        start = until
        if args.smoke_steps:
            break
        call("infer", config_path, "--mode", args.mode, "--checkpoint", checkpoint,
             "--output-dir", run_dir / f"validation-{until:06d}", "--log-dir", run_dir, "--global-step", until,
             "--quality-metrics" if args.quality_metrics else "--no-quality-metrics")
    print(f"Training checkpoint: {checkpoint}", flush=True)


def evaluate(directory):
    import csv
    import numpy as np
    from PIL import Image

    directory = Path(directory)
    report = json.loads((directory / "validation.json").read_text())
    results = []
    for row in report["samples"]:
        with Image.open(directory / row["image"]) as image:
            image.load()
            expected_h, expected_w = report["target_size"]
            if image.size != (expected_w, expected_h):
                raise ValueError(f"Incorrect generated size for {row['sample_id']}")
            difference = None
            native = directory / f"{row['sample_id']}.native.png"
            if native.exists():
                with Image.open(native) as baseline:
                    difference = float(np.abs(np.array(image).astype(float) - np.array(baseline).astype(float)).max())
        results.append({"sample_id": row["sample_id"], "identity_id": row["identity_id"], "pixel_max_abs_vs_native": difference,
                        "seconds": row["seconds"], "peak_cuda_reserved_gib": row["peak_cuda_reserved_gib"]})
    with (directory / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"validated_images": len(results), "per_image_csv": str(directory / "per_image.csv")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "precompute", "train", "infer", "evaluate", "_train-worker", "_infer-worker", "_decode-worker"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("native", "branch_only", "lora_only", "lora_plus_branch"), default="branch_only")
    parser.add_argument("--stage", choices=("encoder", "vae", "all"), default="all")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--until", type=int)
    parser.add_argument("--global-step", type=int, default=0)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--target-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--reference-size", type=int)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--memory-fraction", type=float)
    parser.add_argument("--allow-small-gpu", action="store_true")
    parser.add_argument("--compare-native", action="store_true")
    parser.add_argument("--quality-metrics", action=argparse.BooleanOptionalAction, default=None, help="Score legacy ID/CLIP and the seven face-quality curves (default: enabled)")
    parser.add_argument("--skip-output-masks", action="store_true", help="Only for tensor parity diagnostics without face scoring")
    parser.add_argument("--resume-validation", action="store_true", help="Continue a partial validation in --output-dir")
    parser.add_argument("--no-comet", action="store_true")
    args = parser.parse_args()
    for name in ("limit", "steps", "smoke_steps", "grad_accum", "until"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    saved = Path(args.resume) / "resume_config.yaml" if args.resume else args.config
    if args.resume and not saved.exists():
        saved = Path(args.resume) / "resolved_config.yaml"
    config = load_config(saved)
    if args.target_size:
        config["data"]["target_size"] = args.target_size
    if args.reference_size is not None:
        config["data"]["reference_size"] = args.reference_size
    if args.train_manifest:
        config["data"]["train_manifest"] = str(args.train_manifest.resolve())
    if "train" in args.command and args.limit:
        config["data"]["train_limit"] = args.limit
    if args.grad_accum:
        config["training"]["grad_accum"] = args.grad_accum
    if args.memory_fraction is not None:
        config["training"]["max_reserved_fraction"] = args.memory_fraction
    if args.steps:
        config["training" if "train" in args.command else "validation"]["steps"] = args.steps
    if args.smoke_steps and not args.resume:
        config["training"]["steps"] = args.smoke_steps
    if args.no_comet:
        config["logging"]["enabled"] = False
    if args.quality_metrics is None:
        args.quality_metrics = not args.skip_output_masks
    if args.skip_output_masks and args.quality_metrics:
        parser.error("Quality scoring requires output masks")
    config_path = resolved_file(config)
    config = load_config(config_path)  # Reject invalid CLI overrides through the same schema.
    if "train" in args.command:
        args.limit = config["data"]["train_limit"]
    if args.command == "preflight":
        preflight(config, args.split)
    elif args.command == "precompute":
        if args.stage == "all":
            prepare_all(config_path, args.split, args.limit)
        else:
            from ba_dit.precompute import prepare
            prepare(config, args.stage, args.split, args.limit)
    elif args.command == "train":
        train(config, args, config_path)
    elif args.command == "_train-worker":
        from ba_dit.training import train_segment
        train_segment(config, args.mode, args.output_dir, args.until, args.resume, args.init_adapter, args.allow_small_gpu, args.limit)
    elif args.command == "infer":
        infer(config, args, config_path)
    elif args.command == "_infer-worker":
        from ba_dit.inference import denoise
        denoise(config, args.mode, args.output_dir, args.checkpoint, args.limit, args.compare_native, args.resume_validation)
    elif args.command == "_decode-worker":
        from ba_dit.inference import decode
        decode(config, args.output_dir, args.log_dir, args.global_step)
    else:
        evaluate(args.output_dir)


if __name__ == "__main__":
    main()
