"""Compare the frozen one-ID 24-image panel at BA updates 0/1000/2000."""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file, save_file

from ba_dit.config import ROOT, digest
from ba_dit.data.cache import load_pair
from ba_dit.data.manifest import file_hash, read_manifest
from ba_dit.face_suffix import identity, install, load_branch, settings
from ba_dit.logging import connect
from ba_dit.runtime import backend_module


TRAIN = ROOT / "runs/flux4b_face_one_id_strong_20261001"
PANEL = ROOT / "data/datasets/one_id/validation_24_seeds01.jsonl"
STEPS = (0, 1000, 2000)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def panel_text(original):
    rows = [json.loads(line) for line in original.read_text().splitlines()]
    if len(rows) != 12 or [r["sample_id"] for r in rows] != [f"oneid_{i:02d}" for i in range(12)]:
        raise ValueError("The original 12-prompt one-ID panel changed")
    if len({r["identity_id"] for r in rows}) != 1 or any(r["seed"] != 0 for r in rows):
        raise ValueError("Expected one identity and original seed zero")
    extended = rows + [{**r, "sample_id": r["sample_id"] + "_s1", "seed": 1} for r in rows]
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in extended)


def initialize(run, source):
    spec, training_config = settings(source / "diagnostic.yaml")
    expected = identity(training_config, spec)
    if json.loads((source / "identity.json").read_text()) != expected:
        raise ValueError("Training source/config identity differs from the checkpoints")
    if json.loads((source / "training_summary.json").read_text())["steps"] != 2000:
        raise ValueError("The requested 2,000-update training run is incomplete")
    checkpoints = {}
    for step in (1000, 2000):
        path = source / f"checkpoint-{step:06d}"
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["identity"] != expected or manifest["step"] != step or file_hash(path / "branch.safetensors") != manifest["branch_sha256"]:
            raise ValueError(f"Checkpoint {step} identity/checksum mismatch")
        checkpoints[str(step)] = {"path": str(path), "manifest_sha256": file_hash(path / "manifest.json"),
                                  "branch_sha256": manifest["branch_sha256"]}
    original = Path(training_config["data"]["validation_manifest"])
    text = panel_text(original)
    if PANEL.exists() and PANEL.read_text() != text:
        raise ValueError("The fixed 24-image validation panel changed")
    PANEL.write_text(text)
    config = yaml.safe_load(yaml.safe_dump(training_config))
    config["name"] = "flux4b_face_one_id_24_validation"
    config["data"]["validation_manifest"] = str(PANEL)
    config["validation"]["limit"] = 24
    record = {"source_run": str(source), "training_identity": expected,
              "training_comet_key": json.loads((source / "comet_experiment.json").read_text())["experiment_key"],
              "original_panel_sha256": file_hash(original), "panel_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "panel_rule": "original 12 prompts/reference/order with seeds 0 and 1; first 12 are unchanged",
              "checkpoints": checkpoints, "steps": list(STEPS), "resolution": 768, "inference_steps": 20, "cfg": 4.0}
    run.mkdir(parents=True, exist_ok=True)
    if (run / "provenance.json").exists() and json.loads((run / "provenance.json").read_text()) != record:
        raise ValueError("Existing validation run has different provenance")
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run / "panel.jsonl").write_text(text)
    write_json(run / "provenance.json", record)
    rows = read_manifest(PANEL)
    if len(rows) != 24 or len({r["identity_id"] for r in rows}) != 1:
        raise ValueError("Invalid 24-image one-ID panel")
    return spec, config, record, rows


def experiment(config, run, provenance):
    exp = connect(config, run, name=run.name)
    if exp:
        exp.log_parameters({"experiment_type": "face_suffix_24_image_validation", "training_comet_key": provenance["training_comet_key"],
            "training_identity_sha256": digest(provenance["training_identity"]), "panel_sha256": provenance["panel_sha256"],
            "samples": 24, "prompts": 12, "seeds": "0,1", "updates": "0,1000,2000",
            "model_resolution": 768, "inference_steps": 20, "cfg": 4.0, "branch_sites": "12-19", "branch_rank": 256,
            "validation_source_sha256": file_hash(Path(__file__))})
        for name in ("panel.jsonl", "provenance.json", "resolved_config.yaml"):
            exp.log_asset(str(run / name), file_name=name)
        exp.log_asset(str(Path(__file__)), file_name="validate_face_suffix_24.py")
    return exp


def existing_sample(source, step, row):
    if row["seed"] or int(row["sample_id"].split("_")[1]) >= 4:
        return None
    origin = source / f"validation-{step:06d}"
    report = json.loads((origin / "validation.json").read_text())
    entry = next((s for s in report["samples"] if s["sample_id"] == row["sample_id"]), None)
    if not entry or entry["prompt"] != row["prompt"] or entry["seed"] != row["seed"]:
        raise ValueError("Original four-image validation is not comparable")
    expected = (source / "identity.json").read_text()
    if not expected:
        raise ValueError("Missing original training identity")
    if step and report["checkpoint"]["manifest_sha256"] != file_hash(source / f"checkpoint-{step:06d}" / "manifest.json"):
        raise ValueError("Original image came from another checkpoint")
    if step == 0 and report["checkpoint"] is not None:
        raise ValueError("Original baseline was not zero initialized")
    return entry, origin


def sample_batches(rows, source, step):
    pending = [r for r in rows if existing_sample(source, step, r) is None]
    return [pending[i:i + 2] for i in range(0, len(pending), 2)]


def combine_tensors(pairs):
    if any(not torch.equal(pairs[0]["reference_mask"], p["reference_mask"]) for p in pairs[1:]):
        raise ValueError("Batched rows have different reference-face support")
    return {k: (pairs[0][k] if k == "reference_mask" else torch.cat([p[k] for p in pairs])) for k in pairs[0]}


@torch.no_grad()
def generate(model, tail, backend, config, rows, output):
    from extensions_built_in.diffusion_models.flux2.src.sampling import get_schedule

    target = config["data"]["target_size"][0] // 16
    tensors = []
    metadata = []
    for row in rows:
        pair, meta = load_pair(config, row, "cuda", negative=True)
        tensors.append(pair)
        metadata.append(meta)
    combined = combine_tensors(tensors)
    latent = torch.cat([torch.randn((1, 128, target, target), dtype=torch.bfloat16,
                                generator=torch.Generator().manual_seed(row["seed"])) for row in rows]).cuda()
    gates = []
    times = get_schedule(config["validation"]["steps"], target * target)
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    for current, following in zip(times[:-1], times[1:]):
        sigma = torch.tensor([current], dtype=torch.bfloat16, device="cuda")
        positive = backend.predict(model, combined, latent, sigma, config, branch=True)
        gates.append(torch.stack([block.reference_branch.last_gate.reshape(len(rows), target, target)
                                  for block in tail.blocks]).cpu())
        negative = backend.predict(model, combined, latent, sigma, config, branch=True, negative=True)
        latent += (following - current) * (negative + config["validation"]["guidance"] * (positive - negative))
    peak = torch.cuda.max_memory_reserved() / torch.cuda.get_device_properties(0).total_memory
    if peak >= config["training"]["max_reserved_fraction"]:
        raise RuntimeError(f"Validation CUDA reservation exceeded 90%: {peak:.1%}")
    site_gates = torch.stack(gates)
    samples = []
    for i, row in enumerate(rows):
        filename = row["sample_id"] + ".safetensors"
        selected = site_gates[:, :, i].contiguous()
        save_file({"latent": latent[i:i+1].cpu().contiguous(), "gates": selected.mean(1).contiguous(),
                   "site_gates": selected}, output / filename)
        samples.append({"sample_id": row["sample_id"], "identity_id": row["identity_id"],
                        "prompt": row["prompt"], "seed": row["seed"], "image": row["sample_id"] + ".png",
                        "reference_geometry": metadata[i]["vae"]["reference"], "seconds_per_pair": time.monotonic() - began,
                        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                        "reused_original_four_panel": False})
    return samples


def stage_latents(run, source, config, provenance, rows):
    backend = backend_module(config)
    model = backend.load_transformer(config)
    tail = install(model, provenance["training_identity"]["diagnostic"])
    for step in STEPS:
        if step:
            load_branch(tail, source / f"checkpoint-{step:06d}", provenance["training_identity"])
        output = run / f"validation-{step:06d}"
        output.mkdir(exist_ok=True)
        samples = []
        for row in rows:
            original = existing_sample(source, step, row)
            if not original:
                continue
            entry, location = original
            for name in (row["sample_id"] + ".png", row["sample_id"] + ".safetensors"):
                destination = output / name
                if not destination.exists():
                    os.link(location / name, destination)
                if file_hash(destination) != file_hash(location / name):
                    raise ValueError("Reused original output differs")
            samples.append({**entry, "reused_original_four_panel": True})
        for batch_index, batch in enumerate(sample_batches(rows, source, step), start=1):
            saved = output / f"batch-{batch_index:02d}.json"
            if saved.exists():
                completed = json.loads(saved.read_text())
                if all((output / s["sample_id"]).with_suffix(".safetensors").exists() for s in completed):
                    samples.extend(completed)
                    continue
            result = generate(model, tail, backend, config, batch, output)
            write_json(saved, result)
            samples.extend(result)
            print(f"Step {step}: {len(samples)}/24 latents ready", flush=True)
        by_id = {s["sample_id"]: s for s in samples}
        if len(by_id) != 24:
            raise RuntimeError("Validation stage is missing samples")
        report = {"backend": config["model"]["arch"], "mode": "branch_only", "variant": "face_suffix_24_seed01",
                  "target_size": config["data"]["target_size"], "reference_size": config["data"]["reference_size"],
                  "steps": config["validation"]["steps"], "panel_sha256": provenance["panel_sha256"],
                  "config_sha256": digest(config), "checkpoint": provenance["checkpoints"].get(str(step)),
                  "samples": [by_id[r["sample_id"]] for r in rows]}
        write_json(output / "validation.json", report)
        (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    del model, tail
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def decode(run, config, rows):
    backend = backend_module(config)
    vae = backend.load_vae(config)
    for step in STEPS:
        output = run / f"validation-{step:06d}"
        for row in rows:
            path = output / (row["sample_id"] + ".png")
            if path.exists():
                continue
            latent = load_file(output / (row["sample_id"] + ".safetensors"))["latent"]
            backend.decode(vae, latent, config).save(path)
            print(f"Decoded {step}: {row['sample_id']}", flush=True)
    del vae
    gc.collect()
    torch.cuda.empty_cache()


def score(run):
    python = Path(os.environ.get("BA_ENVS_DIR", str(ROOT / "envs"))) / "metrics/bin/python"
    env = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
    baseline = run / "validation-000000"
    subprocess.run([str(python), str(ROOT / "scripts/build_validation_output_masks.py"),
                    "--validation", str(baseline)], check=True, cwd=ROOT, env=env)
    for step in STEPS:
        output = run / f"validation-{step:06d}"
        if (output / "quality_summary.json").exists():
            continue
        subprocess.run([str(python), str(ROOT / "scripts/evaluate_metrics.py"),
                        "--validation", str(output), "--no-comet"], check=True, cwd=ROOT, env=env)


def report(run, rows):
    baseline = run / "validation-000000"
    from ba_dit.validation_masks import load_masks, mask_directory
    from ba_dit.config import load_config

    config = load_config(run / "resolved_config.yaml")
    masks = load_masks(config, [r["sample_id"] for r in rows])
    quality = {step: json.loads((run / f"validation-{step:06d}" / "quality_summary.json").read_text())["metrics"] for step in STEPS}
    per_image = {}
    for step in STEPS:
        with (run / f"validation-{step:06d}" / "quality_per_image.csv").open() as stream:
            per_image[step] = {r["sample_id"]: r for r in csv.DictReader(stream)}
    changes = []
    font = ImageFont.load_default()
    for seed in (0, 1):
        subset = [r for r in rows if r["seed"] == seed]
        sheet = Image.new("RGB", (3 * 386, len(subset) * 402 + 28), "#e9edf0")
        draw = ImageDraw.Draw(sheet)
        for col, step in enumerate(STEPS):
            draw.text((col * 386 + 6, 8), f"Step {step}", font=font, fill="black")
        for index, row in enumerate(subset):
            key = row["sample_id"]
            pictures = [np.asarray(Image.open(run / f"validation-{step:06d}" / (key + ".png")).convert("RGB")) for step in STEPS]
            maskfile = masks["samples"][key]["pixel_mask"]
            mask = np.asarray(Image.open(mask_directory(config) / maskfile)) > 0
            for col, pixels in enumerate(pictures):
                thumb = Image.fromarray(pixels).resize((384, 384), Image.Resampling.LANCZOS)
                sheet.paste(thumb, (col * 386, index * 402 + 28))
            draw.text((3, index * 402 + 412), key, font=font, fill="black")
            for step, pixels in zip(STEPS[1:], pictures[1:]):
                diff = np.abs(pixels.astype(np.float32) - pictures[0].astype(np.float32)).mean(-1)
                changes.append({"sample_id": key, "prompt_index": int(key.split("_")[1]), "seed": seed, "step": step,
                                "rgb_mean_abs": float(diff.mean()), "face_rgb_mean_abs": float(diff[mask].mean()) if mask.any() else None,
                                "background_rgb_mean_abs": float(diff[~mask].mean()) if (~mask).any() else None,
                                "id_sim": float(per_image[step][key]["id_sim"]), "text_sim": float(per_image[step][key]["text_sim"])})
        sheet.save(run / f"comparison_seed{seed}.jpg", quality=88)
    write_json(run / "image_changes.json", changes)
    summaries = {str(step): quality[step] for step in STEPS}
    write_json(run / "metric_summary.json", summaries)
    lines = ["# FLUX 4B face BA: 24-image one-ID validation", "", "Original 12 prompts, seed 0 and seed 1; fixed reference image, 768 px, 20 steps, CFG 4.",
             "The first four seed-0 outputs at each step are linked from the original completed run; all other outputs use the same model and checkpoints.",
             "Face boxes below are frozen from the step-zero generated images and used for scoring only.", "",
             "| Step | ID similarity | CLIP text similarity | Face missing | Unowned face |", "| ---: | ---: | ---: | ---: | ---: |"]
    for step in STEPS:
        values = quality[step]
        lines.append(f"| {step} | {values['id_sim']:.4f} | {values['text_sim']:.4f} | {values['id_sim_no_face']:.1%} | {values['id_sim_unowned']:.1%} |")
    for step in STEPS[1:]:
        subset = [r for r in changes if r["step"] == step]
        lines.append(f"\nStep {step} versus zero: RGB mean absolute {np.mean([r['rgb_mean_abs'] for r in subset]):.2f}/255 across 24 images.")
    lines += ["", "![Seed zero comparisons](comparison_seed0.jpg)", "", "![Seed one comparisons](comparison_seed1.jpg)", ""]
    (run / "report.md").write_text("\n".join(lines))
    return summaries, changes


def log_comet(config, run, provenance, rows, summaries, changes):
    exp = experiment(config, run, provenance)
    if not exp:
        raise RuntimeError("Comet must be enabled for this dedicated validation run")
    try:
        for step in STEPS:
            values = summaries[str(step)]
            exp.log_metrics({f"validation/{key}": value for key, value in values.items()}, step=step)
            if step:
                subset = [r for r in changes if r["step"] == step]
                exp.log_metrics({"comparison/rgb_mean_abs": float(np.mean([r["rgb_mean_abs"] for r in subset])),
                                 "comparison/face_rgb_mean_abs": float(np.mean([r["face_rgb_mean_abs"] for r in subset if r["face_rgb_mean_abs"] is not None])),
                                 "comparison/background_rgb_mean_abs": float(np.mean([r["background_rgb_mean_abs"] for r in subset if r["background_rgb_mean_abs"] is not None]))}, step=step)
            folder = run / f"validation-{step:06d}"
            for row in rows:
                key = row["sample_id"]
                exp.log_image(str(folder / (key + ".png")), name=f"one_id_24/{key}", step=step)
            for name in ("quality_per_image.csv", "quality_summary.json", "validation.json"):
                exp.log_asset(str(folder / name), file_name=f"step_{step:06d}_{name}")
        for name in ("metric_summary.json", "image_changes.json", "report.md", "comparison_seed0.jpg", "comparison_seed1.jpg"):
            exp.log_asset(str(run / name), file_name=name)
        for seed in (0, 1):
            exp.log_image(str(run / f"comparison_seed{seed}.jpg"), name=f"one_id_24/comparison_seed{seed}", step=2000)
    finally:
        exp.end()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/flux4b_face_one_id_24_val_20261001")
    parser.add_argument("--source-run", type=Path, default=TRAIN)
    args = parser.parse_args()
    run, source = args.run_dir.resolve(), args.source_run.resolve()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(8)
    spec, config, provenance, rows = initialize(run, source)
    exp = experiment(config, run, provenance)
    if exp:
        exp.end()
    subprocess.run([sys.executable, "-m", "ba_dit.cli", "precompute", "--config", str(run / "resolved_config.yaml"),
                    "--split", "validation", "--limit", "24"], check=True, cwd=ROOT)
    stage_latents(run, source, config, provenance, rows)
    decode(run, config, rows)
    score(run)
    summaries, changes = report(run, rows)
    log_comet(config, run, provenance, rows, summaries, changes)
    print(run / "report.md", flush=True)


if __name__ == "__main__":
    main()
