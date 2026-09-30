"""Cached validation; denoising and VAE decode run in separate processes."""

import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ba_dit import adapters
from ba_dit.checkpoint import load_adapters
from ba_dit.config import digest
from ba_dit.data.cache import cache_path, load_pair
from ba_dit.data.manifest import file_hash, read_manifest
from ba_dit.logging import connect
from ba_dit.runtime import backend_module


def denoise(config, mode, output_dir, checkpoint=None, limit=None, compare_native=False, resume=False):
    if compare_native and checkpoint and mode in {"lora_only", "lora_plus_branch"}:
        raise ValueError("A trained LoRA remains active when the branch is disabled. Run mode=native separately for that control.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(config["data"]["validation_manifest"], limit=limit or config["validation"]["limit"])
    backend = backend_module(config)
    torch.manual_seed(config["training"]["seed"])
    model = backend.load_transformer(config)
    inventory = adapters.install(model, config, mode)
    if checkpoint:
        load_adapters(model, checkpoint, config, mode)
    model.eval()
    results = []
    report_path = output_dir / "validation.json"
    if resume and report_path.exists():
        report = json.loads(report_path.read_text())
        if report["mode"] != mode or report["config_sha256"] != digest(config):
            raise ValueError("Validation resume mode/config differs")
        expected_checkpoint = file_hash(Path(checkpoint) / "manifest.json") if checkpoint else None
        if (report.get("checkpoint") or {}).get("manifest_sha256") != expected_checkpoint:
            raise ValueError("Validation resume checkpoint differs")
        results = report["samples"]
        previous_comparison = report.get("compare_native", any(row["parity"] is not None for row in results))
        if previous_comparison != compare_native:
            raise ValueError("Validation resume requires the same native-comparison setting")
        if [row["sample_id"] for row in results] != [row["sample_id"] for row in rows[:len(results)]]:
            raise ValueError("Validation order changed")
        for row in results:
            if not (output_dir / f"{row['sample_id']}.safetensors").exists():
                raise FileNotFoundError(f"Missing completed latent: {row['sample_id']}")
            if compare_native and not (output_dir / f"{row['sample_id']}.native.safetensors").exists():
                raise FileNotFoundError(f"Missing completed native comparison: {row['sample_id']}")
    pending = rows[len(results):]
    while pending:
        group = [pending.pop(0)]
        if config["model"]["backend"] == "flux":
            reference_key = cache_path(config, group[0], "vae")
            while pending and len(group) < config["validation"].get("batch_size", 1) and cache_path(config, pending[0], "vae") == reference_key:
                group.append(pending.pop(0))
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        loaded = [load_pair(config, row, "cuda", negative=config["validation"]["guidance"] > 1) for row in group]
        tensors = loaded[0][0]
        if len(group) > 1:
            if any(not torch.equal(tensors["reference_mask"], item[0]["reference_mask"]) for item in loaded[1:]):
                raise ValueError("Batched FLUX references must share the exact token mask")
            tensors = {key: value if key == "reference_mask" else torch.cat([item[0][key] for item in loaded])
                       for key, value in tensors.items()}
        branch = mode in {"branch_only", "lora_plus_branch"}
        with torch.inference_mode():
            latent = (backend.sample_batch(model, tensors, config, [row["seed"] for row in group], branch)
                      if len(group) > 1 else backend.sample(model, tensors, config, group[0]["seed"], branch))
            baseline = None
            if compare_native:
                baseline = (backend.sample_batch(model, tensors, config, [row["seed"] for row in group], branch=False)
                            if len(group) > 1 else backend.sample(model, tensors, config, group[0]["seed"], branch=False))
        torch.cuda.synchronize()
        elapsed = (time.monotonic() - started) / len(group)
        peak = torch.cuda.max_memory_reserved() / 2**30
        for index, row in enumerate(group):
            sample = latent[index:index + 1]
            save_file({"latent": sample.cpu().contiguous()}, output_dir / f"{row['sample_id']}.safetensors")
            parity = None
            if baseline is not None:
                native = baseline[index:index + 1]
                parity = {"max_abs": float((sample.float() - native.float()).abs().max()), "exact": torch.equal(sample, native)}
                if checkpoint is None and not parity["exact"]:
                    raise RuntimeError(f"Untrained branch changed native output: {parity}")
                save_file({"latent": native.cpu().contiguous()}, output_dir / f"{row['sample_id']}.native.safetensors")
            record = {"sample_id": row["sample_id"], "identity_id": row["identity_id"], "prompt": row["prompt"],
                      "seed": row["seed"], "image": f"{row['sample_id']}.png", "seconds": elapsed,
                      "peak_cuda_reserved_gib": peak, "parity": parity,
                      "reference_geometry": loaded[index][1]["vae"]["reference"]}
            results.append(record)
            temporary_report = report_path.with_suffix(".json.tmp")
            temporary_report.write_text(json.dumps({"backend": config["model"]["arch"], "mode": mode, "compare_native": compare_native,
                "steps": config["validation"]["steps"], "batch_size": config["validation"].get("batch_size", 1),
                "target_size": config["data"]["target_size"], "reference_size": config["data"]["reference_size"],
                "config_sha256": digest(config), "panel_sha256": file_hash(config["data"]["validation_manifest"]),
                "checkpoint": {"path": str(checkpoint), "manifest_sha256": file_hash(Path(checkpoint) / "manifest.json")} if checkpoint else None,
                "adapter_inventory": inventory, "samples": results}, indent=2) + "\n")
            temporary_report.replace(report_path)
            print(json.dumps(record), flush=True)
        del loaded, tensors, latent


def decode(config, output_dir, log_dir=None, step=0):
    output_dir = Path(output_dir)
    report = json.loads((output_dir / "validation.json").read_text())
    backend = backend_module(config)
    vae = backend.load_vae(config)
    experiment = connect(config, log_dir or output_dir, name=None if log_dir else output_dir.name)
    try:
        for row in report["samples"]:
            latent = load_file(output_dir / f"{row['sample_id']}.safetensors")["latent"]
            image = backend.decode(vae, latent, config)
            image.save(output_dir / row["image"])
            native = output_dir / f"{row['sample_id']}.native.safetensors"
            if native.exists():
                native_image = backend.decode(vae, load_file(native)["latent"], config)
                native_image.save(output_dir / f"{row['sample_id']}.native.png")
            if experiment:
                experiment.log_image(str(output_dir / row["image"]), name=f"manual_val/{row['sample_id']}", step=step,
                                     metadata={"prompt": row["prompt"], "identity_id": row["identity_id"], "seed": row["seed"]})
        if experiment:
            experiment.log_asset(str(output_dir / "validation.json"), file_name=f"validation_{step:06d}.json")
            experiment.log_metric("validation/images", len(report["samples"]), step=step)
    finally:
        if experiment:
            experiment.end()
