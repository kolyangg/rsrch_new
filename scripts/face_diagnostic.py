"""Run the isolated, cached final-block FLUX face-BA one-ID diagnostic."""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

from ba_dit.config import ROOT, digest
from ba_dit.data.cache import load_pair
from ba_dit.data.manifest import assert_disjoint, file_hash, read_manifest
from ba_dit.face_diagnostic import (CachedFaceTail, capture_case, identity, install,
                                    load_branch, objective, settings, target_face_tokens)
from ba_dit.logging import connect, log_metrics
from ba_dit.runtime import backend_module


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def memory_check(config):
    used = torch.cuda.max_memory_reserved()
    fraction = used / torch.cuda.get_device_properties(0).total_memory
    if fraction >= config["training"]["max_reserved_fraction"]:
        raise RuntimeError(f"CUDA memory gate failed: {fraction:.3%}")
    return {"hardware/peak_reserved_gib": used / 2**30, "hardware/reserved_fraction": fraction}


def experiment(config, spec, run):
    exp = connect(config, run, name=run.name)
    if exp:
        exp.log_parameters({"model_identity": f"final_single_face_read_v2_shared_cfg:{digest(identity(config, spec))}",
                            "mode": "face_branch_only", "diagnostic": True,
                            "cached_fixed_inputs": True, "branch_site": "single_blocks.19",
                            "training/batch_size": spec["batch_size"], "native_attention_lora_enabled": False,
                            "trainable": "BA output LoRA and BA face router only"})
        exp.log_asset(str(run / "identity.json"), file_name="face_diagnostic_identity.json")
    return exp


def prepare(spec, config, run):
    backend = backend_module(config)
    rows = read_manifest(config["data"]["train_manifest"], training=True, limit=spec["train_images"])
    validation = read_manifest(config["data"]["validation_manifest"], limit=spec["validation_images"])
    assert_disjoint(rows, validation)
    if len({row["identity_id"] for row in rows + validation}) != 1:
        raise ValueError("This diagnostic requires exactly one identity")
    torch.manual_seed(spec["seed"])
    model = backend.load_transformer(config)
    branch = install(model, spec)
    tail = CachedFaceTail(model, spec).cpu()
    save_file(tail.state_dict(), run / "frozen_tail.safetensors")
    tail.cuda().eval()
    inventory = {n: list(p.shape) for n, p in model.named_parameters() if p.requires_grad}
    write_json(run / "optimizer_inventory.json", inventory)
    records = []
    (run / "cases").mkdir()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for split, seed_offset in (("fit", 0), ("probe", 10000)):
        for row_index, row in enumerate(rows):
            tensors, _ = load_pair(config, row, "cuda")
            target = tensors["target_latent"]
            for sigma_index, value in enumerate(spec["sigmas"]):
                seed = spec["seed"] + seed_offset + row_index * 101 + sigma_index
                noise = torch.randn(target.shape, dtype=target.dtype, generator=torch.Generator().manual_seed(seed)).cuda()
                sigma = torch.tensor([value], device="cuda", dtype=target.dtype)
                noisy = (1 - sigma) * target + sigma * noise
                cached, prediction = capture_case(model, backend, tensors, noisy, sigma, config)
                case = {k: v.cuda() for k, v in cached.items()}
                with torch.no_grad():
                    replay, _ = tail(case)
                expected = prediction.flatten(2).transpose(1, 2)
                error = float((expected.float() - replay.float()).abs().max())
                if not torch.equal(expected, replay):
                    raise RuntimeError(f"Cached tail does not exactly reproduce full pipeline: {error}")
                if not records:
                    with torch.no_grad():
                        native = backend.predict(model, tensors, noisy, sigma, config, branch=False)
                    if not torch.equal(native, prediction):
                        raise RuntimeError("Untrained face branch changed native prediction")
                cached.update(flow_target=(noise - target).flatten(2).transpose(1, 2).cpu().contiguous(),
                              face_mask=target_face_tokens(row, config), noisy=noisy.cpu(), sigma=sigma.cpu())
                filename = f"cases/{split}_{row_index:02d}_{sigma_index}.safetensors"
                save_file(cached, run / filename)
                records.append({"file": filename, "sha256": file_hash(run / filename), "split": split,
                                "sample_id": row["sample_id"], "row_index": row_index, "sigma": value, "seed": seed,
                                "cache_full_max_abs": error})
                print(f"Cached {split} {row_index + 1}/{len(rows)}, sigma {value}: exact full-pipeline parity", flush=True)
                del cached, case, prediction, replay, noisy, noise
            del tensors
    write_json(run / "cache_manifest.json", {"identity": identity(config, spec), "rows": rows,
        "native_initial_parity": True, "cases": records, "seconds": time.monotonic() - started,
        "tail_sha256": file_hash(run / "frozen_tail.safetensors"), **memory_check(config)})


def cache_cases(spec, config, run):
    manifest = json.loads((run / "cache_manifest.json").read_text())
    if manifest["identity"] != identity(config, spec) or manifest["tail_sha256"] != file_hash(run / "frozen_tail.safetensors"):
        raise ValueError("Frozen cache source/config/weights identity changed")
    groups = {"fit": [], "probe": []}
    for record in manifest["cases"]:
        path = run / record["file"]
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"Cache integrity failed: {path}")
        groups[record["split"]].append({k: v.cuda() for k, v in load_file(path).items() if k not in ("noisy", "sigma")})
    return groups


@torch.no_grad()
def evaluate(tail, groups):
    metrics = {}
    for split, cases in groups.items():
        totals = {}
        for case in cases:
            prediction, logits = tail(case)
            native, _ = tail(case, branch=False)
            _, losses = objective(prediction, logits, case, 1.0)
            gate, mask = logits.sigmoid(), case["face_mask"]
            activity = (prediction.float() - native.float()).square().mean(-1).sqrt()
            values = {**losses, "gate_face": (gate * mask).sum() / mask.sum(),
                      "gate_background": (gate * (1 - mask)).sum() / (1 - mask).sum(),
                      "gate_iou": ((gate > .5) & (mask > .5)).sum() / (((gate > .5) | (mask > .5)).sum().clamp_min(1)),
                      "delta_face": (activity * mask).sum() / mask.sum(),
                      "delta_background": (activity * (1 - mask)).sum() / (1 - mask).sum()}
            for key, value in values.items():
                totals[key] = totals.get(key, 0.) + float(value) / len(cases)
        metrics.update({f"probe/{split}/{key}": value for key, value in totals.items()})
    return metrics


def train(spec, config, run, resume=None, until=None):
    backend_module(config)  # Pins/imports only; the backbone is not loaded.
    torch.manual_seed(spec["seed"])
    tail = CachedFaceTail.from_file(run / "frozen_tail.safetensors", spec)
    groups = cache_cases(spec, config, run)
    trainable = [p for p in tail.parameters() if p.requires_grad]
    assert all(n.startswith("reference_branch.") for n, p in tail.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(trainable, lr=spec["lr"], weight_decay=0., fused=True)
    start = 0
    if resume:
        manifest = load_branch(tail.reference_branch, resume, identity(config, spec))
        state = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        start = manifest["step"]
    torch.cuda.reset_peak_memory_stats()
    exp = experiment(config, spec, run)
    parameter_count = sum(p.numel() for p in trainable)
    print(json.dumps({"trainable_parameters": parameter_count, "full_backbone_loaded": False,
                      "microbatch": spec["batch_size"], "accumulation": 1, "start": start}), flush=True)
    probes = json.loads((run / "probes.json").read_text()) if start else []
    try:
        if not start:
            baseline = {"step": 0, **evaluate(tail, groups)}
            probes.append(baseline)
            if exp:
                exp.log_metrics({k: v for k, v in baseline.items() if k != "step"}, step=0)
        finish = min(until or spec["steps"], spec["steps"])
        for step in range(start + 1, finish + 1):
            began = time.monotonic()
            chosen = [groups["fit"][((step - 1) * spec["batch_size"] + j) % len(groups["fit"])] for j in range(spec["batch_size"])]
            batch = {k: torch.cat([case[k] for case in chosen]) for k in chosen[0]}
            optimizer.zero_grad(set_to_none=True)
            prediction, logits = tail(batch)
            loss, parts = objective(prediction, logits, batch, spec["router_weight"])
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
                raise RuntimeError("Missing/nonfinite face BA gradients")
            if step == 1 and not tail.reference_branch.output_delta.b.grad.count_nonzero():
                raise RuntimeError("Missing first BA output B gradient")
            if any(p.grad is not None for p in tail.parameters() if not p.requires_grad):
                raise RuntimeError("A frozen native parameter acquired gradients")
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            torch.cuda.synchronize()
            metrics = {"train/loss": float(loss), "train/gradient_norm": float(norm),
                       "train/seconds": time.monotonic() - began,
                       "train/b_norm": float(tail.reference_branch.output_delta.b.norm()),
                       "train/parameters": parameter_count, **memory_check(config),
                       **{f"train/{k}": float(v) for k, v in parts.items()}}
            log_metrics(exp, run, metrics, step)
            if step % spec["probe_every"] == 0 or step == finish:
                measured = {"step": step, **evaluate(tail, groups)}
                probes.append(measured)
                write_json(run / "probes.json", probes)
                if exp:
                    exp.log_metrics({k: v for k, v in measured.items() if k != "step"}, step=step)
                destination = run / f"checkpoint-{step:06d}"
                destination.mkdir(exist_ok=False)
                state = {k: v.detach().cpu().contiguous() for k, v in tail.reference_branch.state_dict().items()}
                save_file(state, destination / "branch.safetensors")
                write_json(destination / "manifest.json", {"identity": identity(config, spec), "step": step,
                           "branch_sha256": file_hash(destination / "branch.safetensors"), "trainable_parameters": parameter_count})
                torch.save({"optimizer": optimizer.state_dict()}, destination / "training_state.pt")
                (run / "latest_checkpoint.txt").write_text(str(destination) + "\n")
                before = {k: v.clone() for k, v in tail.reference_branch.state_dict().items()}
                load_branch(tail.reference_branch, destination, identity(config, spec))
                assert all(torch.equal(v, tail.reference_branch.state_dict()[k]) for k, v in before.items())
        write_json(run / "training_summary.json", {"steps": finish, "resumed_from": start, "trainable_parameters": parameter_count,
                   "all_gradients_finite": True, "checkpoint_reload_exact": True, **memory_check(config)})
    finally:
        if exp:
            exp.end()


@torch.no_grad()
def infer(spec, config, run, checkpoint=None):
    backend = backend_module(config)
    from extensions_built_in.diffusion_models.flux2.src.sampling import get_schedule

    torch.manual_seed(spec["seed"])
    model = backend.load_transformer(config)
    branch = install(model, spec)
    step = load_branch(branch, checkpoint, identity(config, spec))["step"] if checkpoint else 0
    output = run / f"validation-{step:06d}"
    output.mkdir(exist_ok=False)
    # Verify trained cached-tail/full-model equality with exactly the saved noisy input.
    if checkpoint:
        cache_manifest = json.loads((run / "cache_manifest.json").read_text())
        record = cache_manifest["cases"][0]
        case = {k: v.cuda() for k, v in load_file(run / record["file"]).items()}
        tail = CachedFaceTail.from_file(run / "frozen_tail.safetensors", spec)
        load_branch(tail.reference_branch, checkpoint, identity(config, spec))
        tensors, _ = load_pair(config, cache_manifest["rows"][record["row_index"]], "cuda")
        full = backend.predict(model, tensors, case["noisy"], case["sigma"], config, True).flatten(2).transpose(1, 2)
        replay, _ = tail(case)
        error = float((full.float() - replay.float()).abs().max())
        write_json(run / "trained_cache_parity.json", {"exact": torch.equal(full, replay), "max_abs": error, "step": step})
        if not torch.equal(full, replay):
            raise RuntimeError(f"Trained cached and full predictions disagree: {error}")
        del tail, case, tensors, full, replay
        gc.collect()
        torch.cuda.empty_cache()
    rows = read_manifest(config["data"]["validation_manifest"], limit=spec["validation_images"])
    samples = []
    for row in rows:
        torch.cuda.reset_peak_memory_stats()
        began = time.monotonic()
        tensors, metadata = load_pair(config, row, "cuda", negative=True)
        size = spec["resolution"] // 16
        latent = torch.randn((1, 128, size, size), dtype=torch.bfloat16,
                             generator=torch.Generator().manual_seed(row["seed"])).cuda()
        times = get_schedule(spec["inference_steps"], size * size)
        gates = []
        for current, following in zip(times[:-1], times[1:]):
            sigma = torch.tensor([current], device="cuda", dtype=torch.bfloat16)
            positive = backend.predict(model, tensors, latent, sigma, config, branch=True)
            gates.append(branch.last_gate.reshape(size, size).cpu())
            # Both CFG lanes use the same model/adapters, like ordinary LoRA.
            # Positive-only BA would multiply the common correction by guidance.
            negative = backend.predict(model, tensors, latent, sigma, config, branch=True, negative=True)
            velocity = negative + config["validation"]["guidance"] * (positive - negative)
            latent = latent + (following - current) * velocity
        save_file({"latent": latent.cpu(), "gates": torch.stack(gates)}, output / f"{row['sample_id']}.safetensors")
        sample = {"sample_id": row["sample_id"], "identity_id": row["identity_id"], "prompt": row["prompt"], "seed": row["seed"],
                  "image": f"{row['sample_id']}.png", "reference_geometry": metadata["vae"]["reference"],
                  "seconds": time.monotonic() - began, "peak_cuda_reserved_gib": memory_check(config)["hardware/peak_reserved_gib"]}
        samples.append(sample)
        print(f"Validation step {step}: {row['sample_id']} complete", flush=True)
    write_json(output / "validation.json", {"backend": config["model"]["arch"], "mode": "branch_only",
        "variant": "final_single_face_read_v2_shared_cfg", "target_size": config["data"]["target_size"],
        "reference_size": config["data"]["reference_size"], "steps": spec["inference_steps"],
        "panel_sha256": file_hash(config["data"]["validation_manifest"]), "config_sha256": digest(config),
        "checkpoint": {"path": str(checkpoint), "manifest_sha256": file_hash(Path(checkpoint) / "manifest.json")} if checkpoint else None,
        "samples": samples})
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def decode(spec, config, run, step):
    backend = backend_module(config)
    output = run / f"validation-{step:06d}"
    report = json.loads((output / "validation.json").read_text())
    vae = backend.load_vae(config)
    exp = experiment(config, spec, run)
    try:
        for row in report["samples"]:
            latent = load_file(output / f"{row['sample_id']}.safetensors")["latent"]
            backend.decode(vae, latent, config).save(output / row["image"])
            if exp:
                exp.log_image(str(output / row["image"]), name=f"face_diagnostic/{row['sample_id']}", step=step)
        if exp:
            exp.log_asset(str(output / "validation.json"), file_name=f"face_validation_{step:06d}.json")
    finally:
        if exp:
            exp.end()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "prepare", "train", "infer", "decode"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/flux4b_face_one_id_fast.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--until", type=int)
    args = parser.parse_args()
    spec, config = settings(args.config)
    run = args.run_dir.resolve()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    if args.command == "run":
        run.mkdir(parents=True, exist_ok=False)
        (run / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        write_json(run / "identity.json", identity(config, spec))
        exp = experiment(config, spec, run)
        if exp:
            exp.end()
        for split, limit in (("train", spec["train_images"]), ("validation", spec["validation_images"])):
            subprocess.run([sys.executable, "-m", "ba_dit.cli", "precompute", "--config", str(run / "resolved_config.yaml"),
                            "--split", split, "--limit", str(limit)], check=True, cwd=ROOT)
        first = spec["probe_every"]
        for command, extra in (("prepare", []), ("infer", []), ("decode", []), ("train", ["--until", str(first)]),
                               ("train", ["--checkpoint", str(run / f"checkpoint-{first:06d}")]),
                               ("infer", ["--checkpoint", str(run / f"checkpoint-{spec['steps']:06d}")]),
                               ("decode", ["--step", str(spec["steps"])])):
            subprocess.run([sys.executable, "-m", "scripts.face_diagnostic", command, "--config", str(args.config),
                            "--run-dir", str(run), *extra], check=True, cwd=ROOT)
    else:
        if json.loads((run / "identity.json").read_text()) != identity(config, spec):
            raise ValueError("Diagnostic source/config changed since this run started")
        if args.command == "prepare": prepare(spec, config, run)
        elif args.command == "train": train(spec, config, run, args.checkpoint, args.until)
        elif args.command == "infer": infer(spec, config, run, args.checkpoint)
        else: decode(spec, config, run, args.step)


if __name__ == "__main__":
    main()
