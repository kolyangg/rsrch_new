"""Benchmark and train a stronger, BA-only FLUX suffix on the local 16 GB GPU."""

import argparse
import gc
import json
import math
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
from ba_dit.face_diagnostic import target_face_tokens
from ba_dit.face_suffix import FaceSuffix, capture_prefix, identity, install, load_branch, objective, settings
from ba_dit.logging import connect, log_metrics
from ba_dit.runtime import backend_module


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def memory():
    reserved = torch.cuda.max_memory_reserved()
    return {"hardware/peak_reserved_gib": reserved / 2**30,
            "hardware/reserved_fraction": reserved / torch.cuda.get_device_properties(0).total_memory}


def memory_check(config):
    result = memory()
    if result["hardware/reserved_fraction"] >= config["training"]["max_reserved_fraction"]:
        raise RuntimeError(f"Exceeded CUDA memory gate: {result}")
    return result


def experiment(config, spec, run):
    exp = connect(config, run, name=run.name)
    if exp:
        exp.log_parameters({"model_identity": f"face_suffix_v1_shared_cfg:{digest(identity(config, spec))}",
            "trainable": "BA output LoRA and BA face routers only", "native_attention_lora_enabled": False,
            "cached_fixed_inputs": True, "branch_sites": list(range(20 - spec["blocks"], 20)),
            **{f"diagnostic/{key}": value for key, value in spec.items()}})
        if (run / "execution.json").exists():
            exp.log_parameters({f"execution/{k}": v for k, v in json.loads((run / "execution.json").read_text()).items()})
        exp.log_asset(str(run / "identity.json"), file_name="face_suffix_identity.json")
    return exp


@torch.no_grad()
def prepare(spec, config, run):
    backend = backend_module(config)
    rows = read_manifest(config["data"]["train_manifest"], training=True, limit=spec["train_images"])
    validation = read_manifest(config["data"]["validation_manifest"], limit=spec["validation_images"])
    assert_disjoint(rows, validation)
    if len({r["identity_id"] for r in rows + validation}) != 1:
        raise ValueError("Expected exactly one diagnostic identity")
    torch.manual_seed(spec["seed"])
    model = backend.load_transformer(config)
    tail = install(model, spec)
    save_file({k: v.cpu().contiguous() for k, v in tail.state_dict().items()}, run / "frozen_suffix.safetensors")
    write_json(run / "optimizer_inventory.json", {n: list(p.shape) for n, p in tail.trainable().items()})
    records = []
    (run / "cases").mkdir()
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    for row_index, row in enumerate(rows):
        tensors, _ = load_pair(config, row, "cuda")
        target = tensors["target_latent"]
        face = target_face_tokens(row, config)
        for noise_index in range(spec["fit_noise_seeds"] + 1):
            split = "probe" if noise_index == spec["fit_noise_seeds"] else "fit"
            for sigma_index, value in enumerate(spec["sigmas"]):
                seed = spec["seed"] + row_index * 1000 + noise_index * 100 + sigma_index
                noise = torch.randn(target.shape, dtype=target.dtype, generator=torch.Generator().manual_seed(seed)).cuda()
                sigma = torch.tensor([value], dtype=target.dtype, device="cuda")
                noisy = (1 - sigma) * target + sigma * noise
                case = capture_prefix(model, backend, tensors, noisy, sigma, config, spec)
                replay, _ = tail(case)
                # Check real full-model parity at both ends of the cache; every case
                # otherwise uses precisely the same frozen prefix/suffix operations.
                if not records or (row_index == len(rows) - 1 and split == "probe" and sigma_index == len(spec["sigmas"]) - 1):
                    full = backend.predict(model, tensors, noisy, sigma, config, branch=True).flatten(2).transpose(1, 2)
                    native = backend.predict(model, tensors, noisy, sigma, config, branch=False).flatten(2).transpose(1, 2)
                    if not torch.equal(full, replay) or not torch.equal(full, native):
                        raise RuntimeError(f"Initial native/cache parity failed: {(full - replay).abs().max()}")
                case.update(flow_target=(noise - target).flatten(2).transpose(1, 2), face_mask=face,
                            noisy=noisy, sigma=sigma, native_prediction=replay)
                path = f"cases/{split}_{row_index:02d}_{noise_index}_{sigma_index}.safetensors"
                save_file({k: v.cpu().contiguous() for k, v in case.items()}, run / path)
                records.append({"file": path, "sha256": file_hash(run / path), "split": split,
                                "row_index": row_index, "sigma": value, "seed": seed, "noise_index": noise_index})
                del case, replay, noisy, noise
        print(f"Prefix cache: {row_index + 1}/{len(rows)} pairs, {len(records)} fixed inputs", flush=True)
    write_json(run / "cache_manifest.json", {"identity": identity(config, spec), "rows": rows, "cases": records,
        "native_initial_parity": True, "full_cache_parity_first_and_last": True,
        "suffix_sha256": file_hash(run / "frozen_suffix.safetensors"), "seconds": time.monotonic() - began,
        **memory_check(config)})


def cases(run, spec, config, limit=None):
    manifest = json.loads((run / "cache_manifest.json").read_text())
    if manifest["identity"] != identity(config, spec):
        raise ValueError("Cache identity changed")
    if file_hash(run / "frozen_suffix.safetensors") != manifest["suffix_sha256"]:
        raise ValueError("Frozen suffix checksum failed")
    groups = {"fit": {}, "probe": {}}
    for record in manifest["cases"]:
        if limit is not None and record["row_index"] >= limit:
            continue
        path = run / record["file"]
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"Cache checksum failed: {path}")
        tensors = {k: v.pin_memory() for k, v in load_file(path).items() if k not in ("noisy", "sigma")}
        groups[record["split"]].setdefault(record["row_index"], []).append((record, tensors))
    return groups


def batch_to_gpu(chosen):
    # Reference support and sequence geometry must agree within a microbatch.
    if any(not torch.equal(chosen[0][k], c[k]) for c in chosen[1:] for k in ("refs", "layout")):
        raise ValueError("Cannot batch different reference layouts")
    return {k: torch.cat([c[k] for c in chosen]).cuda(non_blocking=True) for k in chosen[0]}


def update(tail, optimizer, batch, spec):
    optimizer.zero_grad(set_to_none=True)
    prediction, logits = tail(batch)
    loss, parts = objective(prediction, logits, batch, spec["router_weight"])
    loss.backward()
    params = list(tail.trainable().values())
    if any(p.grad is None for p in params):
        raise RuntimeError("A BA parameter has no gradient")
    norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    optimizer.step()
    return loss.detach(), norm.detach(), {k: v.detach() for k, v in parts.items()}


def benchmark(spec, config, run):
    backend_module(config)
    results = []
    # Independent processes avoid allocator fragmentation from failed candidates.
    for checkpointed in (False, True):
        for size in (1, 2, 4):
            command = [sys.executable, "-m", "scripts.face_suffix", "bench-one", "--config", str(run / "diagnostic.yaml"),
                       "--run-dir", str(run), "--batch", str(size)]
            if checkpointed:
                command.append("--checkpoint-blocks")
            subprocess.run(command, check=True, cwd=ROOT)
            record = json.loads((run / f"benchmark-b{size}-ckpt{int(checkpointed)}.json").read_text())
            results.append(record)
            if not record["safe"]:
                break
    valid = [r for r in results if r["safe"]]
    if not valid:
        raise RuntimeError("No safe local batch configuration")
    best = max(valid, key=lambda r: r["samples_per_second"])
    write_json(run / "benchmark.json", results)
    write_json(run / "execution.json", {k: best[k] for k in ("batch_size", "checkpoint_blocks", "seconds_per_update", "samples_per_second")})
    print(f"Selected local training settings: {best}", flush=True)


def bench_one(spec, config, run, size, checkpointed):
    backend_module(config)
    result = {"batch_size": size, "checkpoint_blocks": checkpointed, "safe": False}
    try:
        group = cases(run, spec, config, limit=1)["fit"][0]
        tail = FaceSuffix.from_file(run / "frozen_suffix.safetensors", spec)
        tail.checkpoint_blocks = checkpointed
        optimizer = torch.optim.AdamW(tail.trainable().values(), lr=spec["lr"], weight_decay=0., fused=True)
        batch = batch_to_gpu([group[i][1] for i in range(size)])
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        for i in range(7):
            torch.cuda.synchronize()
            began = time.monotonic()
            loss, norm, _ = update(tail, optimizer, batch, spec)
            torch.cuda.synchronize()
            if i >= 2:
                elapsed.append(time.monotonic() - began)
            memory_check(config)
        result.update(safe=True, seconds_per_update=sum(elapsed) / len(elapsed),
                      samples_per_second=size * len(elapsed) / sum(elapsed), loss=float(loss), gradient_norm=float(norm), **memory())
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        # Record actual failures; a non-memory error is fatal and must be repaired.
        if not isinstance(error, torch.cuda.OutOfMemoryError) and "memory gate" not in str(error):
            raise
        result.update(error=str(error).split("\n")[0], **memory())
    write_json(run / f"benchmark-b{size}-ckpt{int(checkpointed)}.json", result)
    print(json.dumps(result), flush=True)


@torch.no_grad()
def evaluate(tail, groups):
    metrics = {}
    for split, rows in groups.items():
        selected = [next(c for r, c in row if r["sigma"] == .5) for row in rows.values()]
        totals = {}
        for item in selected:
            case = batch_to_gpu([item])
            prediction, logits = tail(case)
            _, parts = objective(prediction, logits, case, 1.)
            gate, mask = logits.sigmoid().mean(0), case["face_mask"]
            activity = (prediction.float() - case["native_prediction"].float()).square().mean(-1).sqrt()
            values = {**parts, "gate_face": (gate * mask).sum() / mask.sum(),
                "gate_background": (gate * (1-mask)).sum() / (1-mask).sum(),
                "gate_iou": ((gate > .5) & (mask > .5)).sum() / (((gate > .5) | (mask > .5)).sum().clamp_min(1)),
                "delta_face": (activity * mask).sum() / mask.sum(),
                "delta_background": (activity * (1-mask)).sum() / (1-mask).sum()}
            for key, value in values.items():
                totals[key] = totals.get(key, 0.) + float(value) / len(selected)
        metrics.update({f"probe/{split}/{key}": value for key, value in totals.items()})
    return metrics


def chosen_batch(groups, step, size, seed):
    # Deterministic independent selection makes process resume exact, without RNG snapshots.
    generator = torch.Generator().manual_seed(seed + step * 7919)
    row = int(torch.randint(len(groups["fit"]), (1,), generator=generator))
    choices = groups["fit"][row]
    order = torch.randperm(len(choices), generator=generator)[:size]
    return batch_to_gpu([choices[int(i)][1] for i in order])


def save_checkpoint(tail, optimizer, spec, config, run, step):
    path = run / f"checkpoint-{step:06d}"
    path.mkdir(exist_ok=False)
    state = {k: v.detach().cpu().contiguous() for k, v in tail.branch_state().items()}
    save_file(state, path / "branch.safetensors")
    write_json(path / "manifest.json", {"identity": identity(config, spec), "step": step,
        "branch_sha256": file_hash(path / "branch.safetensors"),
        "execution": json.loads((run / "execution.json").read_text()),
        "trainable_parameters": sum(p.numel() for p in tail.trainable().values())})
    torch.save({"optimizer": optimizer.state_dict()}, path / "training_state.pt")
    load_branch(tail, path, identity(config, spec))
    if not all(torch.equal(v, tail.branch_state()[k].cpu()) for k, v in state.items()):
        raise RuntimeError("Checkpoint reload changed BA parameters")
    (run / "latest_checkpoint.txt").write_text(str(path) + "\n")


def set_lr(optimizer, spec, step):
    warm = min(1., step / max(1, spec["warmup"]))
    progress = max(0., (step - spec["warmup"]) / (spec["steps"] - spec["warmup"]))
    lr = spec["lr"] * warm * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train(spec, config, run, resume=None, until=None):
    backend_module(config)
    execution = json.loads((run / "execution.json").read_text())
    size = execution["batch_size"]
    tail = FaceSuffix.from_file(run / "frozen_suffix.safetensors", spec)
    tail.checkpoint_blocks = execution["checkpoint_blocks"]
    groups = cases(run, spec, config)
    params = tail.trainable()
    optimizer = torch.optim.AdamW(params.values(), lr=spec["lr"], weight_decay=0., fused=True)
    start = 0
    if resume:
        start = load_branch(tail, resume, identity(config, spec))["step"]
        optimizer.load_state_dict(torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=True)["optimizer"])
    exp = experiment(config, spec, run)
    torch.cuda.reset_peak_memory_stats()
    probes = json.loads((run / "probes.json").read_text()) if start else []
    count = sum(p.numel() for p in params.values())
    print(json.dumps({"trainable_parameters": count, "batch": size, "frozen_prefix_loaded": False,
                      "blocks": spec["blocks"], "checkpoint_blocks": tail.checkpoint_blocks, "start": start}), flush=True)
    try:
        if not start:
            probes.append({"step": 0, **evaluate(tail, groups)})
            write_json(run / "probes.json", probes)
            if exp:
                exp.log_metrics(probes[-1], step=0)
        finish = min(until or spec["steps"], spec["steps"])
        for step in range(start + 1, finish + 1):
            began = time.monotonic()
            batch = chosen_batch(groups, step, size, spec["seed"])
            lr = set_lr(optimizer, spec, step)
            loss, norm, parts = update(tail, optimizer, batch, spec)
            if step == 1:
                if any(not p.grad.count_nonzero() for n, p in params.items() if n.endswith("output_delta.b")):
                    raise RuntimeError("A BA site has no first-step output gradient")
                if any(p.grad is not None for p in tail.parameters() if not p.requires_grad):
                    raise RuntimeError("Frozen native parameters acquired gradients")
            torch.cuda.synchronize()
            measurements = {"train/loss": float(loss), "train/gradient_norm": float(norm), "train/lr": lr,
                "train/seconds": time.monotonic() - began, "train/parameters": count, "train/batch_size": size,
                **{f"train/{k}": float(v) for k, v in parts.items()}, **memory_check(config)}
            log_metrics(exp, run, measurements, step)
            if step == 26 and resume and start == 25:
                expected = load_file(run / "resume_expected.safetensors")
                exact = all(torch.equal(v, tail.branch_state()[k].cpu()) for k, v in expected.items())
                write_json(run / "resume_parity.json", {"step": 26, "exact": exact})
                if not exact:
                    raise RuntimeError("Process-resume trajectory differs at update 26")
            if step % spec["probe_every"] == 0 or step == finish:
                probes.append({"step": step, **evaluate(tail, groups)})
                write_json(run / "probes.json", probes)
                if exp:
                    exp.log_metrics(probes[-1], step=step)
                save_checkpoint(tail, optimizer, spec, config, run, step)
        # Before exiting at 25, retain the uninterrupted next update for the resume test.
        if finish == 25:
            set_lr(optimizer, spec, 26)
            update(tail, optimizer, chosen_batch(groups, 26, size, spec["seed"]), spec)
            save_file({k: v.detach().cpu().contiguous() for k, v in tail.branch_state().items()}, run / "resume_expected.safetensors")
        write_json(run / "training_summary.json", {"steps": finish, "resumed_from": start,
            "trainable_parameters": count, "only_ba_trainable": True, "all_gradients_finite": True,
            "all_sites_updated": all(bool(p.detach().count_nonzero()) for n, p in params.items() if n.endswith("output_delta.b")),
            "checkpoint_reload_exact": True, **execution, **memory_check(config)})
    finally:
        if exp:
            exp.end()


@torch.no_grad()
def infer(spec, config, run, checkpoint=None):
    backend = backend_module(config)
    from extensions_built_in.diffusion_models.flux2.src.sampling import get_schedule

    torch.manual_seed(spec["seed"])
    model = backend.load_transformer(config)
    tail = install(model, spec)
    step = load_branch(tail, checkpoint, identity(config, spec))["step"] if checkpoint else 0
    output = run / f"validation-{step:06d}"
    output.mkdir(exist_ok=False)
    if checkpoint:
        cache = json.loads((run / "cache_manifest.json").read_text())
        record = cache["cases"][0]
        case = {k: v.cuda() for k, v in load_file(run / record["file"]).items()}
        tensors, _ = load_pair(config, cache["rows"][record["row_index"]], "cuda")
        full = backend.predict(model, tensors, case["noisy"], case["sigma"], config, True).flatten(2).transpose(1, 2)
        replay, _ = tail(case)
        parity = {"step": step, "exact": torch.equal(full, replay), "max_abs": float((full.float()-replay.float()).abs().max())}
        write_json(run / "trained_cache_parity.json", parity)
        write_json(output / "cache_parity.json", parity)
        if not parity["exact"]:
            raise RuntimeError(f"Trained native/cache mismatch: {parity}")
        del case, tensors, full, replay
        gc.collect()
        torch.cuda.empty_cache()
    samples = []
    rows = read_manifest(config["data"]["validation_manifest"], limit=spec["validation_images"])
    for row in rows:
        torch.cuda.reset_peak_memory_stats()
        began = time.monotonic()
        tensors, metadata = load_pair(config, row, "cuda", negative=True)
        size = spec["resolution"] // 16
        latent = torch.randn((1, 128, size, size), dtype=torch.bfloat16, generator=torch.Generator().manual_seed(row["seed"])).cuda()
        times = get_schedule(spec["inference_steps"], size * size)
        gates = []
        for current, following in zip(times[:-1], times[1:]):
            sigma = torch.tensor([current], device="cuda", dtype=torch.bfloat16)
            positive = backend.predict(model, tensors, latent, sigma, config, branch=True)
            gates.append(torch.stack([b.reference_branch.last_gate.reshape(size, size) for b in tail.blocks]).cpu())
            negative = backend.predict(model, tensors, latent, sigma, config, branch=True, negative=True)
            latent += (following-current) * (negative + config["validation"]["guidance"] * (positive-negative))
        site_gates = torch.stack(gates)
        save_file({"latent": latent.cpu(), "gates": site_gates.mean(1), "site_gates": site_gates}, output / f"{row['sample_id']}.safetensors")
        samples.append({"sample_id": row["sample_id"], "identity_id": row["identity_id"], "prompt": row["prompt"],
            "seed": row["seed"], "image": f"{row['sample_id']}.png", "reference_geometry": metadata["vae"]["reference"],
            "seconds": time.monotonic()-began, "peak_cuda_reserved_gib": memory_check(config)["hardware/peak_reserved_gib"]})
        print(f"Validation {step}: {row['sample_id']} complete", flush=True)
    write_json(output / "validation.json", {"backend": config["model"]["arch"], "mode": "branch_only",
        "variant": "face_suffix_v1_shared_cfg", "target_size": config["data"]["target_size"],
        "reference_size": config["data"]["reference_size"], "steps": spec["inference_steps"],
        "panel_sha256": file_hash(config["data"]["validation_manifest"]), "config_sha256": digest(config),
        "checkpoint": {"path": str(checkpoint), "manifest_sha256": file_hash(Path(checkpoint)/"manifest.json")} if checkpoint else None,
        "samples": samples})
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


@torch.no_grad()
def decode(spec, config, run, step):
    backend = backend_module(config)
    output = run / f"validation-{step:06d}"
    report = json.loads((output / "validation.json").read_text())
    vae = backend.load_vae(config)
    exp = experiment(config, spec, run)
    try:
        for row in report["samples"]:
            backend.decode(vae, load_file(output / f"{row['sample_id']}.safetensors")["latent"], config).save(output / row["image"])
            if exp:
                exp.log_image(str(output / row["image"]), name=f"face_suffix/{row['sample_id']}", step=step)
    finally:
        if exp:
            exp.end()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "prepare", "benchmark", "bench-one", "train", "infer", "decode"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/flux4b_face_one_id_strong.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--until", type=int)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--checkpoint-blocks", action="store_true")
    args = parser.parse_args()
    spec, config = settings(args.config)
    run = args.run_dir.resolve()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(8)
    if args.command == "run":
        run.mkdir(parents=True, exist_ok=False)
        (run / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (run / "diagnostic.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
        write_json(run / "identity.json", identity(config, spec))
        exp = experiment(config, spec, run)
        if exp:
            exp.end()
        for split, limit in (("train", spec["train_images"]), ("validation", spec["validation_images"])):
            subprocess.run([sys.executable, "-m", "ba_dit.cli", "precompute", "--config", str(run / "resolved_config.yaml"),
                            "--split", split, "--limit", str(limit)], check=True, cwd=ROOT)
        phases = [("prepare", []), ("benchmark", []), ("infer", []), ("decode", []), ("train", ["--until", "25"])]
        previous = 25
        for stop in (spec["validation_every"], spec["steps"]):
            phases.extend([("train", ["--checkpoint", str(run/f"checkpoint-{previous:06d}"), "--until", str(stop)]),
                           ("infer", ["--checkpoint", str(run/f"checkpoint-{stop:06d}")]),
                           ("decode", ["--step", str(stop)])])
            previous = stop
        for command, extra in phases:
            subprocess.run([sys.executable, "-m", "scripts.face_suffix", command, "--config", str(run/"diagnostic.yaml"),
                            "--run-dir", str(run), *extra], check=True, cwd=ROOT)
    else:
        if json.loads((run/"identity.json").read_text()) != identity(config, spec):
            raise ValueError("Diagnostic source/config changed after initialization")
        if args.command == "prepare": prepare(spec, config, run)
        elif args.command == "benchmark": benchmark(spec, config, run)
        elif args.command == "bench-one": bench_one(spec, config, run, args.batch, args.checkpoint_blocks)
        elif args.command == "train": train(spec, config, run, args.checkpoint, args.until)
        elif args.command == "infer": infer(spec, config, run, args.checkpoint)
        elif args.command == "decode": decode(spec, config, run, args.step)


if __name__ == "__main__":
    main()
