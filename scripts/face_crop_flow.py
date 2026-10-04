"""Controlled BA-only face reconstruction from noise with frozen FLUX features."""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw
from safetensors.torch import load_file, save_file

from ba_dit.config import ROOT, adapter_identity, digest, load_config
from ba_dit.data.cache import load_pair
from ba_dit.data.manifest import file_hash, read_manifest
from ba_dit.logging import connect, log_metrics
from ba_dit.nn.face_crop_flow import FaceCropFlow, capture, install
from ba_dit.runtime import backend_module


STEPS = (0, 1000, 2000)
SIGMAS = (.05, .2, .4, .6, .8, 1.)
SOURCES = ("scripts/face_crop_flow.py", "ba_dit/nn/face_crop_flow.py")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def memory():
    used = torch.cuda.max_memory_reserved()
    fraction = used / torch.cuda.get_device_properties(0).total_memory
    if fraction >= .9:
        raise RuntimeError(f"CUDA memory limit exceeded: {fraction:.1%}")
    return {"hardware/peak_reserved_gib": used / 2**30, "hardware/reserved_fraction": fraction}


def experiment(config, run):
    exp = connect(config, run, name=run.name)
    identity = json.loads((run / "identity.json").read_text())
    exp.log_parameters({"model_identity": identity["variant"] + ":" + digest(identity),
                        "native_velocity_used": False, "native_attention_lora_enabled": False,
                        "BA_query_residual": identity["query_residual"], "BA_noise_skip": identity["noise_skip"], "live_metric_stride": 25,
                        "BA_timestep_scaling": identity["timestep_scaling"],
                        "trainable_parameters": identity["trainable_parameters"], "BA_width": 256, "BA_heads": 4,
                        "initialization": "random Q/K/V; zero output", "training/batch_size": 8,
                        "loss": identity["loss"], "fixed_input_cache": True})
    return exp


def crop(path, box, destination):
    image = Image.open(path).convert("RGB")
    side = min(round(max(box[2]-box[0], box[3]-box[1]) * 1.4), *image.size)
    left = max(0, min(image.width-side, round((box[0]+box[2]-side)/2)))
    top = max(0, min(image.height-side, round((box[1]+box[3]-side)/2)))
    image.crop((left, top, left+side, top+side)).resize((256, 256), Image.Resampling.LANCZOS).save(destination)
    transformed = [(box[i] - (left if i % 2 == 0 else top)) * 256 / side for i in range(4)]
    return transformed


def initialize(run, query_residual=False, noise_skip=False, timestep_scaling=True):
    run.mkdir(parents=True, exist_ok=False)
    data = run / "data"
    data.mkdir()
    config = load_config(ROOT / "configs/flux4b_48_one_id.yaml")
    source_train = read_manifest(config["data"]["train_manifest"], training=True, limit=4)
    source_ref = read_manifest(config["data"]["validation_manifest"])[0]
    ref_box = crop(source_ref["reference"], source_ref["reference_box"], data / "reference.png")
    common = {"identity_id": source_ref["identity_id"], "reference_image": str(data / "reference.png"),
              "reference_face_bbox": ref_box, "prompt": "A close-up photograph of a person's face.",
              "split_policy": "one_id_diagnostic"}
    train = []
    for i, row in enumerate(source_train):
        if row["identity_id"] != common["identity_id"]:
            raise ValueError("The diagnostic must use one identity")
        box = crop(row["target"], row["target_box"], data / f"target_{i}.png")
        train.append({**common, "sample_id": f"crop_train_{i}", "target_image": str(data / f"target_{i}.png"), "target_face_box": box})
    validation = [{**common, "sample_id": f"crop_val_{i}", "seed": i} for i in range(4)]
    for split, rows in (("train", train), ("validation", validation)):
        (data / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    shutil.copyfile(ROOT / "data/datasets/one_id/id_embeds_one_id.pth", data / "id_embeds_one_id.pth")
    config["name"] = "flux4b_ba_only_face_crop"
    config["data"].update(train_manifest=str(data / "train.jsonl"), validation_manifest=str(data / "validation.jsonl"),
                          target_size=[256, 256], reference_size=256, train_limit=4)
    config["validation"].update(limit=4, steps=20, guidance=1.)
    config["hardware"]["min_vram_gb"] = 12
    config["branch"].update(rank=256, alpha=256)
    config["training"].update(steps=2000, lr=.002, warmup=100, grad_accum=1, seed=142,
                              checkpoint_every=1000, validation_every=1000, gradient_checkpointing=False)
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    torch.manual_seed(142)
    branch = FaceCropFlow(query_residual=query_residual, noise_skip=noise_skip, timestep_scaling=timestep_scaling)
    save_file(branch.state_dict(), run / "initial.safetensors")
    identity = {"variant": "ba_only_face_crop_v1", "base": adapter_identity(config),
                "config_sha256": digest(config), "source_sha256": {p: file_hash(ROOT / p) for p in SOURCES},
                "initial_sha256": file_hash(run / "initial.safetensors"), "train_manifest_sha256": file_hash(data / "train.jsonl"),
                "validation_manifest_sha256": file_hash(data / "validation.jsonl"),
                "source_train": source_train, "source_reference": source_ref,
                "dataset_sha256": {str(p.relative_to(run)): file_hash(p) for p in data.iterdir()},
                "trainable_parameters": sum(p.numel() for p in branch.parameters()),
                "native_velocity_used": False, "query_residual": query_residual, "noise_skip": noise_skip, "target_gate": False,
                "timestep_scaling": timestep_scaling,
                "loss": "native unweighted flow MSE", "sigmas": SIGMAS, "fit_noise_seeds": 2, "probe_noise_seeds": 1}
    write(run / "identity.json", identity)
    write(run / "ownership_boxes.json", {r["sample_id"]: [0, 0, 256, 256] for r in validation})
    (run / "source_snapshot").mkdir()
    for p in SOURCES:
        destination = run / "source_snapshot" / p
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / p, destination)
    exp = experiment(config, run)
    exp.log_asset(str(run / "identity.json"))
    exp.end()


def verify(run):
    config = load_config(run / "resolved_config.yaml")
    identity = json.loads((run / "identity.json").read_text())
    if identity["config_sha256"] != digest(config) or any(file_hash(ROOT / p) != h for p, h in identity["source_sha256"].items()):
        raise ValueError("Experiment source or configuration changed")
    if file_hash(run / "initial.safetensors") != identity["initial_sha256"]:
        raise ValueError("Initial BA weights changed")
    if any(file_hash(run / p) != h for p, h in identity["dataset_sha256"].items()):
        raise ValueError("Face crop dataset changed")
    return config, identity


def prepare(run, config, identity):
    backend = backend_module(config)
    model = backend.load_transformer(config)
    branch = install(model, identity["query_residual"], identity["noise_skip"], identity["timestep_scaling"])
    branch.load_state_dict(load_file(run / "initial.safetensors"))
    assert all("reference_branch." in n for n, p in model.named_parameters() if p.requires_grad)
    rows = read_manifest(config["data"]["train_manifest"], training=True)
    (run / "cases").mkdir()
    records = []
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    for split, seeds in (("fit", (142, 242)), ("probe", (10142,))):
        for i, row in enumerate(rows):
            tensors, _ = load_pair(config, row, "cuda")
            target = tensors["target_latent"]
            for seed in seeds:
                for j, value in enumerate(SIGMAS):
                    sigma = torch.tensor([value], dtype=target.dtype, device="cuda")
                    noise = torch.randn(target.shape, dtype=target.dtype, generator=torch.Generator().manual_seed(seed+i*100+j)).cuda()
                    noisy = (1-sigma)*target + sigma*noise
                    features = capture(model, branch, backend, tensors, noisy, sigma, config)
                    cached = {k: v.cpu().contiguous() for k, v in features.items()}
                    cached.update(flow_target=(noise-target).flatten(2).transpose(1, 2).cpu().contiguous(), noisy=noisy.cpu(), sigma=sigma.cpu())
                    if not records:
                        repeated = capture(model, branch, backend, tensors, noisy, sigma, config)
                        assert all(torch.equal(v, repeated[k].cpu()) for k, v in cached.items() if k in repeated)
                        assert not branch.predict(**repeated).count_nonzero()
                    path = run / "cases" / f"{split}_{i}_{seed}_{j}.safetensors"
                    save_file(cached, path)
                    records.append({"file": str(path.relative_to(run)), "sha256": file_hash(path), "split": split, "row": i, "sigma": value})
            print(f"Cached {split} face {i+1}/4", flush=True)
    write(run / "cache_manifest.json", {"records": records, "seconds": time.monotonic()-began,
                                        "repeat_capture_exact": True, "zero_initial_velocity": True, **memory()})


def cases(run):
    groups = {"fit": [], "probe": []}
    for row in json.loads((run / "cache_manifest.json").read_text())["records"]:
        path = run / row["file"]
        if file_hash(path) != row["sha256"]:
            raise ValueError("Cached input changed")
        groups[row["split"]].append({k: v.cuda() for k, v in load_file(path).items() if k not in ("noisy", "sigma")})
    return groups


def prediction(branch, case, reference_read=True):
    extra={'core_velocity':case['core_velocity']} if 'core_velocity' in case else {}
    return branch.predict(case["query"], case["key"], case["value"], reference_read=reference_read,
                          noise=case.get("noise"), timestep=case.get("timestep"), **extra)


@torch.no_grad()
def evaluate(branch, groups):
    scores = {}
    for split, group in groups.items():
        total = zero = no_read = 0.
        for case in group:
            total += float((prediction(branch, case)-case["flow_target"].float()).square().mean()) / len(group)
            zero += float(case["flow_target"].float().square().mean()) / len(group)
            ablated = prediction(branch, case, reference_read=False)
            no_read += float((ablated-case["flow_target"].float()).square().mean()) / len(group)
        scores.update({f"probe/{split}/mse": total, f"probe/{split}/zero_flow_mse": zero,
                       f"probe/{split}/reduction": 1-total/zero,
                       f"probe/{split}/{getattr(branch, 'ablation_metric', 'reference_read_off_mse')}": no_read})
    return scores


def train(run, config, identity, resume=0, until=2000):
    branch = FaceCropFlow(query_residual=identity["query_residual"], noise_skip=identity["noise_skip"], timestep_scaling=identity["timestep_scaling"]).cuda()
    path = run / f"checkpoint-{resume:06d}/branch.safetensors" if resume else run / "initial.safetensors"
    branch.load_state_dict(load_file(path))
    optimizer = torch.optim.AdamW(branch.parameters(), lr=.002, weight_decay=0., fused=True)
    if resume:
        manifest = json.loads((path.parent / "manifest.json").read_text())
        if manifest["identity"] != identity or manifest["sha256"] != file_hash(path):
            raise ValueError("Resume checkpoint provenance failed")
        optimizer.load_state_dict(torch.load(path.parent / "optimizer.pt", map_location="cpu", weights_only=True))
    groups = cases(run)
    exp = experiment(config, run)
    probes = json.loads((run / "probes.json").read_text()) if resume else [{"step": 0, **evaluate(branch, groups)}]
    if not resume:
        exp.log_metrics({k:v for k,v in probes[0].items() if k != "step"}, step=0)
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(resume+1, until+1):
            start = time.monotonic()
            selected = [groups["fit"][((step-1)*8+j) % len(groups["fit"])] for j in range(8)]
            batch = {k: torch.cat([x[k] for x in selected]) for k in selected[0]}
            for group in optimizer.param_groups:
                group["lr"] = .002 * min(1., step/100)
            optimizer.zero_grad(set_to_none=True)
            loss = (prediction(branch, batch)-batch["flow_target"].float()).square().mean()
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not p.grad.isfinite().all() for p in branch.parameters()):
                raise RuntimeError("Missing or nonfinite BA gradients")
            if step in (1, 2):
                check = {n: float(p.grad.norm()) for n, p in branch.named_parameters()}
                write(run / f"first_gradients_{step}.json", check)
                if not check["out.weight"] or (step == 2 and not all(check.values())):
                    raise RuntimeError("BA projection did not receive its expected first gradient")
            norm = torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.)
            optimizer.step()
            torch.cuda.synchronize()
            if step == 51 and (run / "expected_update_51.safetensors").exists():
                expected = load_file(run / "expected_update_51.safetensors")
                assert all(torch.equal(p.cpu(), expected[n]) for n, p in branch.state_dict().items())
                write(run / "resume_check.json", {"step": 51, "fresh_process_exact": True})
            # Keep every update locally; sampled live charts avoid API throttling.
            log_metrics(exp if step % 25 == 0 or step == 1 else None, run, {"train/loss": float(loss.detach()), "train/gradient_norm": float(norm),
                                   "train/seconds": time.monotonic()-start, **memory()}, step)
            if step % 250 == 0 or step == until:
                probes.append({"step": step, **evaluate(branch, groups)})
                write(run / "probes.json", probes)
                exp.log_metrics({k:v for k,v in probes[-1].items() if k != "step"}, step=step)
            if step in (50, 1000, 2000):
                destination = run / f"checkpoint-{step:06d}"
                destination.mkdir()
                save_file({n:p.cpu().contiguous() for n,p in branch.state_dict().items()}, destination / "branch.safetensors")
                torch.save(optimizer.state_dict(), destination / "optimizer.pt")
                write(destination / "manifest.json", {"step": step, "identity": identity, "sha256": file_hash(destination / "branch.safetensors")})
        write(run / "training_summary.json", {"steps": until, "all_gradients_finite": True, "parameters": sum(p.numel() for p in branch.parameters()), **memory()})
        if until == 50:
            selected = [groups["fit"][(50*8+j)%len(groups["fit"])] for j in range(8)]
            batch = {k:torch.cat([x[k] for x in selected]) for k in selected[0]}
            for group in optimizer.param_groups: group["lr"] = .002 * .51
            optimizer.zero_grad(set_to_none=True)
            (prediction(branch,batch)-batch["flow_target"].float()).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(branch.parameters(),1.)
            optimizer.step()
            save_file({n:p.cpu().contiguous() for n,p in branch.state_dict().items()},run/"expected_update_51.safetensors")
    finally:
        exp.end()


@torch.no_grad()
def infer(run, config, identity):
    backend = backend_module(config)
    from extensions_built_in.diffusion_models.flux2.src.sampling import get_schedule

    model = backend.load_transformer(config)
    branch = install(model, identity["query_residual"], identity["noise_skip"], identity["timestep_scaling"])
    rows = read_manifest(config["data"]["validation_manifest"])
    parity = []
    records = json.loads((run / "cache_manifest.json").read_text())["records"]
    for step in STEPS:
        checkpoint = run / f"checkpoint-{step:06d}" if step else None
        path = checkpoint / "branch.safetensors" if step else run / "initial.safetensors"
        if step:
            manifest = json.loads((checkpoint / "manifest.json").read_text())
            assert manifest["identity"] == identity and manifest["sha256"] == file_hash(path)
        branch.load_state_dict(load_file(path), strict=True)
        for record in (records[0], records[-1]):
            case = {k:v.cuda() for k,v in load_file(run / record["file"]).items()}
            row = read_manifest(config["data"]["train_manifest"], training=True)[record["row"]]
            tensors,_ = load_pair(config,row,"cuda")
            features = capture(model,branch,backend,tensors,case["noisy"],case["sigma"],config)
            fresh, cached = branch.predict(**features), prediction(branch,case)
            assert torch.equal(fresh,cached), "Fresh backbone and cached BA predictions differ"
            parity.append({"step":step,"case":record["file"],"exact":True})
        folder=run/f"validation-{step:06d}"
        folder.mkdir()
        samples=[]
        for row in rows:
            tensors,metadata=load_pair(config,row,"cuda")
            latent=torch.randn((1,128,16,16),dtype=torch.bfloat16,generator=torch.Generator().manual_seed(row["seed"])).cuda()
            initial=latent.clone()
            if step:
                times=get_schedule(20,256)
                for current,following in zip(times[:-1],times[1:]):
                    sigma=torch.tensor([current],dtype=latent.dtype,device="cuda")
                    features=capture(model,branch,backend,tensors,latent,sigma,config)
                    velocity=branch.predict(**features).transpose(1,2).reshape_as(latent).to(latent.dtype)
                    latent=latent+(following-current)*velocity
            else:
                assert not branch.out.weight.count_nonzero()
                assert torch.equal(latent,initial)
            if not torch.isfinite(latent).all(): raise RuntimeError("Nonfinite sampled latent")
            save_file({"latent":latent.cpu()},folder/f"{row['sample_id']}.safetensors")
            samples.append({"sample_id":row["sample_id"],"identity_id":row["identity_id"],"seed":row["seed"],"prompt":row["prompt"],
                            "image":row["sample_id"]+".png","native_velocity_used":False,"initial_noise_preserved":step==0})
            print(f"BA-only validation {step}: {row['sample_id']}",flush=True)
        write(folder/"validation.json",{"samples":samples,"checkpoint":str(checkpoint) if checkpoint else None,
                                        "panel_sha256":file_hash(config["data"]["validation_manifest"]),"native_velocity_used":False})
        (folder/"resolved_config.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    write(run/"cached_full_parity.json",parity)
    write(run/"inference_memory.json",memory())
    del model,branch,features,case,tensors
    gc.collect(); torch.cuda.empty_cache()
    vae=backend.load_vae(config)
    exp=experiment(config,run)
    for step in STEPS:
        folder=run/f"validation-{step:06d}"
        for row in rows:
            image=backend.decode(vae,load_file(folder/f"{row['sample_id']}.safetensors")["latent"],config)
            image.save(folder/(row["sample_id"]+".png"))
            exp.log_image(image,name=f"face_crop/{row['sample_id']}",step=step)
    exp.end()


def review(run,config):
    python=ROOT/"envs/metrics/bin/python"
    for step in STEPS:
        subprocess.run([str(python),str(ROOT/"scripts/evaluate_metrics.py"),"--validation",str(run/f"validation-{step:06d}"),
                        "--ownership-boxes",str(run/"ownership_boxes.json"),"--log-dir",str(run),"--global-step",str(step)],check=True)
    sheet=Image.new("RGB",(3*260,4*280+28),"white")
    draw=ImageDraw.Draw(sheet)
    for col,step in enumerate(STEPS):
        draw.text((col*260+5,8),f"Step {step}: BA-only face flow",fill="black")
        for i in range(4):
            sheet.paste(Image.open(run/f"validation-{step:06d}/crop_val_{i}.png"),(col*260,i*280+28))
    sheet.save(run/"comparison.png")
    probes=json.loads((run/"probes.json").read_text())
    summary={str(step):json.loads((run/f"validation-{step:06d}/quality_summary.json").read_text())["metrics"] for step in STEPS}
    write(run/"metric_summary.json",summary)
    lines=["# BA-only face-crop reconstruction", "", "The native FLUX velocity is disabled. Zero BA output leaves Gaussian noise unchanged. Only a random-initialized Q/K/V attention network with a zero output projection is trained.","",
           "Four 256px crops of identity 51; separate fixed fit/probe noise; four generation seeds. This is a reconstruction diagnostic using frozen FLUX features, not the full-scene residual BA architecture.","",
           "| Step | Fit flow MSE | Probe flow MSE |", "| ---: | ---: | ---: |"]
    identity=json.loads((run/"identity.json").read_text())
    lines.insert(4, f"Query residual inside BA: {identity['query_residual']}. This never adds the native FLUX velocity.")
    lines.insert(5, f"Learned noisy-latent projection inside BA: {identity['noise_skip']}; division by timestep: {identity['timestep_scaling']}. This projection also starts at zero.")
    lines += [f"| {p['step']} | {p['probe/fit/mse']:.6f} | {p['probe/probe/mse']:.6f} |" for p in probes]
    lines += ["", "| Step | ID similarity | Face missing | CLIP |", "| ---: | ---: | ---: | ---: |"]
    lines += [f"| {s} | {summary[str(s)]['id_sim']:.4f} | {summary[str(s)]['id_sim_no_face']:.0%} | {summary[str(s)]['text_sim']:.3f} |" for s in STEPS]
    lines += ["", "Scoring ownership is the whole predefined crop canvas, used only after generation. The crop contains the face; no detector box or target mask enters inference.","", "![Noise-to-face comparison](comparison.png)",""]
    (run/"report.md").write_text("\n".join(lines))
    exp=experiment(config,run)
    for path in (run/"comparison.png",run/"report.md",run/"metric_summary.json",run/"probes.json",run/"resume_check.json",run/"cached_full_parity.json",run/"training_summary.json",run/"metrics.jsonl"):
        exp.log_asset(str(path))
    exp.log_image(str(run/"comparison.png"),name="face_crop/comparison",step=2000)
    exp.end()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("run","prepare","train","infer","review"))
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--resume",type=int,default=0)
    parser.add_argument("--until",type=int,default=2000)
    parser.add_argument("--query-residual",action="store_true",help="Use a query skip within BA; native velocity stays disabled")
    parser.add_argument("--noise-skip",action="store_true",help="Learn the noisy-latent projection inside BA, with flow/x0 timestep scaling")
    parser.add_argument("--unscaled-noise",action="store_true",help="Predict velocity directly, without division by timestep")
    args=parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    run=args.run_dir.resolve()
    if args.command=="run": initialize(run,args.query_residual,args.noise_skip,not args.unscaled_noise)
    config,identity=verify(run)
    if args.command=="run":
        for split in ("train","validation"):
            subprocess.run([sys.executable,"-m","ba_dit.cli","precompute","--config",str(run/"resolved_config.yaml"),"--split",split,"--limit","4"],check=True)
        for command,extra in (("prepare",[]),("train",["--until","50"]),("train",["--resume","50"]),("infer",[]),("review",[])):
            subprocess.run([sys.executable,"-m","scripts.face_crop_flow",command,"--run-dir",str(run),*extra],check=True)
    elif args.command=="prepare": prepare(run,config,identity)
    elif args.command=="train": train(run,config,identity,args.resume,args.until)
    elif args.command=="infer": infer(run,config,identity)
    else: review(run,config)


if __name__=="__main__": main()
