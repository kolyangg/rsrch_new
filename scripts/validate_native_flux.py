#!/usr/bin/env python3
"""Serial FLUX.2-klein Base native/initialized-branch validation with CPU offload."""

import argparse
import json
import sys
import time
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLKIT = ROOT / "sources/ai-toolkit-flux"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLKIT))
# Import the pinned FLUX package directly. Toolkit's broad diffusion_models
# initializer imports unrelated audio/UI backends during an image-only run.
for package, relative in (
    ("extensions_built_in", "extensions_built_in"),
    ("extensions_built_in.diffusion_models", "extensions_built_in/diffusion_models"),
    ("extensions_built_in.diffusion_models.flux2", "extensions_built_in/diffusion_models/flux2"),
):
    module = types.ModuleType(package)
    module.__path__ = [str(TOOLKIT / relative)]
    sys.modules[package] = module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("4b", "9b"), default="4b")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--reference-size", type=int, choices=(512, 768), default=512)
    parser.add_argument("--branch-initialized", action="store_true", help="Exercise zero-initialized reference branch")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/native_flux4b_local")
    args = parser.parse_args()
    if min(args.limit, args.width, args.height, args.steps) <= 0:
        parser.error("limit, dimensions and steps must be positive")
    if args.width % 16 or args.height % 16:
        parser.error("FLUX dimensions must be divisible by 16")
    from PIL import Image
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen2Tokenizer, Qwen3ForCausalLM
    from extensions_built_in.diffusion_models.flux2.src.model import Flux2, Klein4BParams, Klein9BParams
    from extensions_built_in.diffusion_models.flux2.src.pipeline import Flux2Pipeline
    from toolkit.models.v2.vae.flux2_kl import AutoEncoder
    from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    weights = ROOT / "weights"
    generator_path = weights / f"flux{args.arch}/flux-2-klein-base-{args.arch}.safetensors"
    encoder_path = weights / f"flux{args.arch}_text"
    vae_path = weights / "flux_vae/ae.safetensors"
    for path in (generator_path, encoder_path, vae_path):
        if not path.exists():
            raise FileNotFoundError(path)
    print(f"Loading pinned native FLUX {args.arch.upper()} Base components on CPU", flush=True)
    started = time.monotonic()
    state = load_file(str(generator_path), device="cpu")
    transformer = Flux2.load_from_state_dict(
        state, dtype=torch.bfloat16, config=Klein4BParams() if args.arch == "4b" else Klein9BParams(),
    )
    if args.branch_initialized:
        from ba_dit.backends.flux2_native import install_reference_branch

        install_reference_branch(transformer, f"flux2_klein_{args.arch}")
    del state
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        str(encoder_path), dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(str(encoder_path), local_files_only=True)
    vae = AutoEncoder.load_model(str(vae_path), dtype=torch.bfloat16)
    scheduler = CustomFlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256, base_shift=0.5, max_image_seq_len=4096,
        max_shift=1.15, num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True,
    )
    pipeline = Flux2Pipeline(
        scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        transformer=transformer, text_encoder_type="qwen", is_guidance_distilled=False,
    )
    pipeline.enable_model_cpu_offload(gpu_id=0)
    panel = ROOT / "data/validation"
    rows = [json.loads(line) for line in (panel / "manual_val_96.jsonl").read_text().splitlines()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"Loaded components in {time.monotonic()-started:.1f}s; {len(rows)} panel items available", flush=True)
    for row in rows[:args.limit]:
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with Image.open(panel / row["reference_image"]) as source:
            reference = source.convert("RGB")
        seed_generator = torch.Generator(device="cpu").manual_seed(row["seed"])
        branch_mask = None
        if args.branch_initialized:
            mask_record = next(
                item for line in (panel / "reference_masks.jsonl").read_text().splitlines()
                if (item := json.loads(line))["backend"] == "flux2"
                and item["reference_size"] == args.reference_size and item["identity_id"] == row["identity_id"]
            )
            import numpy as np

            with Image.open(panel / mask_record["token_mask"]) as mask_image:
                branch_mask = torch.from_numpy(np.array(mask_image) > 0)
        with torch.inference_mode():
            output = pipeline(
                prompt=row["prompt"], negative_prompt="", height=args.height, width=args.width,
                num_inference_steps=args.steps, guidance_scale=4.0, generator=seed_generator,
                control_img_list=[reference], branch_reference_mask=branch_mask,
                reference_limit_pixels=args.reference_size**2,
            ).images[0]
        path = args.output_dir / f'{row["sample_id"]}.png'
        output.save(path)
        record = {
            "sample_id": row["sample_id"], "identity_id": row["identity_id"],
            "image": path.name, "seconds": round(time.monotonic()-started, 2),
            "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated()/2**30, 3),
            "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved()/2**30, 3),
        }
        results.append(record)
        print(json.dumps(record), flush=True)
    (args.output_dir / "validation.json").write_text(json.dumps({
        "backend": f"toolkit_flux{args.arch}_branch_initialized" if args.branch_initialized else f"toolkit_native_flux{args.arch}_base", "steps": args.steps,
        "width": args.width, "height": args.height, "reference_size": args.reference_size,
        "samples": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
