#!/usr/bin/env python3
"""Serial Qwen-Image-2.1 native/initialized-branch reference validation."""

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--reference-size", type=int, choices=(512, 768), default=512)
    parser.add_argument("--branch-initialized", action="store_true", help="Exercise zero-initialized reference branch")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/native_qwen21_local_smoke")
    args = parser.parse_args()
    if min(args.steps, args.limit, args.target_size) <= 0 or args.target_size % 32:
        parser.error("Steps/limit must be positive and target size divisible by 32")
    from PIL import Image
    import torch
    from diffusers import QwenImage21Pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    gpu_gib = torch.cuda.get_device_properties(0).total_memory // 2**30
    gpu_budget_gib = min(64, max(14, int(gpu_gib * 0.67)))
    panel = ROOT / "data/validation"
    samples = [json.loads(line) for line in (panel / "manual_val_96.jsonl").read_text().splitlines()]
    model_dir = ROOT / "weights/qwen21"
    if not (model_dir / "model_index.json").is_file():
        raise FileNotFoundError(model_dir)
    (ROOT / "scratch/qwen21_offload").mkdir(parents=True, exist_ok=True)
    print("Loading Qwen-Image-2.1 in BF16 with component-level CPU/GPU placement", flush=True)
    started = time.monotonic()
    pipe = QwenImage21Pipeline.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, local_files_only=True,
        device_map="balanced", max_memory={0: f"{gpu_budget_gib}GiB", "cpu": "64GiB"},
        offload_folder=str(ROOT / "scratch/qwen21_offload"),
    )
    if args.branch_initialized:
        from ba_dit.backends.qwen21 import install_reference_branch

        install_reference_branch(pipe.transformer)
    print(f"Loaded in {time.monotonic()-started:.1f}s; placement {pipe.hf_device_map}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks = {}
    if args.branch_initialized:
        masks = {
            (item["identity_id"], item["reference_size"]): item
            for line in (panel / "reference_masks.jsonl").read_text().splitlines()
            if (item := json.loads(line))["backend"] == "qwen21"
        }
    results = []
    for sample in samples[:args.limit]:
        with Image.open(panel / sample["reference_image"]) as source:
            reference = source.convert("RGB")
        attention_kwargs = None
        if args.branch_initialized:
            import numpy as np

            mask_record = masks[(sample["identity_id"], args.reference_size)]
            with Image.open(panel / mask_record["token_mask"]) as mask_image:
                attention_kwargs = {"branch_reference_mask": torch.from_numpy(np.array(mask_image) > 0)}
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode():
            image = pipe(
                prompt=sample["prompt"], image=reference, height=args.target_size, width=args.target_size,
                output_resolution=args.reference_size, num_inference_steps=args.steps, true_cfg_scale=1.0,
                use_kv_cache=False, attention_kwargs=attention_kwargs,
                generator=torch.Generator(device="cpu").manual_seed(sample["seed"]),
            ).images[0]
        image_name = f'{sample["sample_id"]}.png'
        image.save(args.output_dir / image_name)
        record = {
            "sample_id": sample["sample_id"], "identity_id": sample["identity_id"], "image": image_name,
            "seconds": round(time.monotonic()-started, 2),
            "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved()/2**30, 3),
        }
        results.append(record)
        print(json.dumps(record), flush=True)
    result = {
        "backend": "qwen_image_2_1_branch_initialized" if args.branch_initialized else "native_qwen_image_2_1",
        "steps": args.steps, "width": args.target_size, "height": args.target_size,
        "reference_size": args.reference_size, "samples": results,
        "device_map": str(pipe.hf_device_map),
    }
    (args.output_dir / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
