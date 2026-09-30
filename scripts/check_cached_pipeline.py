#!/usr/bin/env python3
"""Compare cached denoising with the original pipeline and live VAE reference path.

Text embeddings come from the native encode_prompt method in the encoder stage.
Use the resolved config and output directory of a completed short validation.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from ba_dit.config import load_config
from ba_dit.data.cache import load_pair
from ba_dit.data.manifest import read_manifest
from ba_dit.runtime import backend_module


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.validation / "resolved_config.yaml")
    row = read_manifest(config["data"]["validation_manifest"], limit=1)[0]
    backend = backend_module(config)
    tensors, _ = load_pair(config, row, "cuda", negative=config["validation"]["guidance"] > 1)
    model, vae = backend.load_transformer(config), backend.load_vae(config)
    height, width = config["data"]["target_size"]
    reference = Image.open(row["reference"]).convert("RGB")
    options = dict(height=height, width=width, num_inference_steps=config["validation"]["steps"],
                   generator=torch.Generator().manual_seed(row["seed"]), output_type="latent")
    conditioning = {}
    if config["model"]["backend"] == "flux":
        pipe = backend.Flux2Pipeline(scheduler=backend.scheduler(), vae=vae, transformer=model,
                                    tokenizer=None, text_encoder=None, text_encoder_type="qwen")
        live_ref, live_ids = backend.encode_image_refs(vae, [reference], limit_pixels=config["data"]["reference_size"]**2)
        conditioning["reference_ids_exact"] = torch.equal(live_ids, tensors["reference_ids"])
        conditioning["reference_max_abs"] = float((live_ref - tensors["reference_tokens"]).abs().max())
        native = pipe(prompt_embeds=tensors["prompt_embeds"], negative_prompt_embeds=tensors["negative_prompt_embeds"],
                      control_img_list=[reference], reference_limit_pixels=config["data"]["reference_size"]**2,
                      guidance_scale=config["validation"]["guidance"], **options).images
    else:
        from transformers import Qwen3VLProcessor
        processor = Qwen3VLProcessor.from_pretrained(config["model"]["weights"], subfolder="processor", local_files_only=True)
        pipe = backend.QwenImage21Pipeline(scheduler=backend.scheduler(config), vae=vae, transformer=model,
                                          processor=processor, text_encoder=None)
        # The pinned public pipeline cannot accept condition-image slots with cached
        # embeddings. Reuse the native encoder's exact tuple only for this comparison.
        def encoded(**kwargs):
            return tensors["prompt_embeds"], tensors.get("prompt_mask"), tensors["image_slots"]
        pipe.encode_prompt = encoded
        original_prepare = pipe.prepare_latents

        def prepare(*args, **kwargs):
            latent, live_ref = original_prepare(*args, **kwargs)
            conditioning["reference_max_abs"] = float((live_ref - tensors["reference_tokens"]).abs().max())
            return latent, live_ref
        pipe.prepare_latents = prepare
        packed = pipe(prompt=row["prompt"], image=reference, output_resolution=config["data"]["reference_size"],
                      true_cfg_scale=config["validation"]["guidance"], use_kv_cache=False, **options).images
        native = pipe._unpack_latents(packed, height, width, pipe.vae_scale_factor)
    cached = load_file(args.validation / f"{row['sample_id']}.safetensors")["latent"].to(native.device)
    report = {**conditioning, "latent_exact": torch.equal(native, cached),
              "latent_max_abs": float((native.float() - cached.float()).abs().max()),
              "vae_device": str(vae.device), "transformer_device": str(model.device)}
    backend.decode(vae, native, config).save(args.validation / f"{row['sample_id']}.live.png")
    (args.validation / "live_cache_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["latent_exact"] or report["reference_max_abs"]:
        raise RuntimeError("Cached conditioning/sampling differs from the live native pipeline")


if __name__ == "__main__":
    main()
