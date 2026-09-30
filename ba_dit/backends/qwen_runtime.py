"""Native Qwen multimodal slots, frozen components, flow loss, and sampling."""

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline, QwenImage21Transformer2DModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from ba_dit.data.geometry import reference_geometry, target_geometry


def load_transformer(config, device="cuda"):
    return QwenImage21Transformer2DModel.from_pretrained(config["model"]["weights"], subfolder="transformer",
            torch_dtype=torch.bfloat16, local_files_only=True).requires_grad_(False).eval().to(device)


def load_encoder(config):
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    path = config["model"]["weights"]
    device = "cuda" if torch.cuda.get_device_properties(0).total_memory >= 24 * 2**30 else "cpu"
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(config["model"]["encoder"], torch_dtype=torch.bfloat16,
            local_files_only=True).eval().requires_grad_(False).to(device)
    processor = Qwen3VLProcessor.from_pretrained(path, subfolder="processor", local_files_only=True)
    return QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=encoder, processor=processor, transformer=None)


@torch.no_grad()
def encode_text(pipe, config, row):
    with Image.open(row["reference"]) as source:
        image, _, geometry = reference_geometry(source.convert("RGBA"), row["reference_box"], "qwen", config["data"]["reference_size"])
    embeddings, mask, slots = pipe.encode_prompt(row["prompt"], image=[image], device=pipe.text_encoder.device)
    expected = geometry["token_hw"][0] * geometry["token_hw"][1]
    if int(slots.sum()) * 4 != expected:
        raise ValueError(f"Qwen processor slots ({int(slots.sum())}x4) differ from reference grid ({expected})")
    tensors = {"prompt_embeds": embeddings, "image_slots": slots}
    if mask is not None:
        tensors["prompt_mask"] = mask
    return tensors, {"reference": geometry, "text_tokens": embeddings.shape[1], "prompt_mask_none": mask is None}


def load_vae(config, device="cuda"):
    return AutoencoderKLQwenImage21.from_pretrained(config["model"]["vae"], torch_dtype=torch.bfloat16,
            local_files_only=True).requires_grad_(False).eval().to(device)


def normalize(vae, latent, inverse=False):
    mean = torch.tensor(vae.config.latents_mean, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
    return latent * std + mean if inverse else (latent - mean) / std


def pack(latent):
    return QwenImage21Pipeline._pack_latents(latent, latent.shape[0], latent.shape[1], *latent.shape[-2:])


def normalize_training(vae, latent):
    # The paired upstream trainer normalizes in FP32, then casts to weight_dtype.
    mean = torch.tensor(vae.config.latents_mean, device=latent.device).view(1, -1, 1, 1, 1)
    inverse_std = 1 / torch.tensor(vae.config.latents_std, device=latent.device).view(1, -1, 1, 1, 1)
    return ((latent - mean) * inverse_std).to(torch.bfloat16)


@torch.no_grad()
def encode_images(vae, config, row):
    height, width = config["data"]["target_size"]
    processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
    with Image.open(row["reference"]) as source:
        image, token_mask, geometry = reference_geometry(source.convert("RGBA"), row["reference_box"], "qwen", config["data"]["reference_size"])
    pixels = processor.preprocess(image, height=image.height, width=image.width).unsqueeze(2).to(vae.device, vae.dtype)
    normalization = normalize_training if row.get("target") else normalize
    reference = normalization(vae, vae.encode(pixels).latent_dist.mode())
    tensors = {"reference_tokens": pack(reference), "reference_mask": torch.from_numpy(token_mask)}
    if tensors["reference_tokens"].shape[1] != token_mask.size:
        raise ValueError("Qwen reference mask differs from the native latent grid")
    metadata = {"reference": geometry, "target_hw": [height // 16, width // 16], "reference_tokens": token_mask.size}
    if row.get("target"):
        with Image.open(row["target"]) as source:
            target, target_transform = target_geometry(source.convert("RGBA"), [height, width], row.get("target_box"))
        pixels = processor.preprocess(target, height=height, width=width).unsqueeze(2).to(vae.device, vae.dtype)
        seed = (int(row["target_hash"][:16], 16) ^ config["training"]["seed"]) % 2**63
        generator = torch.Generator(device=vae.device).manual_seed(seed)
        tensors["target_latent"] = normalize_training(vae, vae.encode(pixels).latent_dist.sample(generator=generator))
        metadata["target"] = target_transform
    return tensors, metadata


def predict(model, tensors, noisy, sigma, config, branch=True, negative=False):
    packed = pack(noisy)
    prefix = "negative_" if negative else ""
    slots = tensors[prefix + "image_slots"]
    ref_hw = list(tensors["reference_mask"].shape)
    if int(slots.sum()) * 4 != ref_hw[0] * ref_hw[1]:
        raise ValueError("Cached Qwen encoder and reference latent slots disagree")
    img_mask = torch.cat((slots, slots.new_ones((1, packed.shape[1] // 4))), dim=1)
    result = model(hidden_states=torch.cat((tensors["reference_tokens"], packed), dim=1),
        encoder_hidden_states=tensors[prefix + "prompt_embeds"], encoder_hidden_states_mask=tensors.get(prefix + "prompt_mask"),
        timestep=sigma.reshape(1).to(noisy.dtype), img_shapes=[[(1, *ref_hw), (1, *noisy.shape[-2:])]], img_mask=img_mask,
        attention_kwargs={"branch_reference_mask": tensors["reference_mask"], "branch_max_reference_keys": config["branch"]["max_reference_keys"]} if branch else None,
        kv_cache=None, kv_cache_mode=None, return_dict=False)[0][:, -packed.shape[1]:]
    return result.transpose(1, 2).reshape_as(noisy)


def scheduler(config):
    return FlowMatchEulerDiscreteScheduler.from_pretrained(config["model"]["weights"], subfolder="scheduler", local_files_only=True)


def training_loss(model, tensors, config, branch=True):
    target = tensors["target_latent"]
    noise = torch.randn_like(target)
    schedule = scheduler(config)
    u = compute_density_for_timestep_sampling(weighting_scheme="none", batch_size=1, logit_mean=0, logit_std=1, mode_scale=1.29)
    index = (u * schedule.config.num_train_timesteps).long()
    sigma = schedule.sigmas[index].to(target.device, target.dtype)
    timestep = schedule.timesteps[index].to(target.device)
    noisy = (1 - sigma) * target + sigma * noise
    prediction = predict(model, tensors, noisy, timestep / 1000, config, branch)
    weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigma)
    return (weighting.float() * (prediction.float() - (noise - target).float()).square()).mean()


@torch.no_grad()
def sample(model, tensors, config, seed, branch=True, callback=None):
    height, width = config["data"]["target_size"]
    latent = torch.randn((1, 64, 1, height // 16, width // 16), generator=torch.Generator().manual_seed(seed), dtype=torch.bfloat16).to(model.device)
    schedule = scheduler(config)
    count = config["validation"]["steps"]
    mu = calculate_shift(height * width // 256, schedule.config.get("base_image_seq_len", 256), schedule.config.get("max_image_seq_len", 4096),
                         schedule.config.get("base_shift", 0.5), schedule.config.get("max_shift", 1.15))
    times, _ = retrieve_timesteps(schedule, count, latent.device, sigmas=np.linspace(1.0, 1 / count, count), mu=mu)
    schedule.set_begin_index(0)
    for index, timestep in enumerate(times):
        sigma = timestep.expand(1).to(latent.dtype) / 1000
        velocity = predict(model, tensors, latent, sigma, config, branch)
        if config["validation"]["guidance"] > 1:
            negative = predict(model, tensors, latent, sigma, config, branch, negative=True)
            velocity = negative + config["validation"]["guidance"] * (velocity - negative)
        latent = schedule.step(velocity, timestep, latent, return_dict=False)[0]
        if callback:
            callback(index, latent)
    return latent


@torch.no_grad()
def decode(vae, latent, config):
    pixels = vae.decode(normalize(vae, latent.to(vae.device, vae.dtype), inverse=True), return_dict=False)[0][:, :, 0]
    return VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64).postprocess(pixels, output_type="pil")[0]
