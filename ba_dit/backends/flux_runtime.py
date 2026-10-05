"""Native FLUX components, packed conditioning, flow loss, and Euler sampling."""

from pathlib import Path

import torch
from PIL import Image

from ba_dit.data.geometry import reference_geometry, target_geometry
from extensions_built_in.diffusion_models.flux2.src.model import Flux2, Klein4BParams, Klein9BParams
from extensions_built_in.diffusion_models.flux2.src.pipeline import Flux2Pipeline
from extensions_built_in.diffusion_models.flux2.src.sampling import batched_prc_img, batched_prc_txt, default_images_prep, encode_image_refs, get_schedule
from toolkit.models.v2.vae.flux2_kl import AutoEncoder
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler


def scheduler():
    return CustomFlowMatchEulerDiscreteScheduler(base_image_seq_len=256, base_shift=0.5, max_image_seq_len=4096,
                                                 max_shift=1.15, num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True)


def model_dtype(config, conditioning=False):
    return getattr(torch, config['model'].get('conditioning_dtype' if conditioning else 'dtype', 'bfloat16'))


def load_transformer(config, device="cuda"):
    from safetensors.torch import load_file

    is4 = config["model"]["arch"] == "flux2_klein_4b"
    filename = f"flux-2-klein-base-{'4b' if is4 else '9b'}.safetensors"
    weights = load_file(str(Path(config["model"]["weights"]) / filename), device="cpu")
    model = Flux2.load_from_state_dict(weights, dtype=model_dtype(config), config=Klein4BParams() if is4 else Klein9BParams())
    del weights
    from ba_dit.precision import configure_residuals
    return configure_residuals(model.requires_grad_(False).to(device).eval(), config)


def load_encoder(config):
    from transformers import Qwen2Tokenizer, Qwen3ForCausalLM

    path = config["model"]["encoder"]
    device = "cpu" if config["model"]["arch"].endswith("9b") and torch.cuda.get_device_properties(0).total_memory < 24 * 2**30 else "cuda"
    device = config['data'].get('encoder_device', device)
    encoder = Qwen3ForCausalLM.from_pretrained(path, dtype=model_dtype(config, conditioning=True), local_files_only=True).eval().requires_grad_(False).to(device)
    tokenizer = Qwen2Tokenizer.from_pretrained(path, local_files_only=True)
    return Flux2Pipeline(scheduler=None, vae=None, text_encoder=encoder, tokenizer=tokenizer, transformer=None,
                         text_encoder_type="qwen", is_guidance_distilled=False)


@torch.no_grad()
def encode_text(pipe, config, row):
    embeddings, _ = pipe.encode_prompt(row["prompt"], device=pipe.text_encoder.device)
    embeddings, ids = batched_prc_txt(embeddings)
    return {"prompt_embeds": embeddings, "text_ids": ids}, {"text_tokens": embeddings.shape[1]}


def load_vae(config, device="cuda"):
    return AutoEncoder.load_model(config["model"]["vae"], dtype=model_dtype(config, conditioning=True)).requires_grad_(False).eval().to(device)


@torch.no_grad()
def encode_images(vae, config, row):
    height, width = config["data"]["target_size"]
    with Image.open(row["reference"]) as source:
        original = source.convert("RGB")
    _, token_mask, ref_geometry = reference_geometry(original, row["reference_box"], "flux", config["data"]["reference_size"])
    reference, reference_ids = encode_image_refs(vae, [original], limit_pixels=config["data"]["reference_size"]**2)
    if reference.shape[1] != token_mask.size:
        raise ValueError("FLUX reference face mask differs from native packed grid")
    _, target_ids = batched_prc_img(torch.zeros(1, 128, height // 16, width // 16))
    tensors = {"reference_tokens": reference, "reference_ids": reference_ids,
               "reference_mask": torch.from_numpy(token_mask), "target_ids": target_ids}
    metadata = {"reference": ref_geometry, "target_hw": [height // 16, width // 16], "reference_tokens": reference.shape[1]}
    if row.get("target"):
        with Image.open(row["target"]) as source:
            target, geometry = target_geometry(source.convert("RGB"), [height, width], row.get("target_box"))
        tensors["target_latent"] = vae.encode(default_images_prep(target).unsqueeze(0).to(vae.device, vae.dtype))
        metadata["target"] = geometry
    return tensors, metadata


def predict(model, tensors, noisy, sigma, config, branch=True, negative=False, reference_source=None):
    from ba_dit.precision import compute_context
    with compute_context(config):
        return _predict(model, tensors, noisy, sigma, config, branch, negative, reference_source)


def _predict(model, tensors, noisy, sigma, config, branch=True, negative=False, reference_source=None):
    if reference_source is not None and (not branch or config['branch'].get('reference_bank') != 'isolated_image'):
        raise ValueError('A branch-only reference swap requires an active isolated reference bank')
    if 'dtype' in config['model']:
        noisy = noisy.to(model_dtype(config))
        tensors = {k: v.to(model_dtype(config)) if k in {'reference_tokens', 'prompt_embeds', 'negative_prompt_embeds'} else v
                   for k, v in tensors.items()}
    if branch and config['branch'].get('kind') == 'flux2_face' and tensors['target_face_mask'].any():
        from ba_dit.nn.flux2_face import predict as face_predict
        return face_predict(model, tensors, noisy, sigma.to(noisy.dtype), negative)
    packed, _ = batched_prc_img(noisy)
    prefix = "negative_" if negative else ""
    extra = {}
    if branch and config['branch'].get('reference_bank') == 'isolated_image':
        from ba_dit.nn.isolated_reference import reference_bank
        if tensors.get('reference_masks') is not None:
            raise ValueError('Isolated-reference training currently uses microbatch 1')
        source = tensors if reference_source is None else reference_source
        extra['branch_reference_bank'] = reference_bank(model, source['reference_tokens'].to(noisy.dtype),
            source['reference_ids'], sigma.to(noisy.dtype), source['reference_mask'],
            config['branch']['max_reference_keys'])
    prediction = model(
        x=torch.cat((packed, tensors["reference_tokens"]), dim=1),
        x_ids=torch.cat((tensors["target_ids"], tensors["reference_ids"]), dim=1),
        timesteps=sigma.to(noisy.dtype).reshape(-1).expand(noisy.shape[0]), ctx=tensors[prefix + "prompt_embeds"],
        ctx_ids=tensors[prefix + "text_ids"], guidance=None,
        branch_reference_mask=tensors["reference_mask"] if branch else None,
        branch_target_tokens=packed.shape[1] if branch else None,
        branch_max_reference_keys=config["branch"]["max_reference_keys"],
        branch_target_mask=tensors.get('target_face_mask') if branch else None,
        **extra,
    )[:, :packed.shape[1]]
    return prediction.transpose(1, 2).reshape_as(noisy)


def training_loss(model, tensors, config, branch=True, identity_objective=None, row=None, step=0, force_identity=False):
    target = tensors["target_latent"]
    if 'dtype' in config['model']:
        target = target.to(model_dtype(config))
    noise = torch.randn_like(target)
    if config['branch'].get('kind') == 'flux2_face':
        tensors = {**tensors, 'flux2_context_noise':noise}
    noise_schedule = scheduler()
    # Pinned Toolkit defaults: sigmoid timesteps, balanced index sampling, MSE velocity target.
    times = noise_schedule.set_train_timesteps(1000, device=target.device, timestep_type="sigmoid", latents=target, patch_size=1)
    index = torch.randint(0, 999, (1,), device=target.device)
    timestep = times[index]
    noisy = noise_schedule.add_noise(target, noise, timestep).to(target.dtype)
    # Toolkit casts to the backbone dtype before dividing; BF16 rounding order matters.
    prediction = predict(model, tensors, noisy, timestep.to(target.dtype) / 1000, config, branch)
    error = (prediction.float() - (noise - target).float()).square()
    if config['branch'].get('kind') in {'masked_face_qkvo', 'flux2_face'}:
        mask = tensors['target_face_mask'].reshape(target.shape[0], 1, *target.shape[-2:])
        if not mask.sum() > 0:
            raise ValueError('Empty face supervision')
        flow = (error * mask).sum() / (mask.sum() * target.shape[1])
        if identity_objective is not None:
            # Reconstruct using the actual FP32 noising coefficient. Native
            # model conditioning retains its upstream BF16 rounding above.
            sigma = timestep.float() / 1000
            auxiliary = identity_objective.loss(prediction, noisy, sigma, row, step, force=force_identity)
            identity_objective.last_metrics['train/flow_loss'] = float(flow.detach())
            return flow + auxiliary
        return flow
    return error.mean()


@torch.no_grad()
def sample(model, tensors, config, seed, branch=True, callback=None):
    return sample_batch(model, tensors, config, [seed], branch, callback)


@torch.no_grad()
def sample_batch(model, tensors, config, seeds, branch=True, callback=None):
    height, width = config["data"]["target_size"]
    latent = torch.cat([torch.randn((1, 128, height // 16, width // 16), generator=torch.Generator().manual_seed(seed),
                                    dtype=model_dtype(config)) for seed in seeds]).to(model.device)
    times = get_schedule(config["validation"]["steps"], latent.shape[-2] * latent.shape[-1])
    guidance = config["validation"]["guidance"]
    for index, (current, following) in enumerate(zip(times[:-1], times[1:])):
        sigma = torch.tensor([current], device=latent.device, dtype=latent.dtype)
        velocity = predict(model, tensors, latent, sigma, config, branch)
        if guidance > 1:
            negative = predict(model, tensors, latent, sigma, config, branch, negative=True)
            velocity = negative + guidance * (velocity - negative)
        latent = latent + (following - current) * velocity
        if callback:
            callback(index, latent)
    return latent


@torch.no_grad()
def decode(vae, latent, config):
    from diffusers.image_processor import VaeImageProcessor

    pixels = vae.decode(latent.to(vae.device, vae.dtype)).float()
    return VaeImageProcessor(vae_scale_factor=16).postprocess(pixels, output_type="pil")[0]
