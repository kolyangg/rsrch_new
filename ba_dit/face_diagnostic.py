"""Exact cached final-block training for a small FLUX one-ID wiring experiment.

Only the BA output LoRA and its scalar face router train. Native Q/K/V, the
last-block output projection, final layer and the complete prefix stay frozen.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from ba_dit.config import ROOT, adapter_identity, load_config
from ba_dit.data.geometry import face_mask, target_geometry
from ba_dit.nn.face_reference_read import FaceReferenceRead


def settings(path):
    import yaml

    spec = yaml.safe_load(Path(path).read_text())
    config = load_config(ROOT / spec["base_config"])
    if config["model"]["arch"] != "flux2_klein_4b":
        raise ValueError("This explicitly named diagnostic supports FLUX Base 4B only")
    config["name"] = spec["name"]
    config["data"]["target_size"] = [spec["resolution"]] * 2
    config["data"]["train_limit"] = spec["train_images"]
    config["validation"].update(steps=spec["inference_steps"], limit=spec["validation_images"])
    config["training"].update(steps=spec["steps"], lr=spec["lr"], grad_accum=1, gradient_checkpointing=False,
                              seed=spec["seed"], warmup=0, checkpoint_every=spec["probe_every"],
                              validation_every=spec["steps"])
    config["hardware"]["min_vram_gb"] = 12
    config["branch"].update(rank=spec["rank"], alpha=spec["rank"], gamma=spec["gamma"])
    return spec, config


def identity(config, spec):
    paths = ["ba_dit/nn/face_reference_read.py", "ba_dit/face_diagnostic.py", "scripts/face_diagnostic.py"]
    return {"variant": "final_single_face_read_v2_shared_cfg", "base": adapter_identity(config),
            "diagnostic": spec, "code_sha256": hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in paths)).hexdigest()}


def install(model, spec):
    model.requires_grad_(False)
    if any(hasattr(block, "reference_branch") for block in (*model.double_blocks, *model.single_blocks)):
        raise ValueError("Face diagnostic requires a fresh backbone, without previous adapters")
    branch = FaceReferenceRead(model.hidden_size, model.num_heads, spec["rank"], spec["gamma"], spec["gate_prior"])
    model.single_blocks[-1].reference_branch = branch.to(model.device)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not trainable or any(".reference_branch." not in n for n in trainable):
        raise RuntimeError("Only the face branch may be trainable")
    return branch


def target_face_tokens(row, config):
    """Training supervision only, using the exact cached target resize/crop."""
    with Image.open(row["target"]) as source:
        source = source.convert("RGB")
        _, geometry = target_geometry(source, config["data"]["target_size"], row["target_box"])
        mask = face_mask(source, row["target_box"])
        mask = mask.resize(tuple(geometry["resize_wh"]), Image.Resampling.NEAREST).crop(geometry["crop_xyxy"])
    pixels = np.asarray(mask) > 0
    h, w = pixels.shape
    # Fractional edge coverage avoids moving the face boundary by a full token.
    tokens = pixels.reshape(h // 16, 16, w // 16, 16).mean((1, 3)).reshape(1, -1)
    if not 0 < tokens.mean() < 1:
        raise ValueError("Training face mask is empty or covers the whole target")
    return torch.tensor(tokens, dtype=torch.float32)


@torch.no_grad()
def capture_case(model, backend, tensors, noisy, sigma, config):
    """Cache the exact tensors around the sole trainable insertion point."""
    block = model.single_blocks[-1]
    branch = block.reference_branch
    if branch.output_delta.b.count_nonzero():
        raise ValueError("Prefix cache must be built with zero-initialized BA output")
    target_count = tensors["target_ids"].shape[1]
    text_count = tensors["text_ids"].shape[1]
    record = {}

    def before_block(module, args):
        record["block_input"] = args[0].detach()
        record["native_gate"] = args[2][2].detach()
        record["target_indices"] = torch.arange(text_count, text_count + target_count, device=args[0].device)[None]
        record["image_indices"] = torch.arange(text_count, args[0].shape[1], device=args[0].device)[None]

    def before_projection(module, args):
        record["native_concat"] = args[0].detach()

    def before_final(module, args):
        record["final_vec"] = args[1].detach()

    hooks = [block.register_forward_pre_hook(before_block), block.linear2.register_forward_pre_hook(before_projection),
             model.final_layer.register_forward_pre_hook(before_final)]
    branch.capture = True
    try:
        prediction = backend.predict(model, tensors, noisy, sigma, config, branch=True)
        record.update(branch.features)
    finally:
        branch.capture = False
        branch.features = None
        for hook in hooks:
            hook.remove()
    return {k: v.cpu().contiguous() for k, v in record.items()}, prediction


class CachedFaceTail(nn.Module):
    """Replay native linear2 + residual + final layer, with the same BA module.

Keeping the original concatenated linear2 call preserves BF16 rounding; adding
an independently projected delta to a cached output would change that rounding.
"""
    def __init__(self, model, spec):
        super().__init__()
        self.linear2 = copy.deepcopy(model.single_blocks[-1].linear2).requires_grad_(False)
        self.final_layer = copy.deepcopy(model.final_layer).requires_grad_(False)
        self.reference_branch = copy.deepcopy(model.single_blocks[-1].reference_branch)
        self.reference_branch.capture = False
        self.reference_branch.features = None

    @classmethod
    def from_file(cls, path, spec, device="cuda"):
        from safetensors.torch import load_file
        from extensions_built_in.diffusion_models.flux2.src.model import LastLayer

        state = load_file(path)
        width, inputs = state["linear2.weight"].shape
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.linear2 = nn.Linear(inputs, width, bias=False, dtype=torch.bfloat16).requires_grad_(False)
        result.final_layer = LastLayer(width, state["final_layer.linear.weight"].shape[0]).to(torch.bfloat16).requires_grad_(False)
        result.reference_branch = FaceReferenceRead(width, 24, spec["rank"], spec["gamma"], spec["gate_prior"])
        result.load_state_dict(state, strict=True)
        return result.to(device)

    def forward(self, case, branch=True):
        width = self.reference_branch.width
        original = case["native_concat"]
        delta, logits = self.reference_branch.from_features(case["reference_read"], case["query_features"])
        if branch:
            correction = torch.zeros_like(original[..., :width]).scatter(1, case["target_indices"].unsqueeze(-1).expand(-1, -1, width), delta)
            attention = original[..., :width] + correction
        else:
            attention = original[..., :width]
        joined = torch.cat((attention, original[..., width:]), dim=-1)
        hidden = case["block_input"] + case["native_gate"] * self.linear2(joined)
        images = hidden.index_select(1, case["image_indices"][0])
        prediction = self.final_layer(images, case["final_vec"])
        return prediction[:, :delta.shape[1]], logits


def objective(prediction, logits, case, router_weight):
    mask = case["face_mask"]
    error = (prediction.float() - case["flow_target"].float()).square().mean(-1)
    face = (error * mask).sum() / mask.sum().clamp_min(1)
    background = (error * (1 - mask)).sum() / (1 - mask).sum().clamp_min(1)
    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    router = .5 * ((bce * mask).sum() / mask.sum().clamp_min(1) +
                   (bce * (1 - mask)).sum() / (1 - mask).sum().clamp_min(1))
    # Equal face/background means give the small face a useful gradient budget.
    return face + background + router_weight * router, {"face_mse": face, "background_mse": background, "router_bce": router}


def load_branch(branch, path, expected_identity):
    from safetensors.torch import load_file

    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["identity"] != expected_identity:
        raise ValueError("Diagnostic checkpoint source/config identity differs")
    branch.load_state_dict(load_file(path / "branch.safetensors"), strict=True)
    return manifest
