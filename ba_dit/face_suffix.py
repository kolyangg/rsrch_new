"""Face BA in a differentiable FLUX suffix, after an exactly cached prefix."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file

from ba_dit.config import ROOT, adapter_identity
from ba_dit.face_diagnostic import settings as base_settings
from ba_dit.nn.face_reference_read import FaceReferenceRead


def settings(path):
    spec, config = base_settings(path)
    config["training"].update(warmup=spec["warmup"], validation_every=spec["validation_every"])
    if not 1 <= spec["blocks"] <= 20:
        raise ValueError("Expected a suffix of the 20 native single-stream blocks")
    return spec, config


def identity(config, spec):
    paths = ["ba_dit/face_suffix.py", "scripts/face_suffix.py", "ba_dit/nn/face_reference_read.py",
             "ba_dit/face_diagnostic.py", "ba_dit/backends/attention.py"]
    return {"variant": "face_suffix_v1_shared_cfg", "base": adapter_identity(config), "diagnostic": spec,
            "code_sha256": hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in paths)).hexdigest()}


class SupervisedFaceRead(FaceReferenceRead):
    def from_features(self, reference_read, query_features):
        delta, logits = super().from_features(reference_read, query_features)
        self.router_logits = logits
        return delta, logits


def install(model, spec):
    model.requires_grad_(False)
    if any(hasattr(b, "reference_branch") for b in (*model.double_blocks, *model.single_blocks)):
        raise ValueError("Expected a fresh native model")
    for block in model.single_blocks[-spec["blocks"]:]:
        block.reference_branch = SupervisedFaceRead(model.hidden_size, model.num_heads, spec["rank"],
                                                    spec["gamma"], spec["gate_prior"]).to(model.device)
    return FaceSuffix(model.single_blocks[-spec["blocks"]:], model.final_layer)


class PrefixCaptured(Exception):
    pass


@torch.no_grad()
def capture_prefix(model, backend, tensors, noisy, sigma, config, spec):
    record = {}

    def vec_hook(module, args, output):
        record["vec"] = output.detach()

    def prefix_hook(module, args):
        x, pe, mod, context = args
        record.update(x=x.detach(), pe=pe.detach(), shift=mod[0].detach(), scale=mod[1].detach(),
                      gate=mod[2].detach(), refs=context.reference_indices[None],
                      layout=torch.tensor([[context.text_tokens, context.target_tokens]], device=x.device))
        raise PrefixCaptured

    hooks = [model.time_in.register_forward_hook(vec_hook),
             model.single_blocks[-spec["blocks"]].register_forward_pre_hook(prefix_hook)]
    try:
        backend.predict(model, tensors, noisy, sigma, config, True)
        raise RuntimeError("Prefix capture hook was not reached")
    except PrefixCaptured:
        pass
    finally:
        for hook in hooks:
            hook.remove()
    return record


class FaceSuffix(nn.Module):
    def __init__(self, blocks, final_layer):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.final_layer = final_layer
        self.checkpoint_blocks = False

    @classmethod
    def from_file(cls, path, spec, device="cuda"):
        from extensions_built_in.diffusion_models.flux2.src.model import SingleStreamBlock, LastLayer

        # Construct without a second random CPU copy of the frozen billion-parameter suffix.
        with torch.device("meta"):
            blocks = [SingleStreamBlock(3072, 24, mlp_ratio=3.).to(torch.bfloat16).requires_grad_(False)
                      for _ in range(spec["blocks"])]
            final = LastLayer(3072, 128).to(torch.bfloat16).requires_grad_(False)
            for block in blocks:
                block.reference_branch = SupervisedFaceRead(3072, 24, spec["rank"], spec["gamma"], spec["gate_prior"])
            result = cls(blocks, final)
        result.load_state_dict(load_file(path), strict=True, assign=True)
        return result.to(device)

    def forward(self, case, branch=True):
        from extensions_built_in.diffusion_models.flux2.src.model import FluxBranchContext

        text, targets = (int(v) for v in case["layout"][0])
        context = FluxBranchContext(targets, case["refs"][0], text) if branch else None
        x, pe = case["x"], case["pe"]
        mod = (case["shift"], case["scale"], case["gate"])
        logits = []

        def apply(block, hidden):
            hidden = block(hidden, pe, mod, context)
            return hidden, block.reference_branch.router_logits if branch else hidden.new_zeros(hidden.shape[0], targets)

        for block in self.blocks:
            # Every frozen block after the first BA remains in the autograd graph.
            if self.checkpoint_blocks and torch.is_grad_enabled():
                x, gate = checkpoint(apply, block, x, use_reentrant=False)
            else:
                x, gate = apply(block, x)
            logits.append(gate)
        prediction = self.final_layer(x[:, text:], case["vec"])[:, :targets]
        return prediction, torch.stack(logits)

    def branch_state(self):
        return {k: v for k, v in self.state_dict().items() if ".reference_branch." in k}

    def trainable(self):
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if not params or any(".reference_branch." not in n for n in params):
            raise RuntimeError("Only BA parameters may train")
        return params


def load_branch(tail, path, expected):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["identity"] != expected:
        raise ValueError("Checkpoint source/config identity changed")
    from ba_dit.data.manifest import file_hash
    if file_hash(path / "branch.safetensors") != manifest["branch_sha256"]:
        raise ValueError("Checkpoint checksum failed")
    state = load_file(path / "branch.safetensors")
    if set(state) != set(tail.branch_state()):
        raise ValueError("Checkpoint branch sites/keys differ")
    tail.load_state_dict(state, strict=False)
    return manifest


def objective(prediction, logits, case, router_weight):
    from ba_dit.face_diagnostic import objective as single_objective

    # Supervise each router separately, rather than supervising only its average.
    loss, parts = single_objective(prediction, logits[0], case, router_weight=0.)
    mask = case["face_mask"].expand_as(logits)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    router = .5 * ((bce * mask).sum() / mask.sum().clamp_min(1) +
                   (bce * (1 - mask)).sum() / (1 - mask).sum().clamp_min(1))
    parts["router_bce"] = router
    return loss + router_weight * router, parts
