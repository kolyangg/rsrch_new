"""A standalone reference-attention flow head: random Q/K/V, zero output.

This diagnostic predicts the entire velocity for a face crop. Frozen FLUX
provides features; its native velocity is never added to this prediction.
"""

import torch
from torch import nn
from torch.nn import functional as F


class FaceFeaturesCaptured(Exception):
    pass


class FaceCropFlow(nn.Module):
    def __init__(self, width=3072, hidden=256, heads=4, channels=128, query_residual=False, noise_skip=False, timestep_scaling=True):
        super().__init__()
        self.heads = heads
        self.query_residual = query_residual
        self.noise_skip = noise_skip
        self.timestep_scaling = timestep_scaling
        self.q = nn.Linear(width, hidden, bias=False, dtype=torch.float32)
        self.k = nn.Linear(width, hidden, bias=False, dtype=torch.float32)
        self.v = nn.Linear(width, hidden, bias=False, dtype=torch.float32)
        self.out = nn.Linear(hidden, channels, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.out.weight)
        if noise_skip:
            self.noisy_out = nn.Linear(channels, channels, bias=False, dtype=torch.float32)
            nn.init.zeros_(self.noisy_out.weight)
        self.features = None

    def predict(self, query, key, value, reference_read=True, noise=None, timestep=None):
        # Native Q/K were normalized and rotated before capture. Branch-only
        # projections operate on these frozen features; no second RoPE is added.
        q = self.q(F.layer_norm(query.float(), (query.shape[-1],)))
        k = self.k(F.layer_norm(key.float(), (key.shape[-1],)))
        v = self.v(F.layer_norm(value.float(), (value.shape[-1],)))
        split = lambda x: x.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        read = F.scaled_dot_product_attention(split(q), split(k), split(v))
        read = read.transpose(1, 2).flatten(2)
        if not reference_read:
            read = torch.zeros_like(read)
        output = self.out(read + q if self.query_residual else read)
        if self.noise_skip:
            if noise is None or timestep is None:
                raise ValueError("Noise-skip BA requires the current noisy latent and timestep")
            # Learned x0/flow parameterization: both numerator paths start at
            # zero. There is no hardcoded native denoising or identity skip.
            output = output + self.noisy_out(noise.float())
            if self.timestep_scaling:
                output = output / timestep.reshape(-1, 1, 1).float().clamp_min(.001)
        return output

    def forward(self, query_rotated, reference_key_pre, reference_value, reference_hidden,
                target_indices, key_postprocess, target_gate=None):
        if target_gate is not None:
            raise ValueError("Face-crop inference never accepts target boxes or masks")
        flatten = lambda x: x.transpose(1, 2).flatten(2).detach()
        self.features = {"query": flatten(query_rotated.index_select(2, target_indices)),
                         "key": flatten(key_postprocess(reference_key_pre)),
                         "value": flatten(reference_value)}
        # Stop before native final projections: only the new BA head supplies
        # denoising velocity. The same capture seam is used in train and infer.
        raise FaceFeaturesCaptured


def install(model, query_residual=False, noise_skip=False, timestep_scaling=True):
    model.requires_grad_(False)
    if any(hasattr(b, "reference_branch") for b in (*model.double_blocks, *model.single_blocks)):
        raise ValueError("Expected a fresh frozen FLUX backbone")
    branch = FaceCropFlow(query_residual=query_residual, noise_skip=noise_skip, timestep_scaling=timestep_scaling).to(model.device)
    model.single_blocks[-1].reference_branch = branch
    return branch


@torch.no_grad()
def capture(model, branch, backend, tensors, noisy, sigma, config):
    branch.features = None
    try:
        backend.predict(model, tensors, noisy, sigma, config, branch=True)
    except FaceFeaturesCaptured:
        if branch.features is None:
            raise RuntimeError("The BA feature seam did not run")
        if branch.noise_skip:
            branch.features.update(noise=noisy.flatten(2).transpose(1, 2).detach(), timestep=sigma.detach())
        return branch.features
    raise RuntimeError("Unexpected native prediction: BA feature capture was bypassed")
