"""A zero-output reference refiner on a frozen, already trained BA flow head."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ba_dit.nn.conditioned_face_flow import ConditionedFaceFlow
from ba_dit.nn.face_crop_flow import FaceCropFlow


class ReferenceRefinerFlow(FaceCropFlow):
    ablation_metric = "refiner_off_mse"

    def __init__(self, width=3072, hidden=1024, heads=16, channels=128, core_hidden=512, core_heads=8):
        super().__init__(width, hidden, heads, channels, False, True, False)
        self.core = ConditionedFaceFlow(width, core_hidden, core_heads, channels).requires_grad_(False)
        self.time = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.noisy_in = nn.Linear(channels, hidden, bias=False)
        self.identity_modulation = nn.Linear(hidden, 2*hidden)
        self.identity_noise_gate = nn.Linear(hidden, channels)
        self.mix = nn.Linear(hidden, hidden)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(8.)))

    def predict(self, query, key, value, reference_read=True, noise=None, timestep=None, core_velocity=None):
        if noise is None or timestep is None:
            raise ValueError("Reference refinement needs the current latent and sigma")
        if core_velocity is None:
            with torch.no_grad():
                core_velocity = self.core.predict(query, key, value, noise=noise, timestep=timestep)
        base = core_velocity.detach()
        # This ablation disables only the NEW refiner. The trained core still
        # reads the reference, so it must not be labeled 'all reference off'.
        if not reference_read:
            return base
        norm = lambda x: F.layer_norm(x.float(), (x.shape[-1],))
        q, k, v = self.q(norm(query)), self.k(norm(key)), self.v(norm(value))
        split = lambda x: x.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        temperature = self.log_temperature.exp().clamp(1., 16.)
        read = F.scaled_dot_product_attention(
            F.normalize(split(q), dim=-1)*temperature,
            F.normalize(split(k), dim=-1), split(v), scale=1.)
        read = norm(read.transpose(1, 2).flatten(2))
        identity = norm(v.mean(dim=1, keepdim=True))
        gain, shift = self.identity_modulation(identity).chunk(2, dim=-1)
        sigma = timestep.float().reshape(-1, 1)
        time = self.time(torch.cat((sigma, 1-sigma), -1))[:, None]
        context = norm(q) + time + self.noisy_in(noise.float())
        context = context*(1 + .5*gain.tanh()) + .5*shift.tanh()
        features = read*F.silu(context) + identity
        features = features + F.silu(self.mix(F.silu(features)))
        correction = self.out(features)
        correction = correction + self.identity_noise_gate(identity).sigmoid()*self.noisy_out(noise.float())
        return base + correction
