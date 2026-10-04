"""Time-conditioned BA flow head with bounded attention logits.

The frozen feature capture seam is identical to FaceCropFlow. Only this module
is trained; native velocity is never used to predict the masked face.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ba_dit.nn.face_crop_flow import FaceCropFlow


class ConditionedFaceFlow(FaceCropFlow):
    def __init__(self, width=3072, hidden=512, heads=8, channels=128):
        super().__init__(width, hidden, heads, channels, True, True, False)
        self.time = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.mix = nn.Linear(hidden, hidden)

    def predict(self, query, key, value, reference_read=True, noise=None, timestep=None):
        if noise is None or timestep is None:
            raise ValueError("Conditioned BA needs the current noisy latent and sigma")
        norm = lambda x: F.layer_norm(x.float(), (x.shape[-1],))
        q, k, v = self.q(norm(query)), self.k(norm(key)), self.v(norm(value))
        split = lambda x: x.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        # Cosine attention bounds logits to [-4, 4], even after long training.
        read = F.scaled_dot_product_attention(
            F.normalize(split(q), dim=-1), F.normalize(split(k), dim=-1), split(v), scale=4.)
        read = norm(read.transpose(1, 2).flatten(2))
        if not reference_read:
            read = torch.zeros_like(read)
        sigma = timestep.float().reshape(-1, 1)
        time = self.time(torch.cat((sigma, 1-sigma), -1))[:, None]
        features = norm(q) + read + time
        features = features + F.silu(self.mix(F.silu(features)))
        return self.out(features) + self.noisy_out(noise.float())
