"""Three reference reads, initialized from the best single-read BA checkpoint."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from ba_dit.nn.reference_refiner_flow import ReferenceRefinerFlow


def norm(x):
    return F.layer_norm(x.float(), (x.shape[-1],))


class ReferenceReadBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.SiLU(), nn.Linear(2*width, width))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(8.)))
        # Exact identity at initialization; both residuals learn immediately.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, x, key, value):
        split = lambda z: z.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        q, k, v = self.q(norm(x)), self.k(norm(key)), self.v(norm(value))
        read = F.scaled_dot_product_attention(
            F.normalize(split(q), dim=-1)*self.log_temperature.exp().clamp(1., 16.),
            F.normalize(split(k), dim=-1), split(v), scale=1.)
        x = x + self.out(read.transpose(1, 2).flatten(2))
        return x + self.ff(norm(x))


class DeepIdentityFlow(ReferenceRefinerFlow):
    """Only the BA refiner trains; the learned 512-wide BA core stays frozen.

    Each query is independent: sampled-token training and full-scene inference
    use the same computation. K/V are the captured reference-face tokens.
    """
    def __init__(self, width=3072, hidden=1024, heads=16, channels=128, depth=2):
        super().__init__(width, hidden, heads, channels)
        self.reads = nn.ModuleList([ReferenceReadBlock(hidden, heads) for _ in range(depth)])

    def predict(self, query, key, value, reference_read=True, noise=None, timestep=None, core_velocity=None):
        if noise is None or timestep is None:
            raise ValueError('Identity flow needs the current latent and sigma')
        if core_velocity is None:
            with torch.no_grad():
                core_velocity = self.core.predict(query, key, value, noise=noise, timestep=timestep)
        base = core_velocity.detach()
        if not reference_read:
            return base
        q, k, v = self.q(norm(query)), self.k(norm(key)), self.v(norm(value))
        split = lambda x: x.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        read = F.scaled_dot_product_attention(
            F.normalize(split(q), dim=-1)*self.log_temperature.exp().clamp(1., 16.),
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
        for block in self.reads:
            features = block(features, k, v)
        correction = self.out(features)
        correction = correction + self.identity_noise_gate(identity).sigmoid()*self.noisy_out(noise.float())
        return base + correction
