"""One reference-face read, a low-rank output update, and a learned query gate.

The gate receives native target queries, never target boxes or output masks.
This diagnostic is installed only at the final FLUX single-stream block.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ba_dit.nn.reference_read_delta import LowRankProjection


class FaceReferenceRead(nn.Module):
    def __init__(self, width, heads, rank=16, gamma=1.0, gate_prior=0.1):
        super().__init__()
        if width % heads or not 0 < gate_prior < 1 or gamma <= 0:
            raise ValueError("Invalid face branch geometry or initialization")
        self.width, self.heads, self.gamma = width, heads, gamma
        self.output_delta = LowRankProjection(width, rank, rank)
        self.face_gate = nn.Linear(width, 1, dtype=torch.float32)
        nn.init.zeros_(self.face_gate.weight)
        nn.init.constant_(self.face_gate.bias, math.log(gate_prior / (1 - gate_prior)))
        self.capture = False
        self.features = None
        self.last_gate = None

    def from_features(self, reference_read, query_features):
        # Native per-head Q is normalized already; global RMS keeps the router's
        # input scale stable across timesteps without changing the attention Q.
        features = F.normalize(query_features.float(), dim=-1) * math.sqrt(self.width)
        logits = self.face_gate(features).squeeze(-1)
        gate = logits.sigmoid()
        delta = self.output_delta(reference_read) * gate.unsqueeze(-1).to(reference_read.dtype) * self.gamma
        return delta, logits

    def forward(self, query_rotated, reference_key_pre, reference_value, reference_hidden,
                target_indices, key_postprocess, target_gate=None):
        if target_gate is not None:
            raise ValueError("FaceReferenceRead predicts its own gate; external target masks are forbidden")
        batch, heads, queries, head_width = query_rotated.shape
        if (heads, heads * head_width) != (self.heads, self.width) or reference_value.shape[2] == 0:
            raise ValueError("Invalid query geometry or empty reference face")
        q = query_rotated.index_select(2, target_indices)
        read = F.scaled_dot_product_attention(q, key_postprocess(reference_key_pre), reference_value)
        read = read.transpose(1, 2).reshape(batch, -1, self.width)
        features = q.transpose(1, 2).reshape_as(read)
        delta, logits = self.from_features(read, features)
        self.last_gate = logits.detach().sigmoid()
        if self.capture:
            self.features = {"reference_read": read.detach(), "query_features": features.detach()}
        result = query_rotated.new_zeros(batch, queries, self.width)
        result[:, target_indices] = delta
        return result
