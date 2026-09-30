"""A low-rank correction to native target queries' reference read."""

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LowRankProjection(nn.Module):
    def __init__(self, width: int, rank: int, alpha: float):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, width, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(width, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)
        self.scale = alpha / rank

    def forward(self, hidden: Tensor) -> Tensor:
        # FP32 parameters remain trainable while BF16 activations use the native dtype.
        return (F.linear(F.linear(hidden.float(), self.a), self.b) * self.scale).to(hidden.dtype)


def stratified_reference_indices(mask_hw: Tensor, maximum: int) -> Tensor:
    """Select spatially spread nonzero reference tokens in row-major order."""
    if mask_hw.ndim != 2 or maximum <= 0:
        raise ValueError("Expected a 2D token mask and positive key cap")
    valid = mask_hw.flatten().nonzero(as_tuple=False).flatten()
    if valid.numel() <= maximum:
        return valid
    positions = torch.linspace(0, valid.numel() - 1, steps=maximum, device=valid.device).round().long()
    return valid[positions]


class ReferenceReadDelta(nn.Module):
    """Compute R1-R0 with identical reference support and target-only output."""

    def __init__(self, width: int, heads: int, rank: int = 16, alpha: float = 16, gamma: float = 0.1, query_chunk: int = 128):
        super().__init__()
        if width % heads or rank <= 0 or query_chunk <= 0 or gamma < 0:
            raise ValueError("Invalid branch geometry or scale")
        self.width = width
        self.heads = heads
        self.gamma = gamma
        self.query_chunk = query_chunk
        self.k_delta = LowRankProjection(width, rank, alpha)
        self.v_delta = LowRankProjection(width, rank, alpha)

    def forward(
        self,
        query_rotated: Tensor,
        reference_key_pre: Tensor,
        reference_value: Tensor,
        reference_hidden: Tensor,
        target_indices: Tensor,
        key_postprocess: Callable[[Tensor], Tensor],
        target_gate: Tensor | None = None,
    ) -> Tensor:
        """Return [B,Q,D] correction before the native attention output projection.

        Inputs Q/K/V are [B,H,L,head_dim]; reference_hidden is [B,R,D].
        key_postprocess must apply the native key norm and original reference RoPE.
        Callers gather the *same* valid reference indices for both reads first.
        """
        batch, heads, queries, head_width = query_rotated.shape
        if heads != self.heads or heads * head_width != self.width:
            raise ValueError("Query width does not match registered branch")
        references = reference_hidden.shape[1]
        if reference_hidden.shape != (batch, references, self.width):
            raise ValueError("Invalid reference hidden shape")
        if reference_key_pre.shape != (batch, heads, references, head_width) or reference_value.shape != reference_key_pre.shape:
            raise ValueError("Reference K/V layout mismatch")
        if target_indices.ndim != 1 or (target_indices.numel() and (target_indices.min() < 0 or target_indices.max() >= queries)):
            raise ValueError("Invalid target query indices")
        correction = query_rotated.new_zeros((batch, queries, self.width))
        if references == 0 or target_indices.numel() == 0:
            return correction

        dk = self.k_delta(reference_hidden).reshape(batch, references, heads, head_width).permute(0, 2, 1, 3)
        dv = self.v_delta(reference_hidden).reshape(batch, references, heads, head_width).permute(0, 2, 1, 3)
        k0 = key_postprocess(reference_key_pre)
        k1 = key_postprocess(reference_key_pre + dk)
        v1 = reference_value + dv
        if k0.shape != reference_key_pre.shape or k1.shape != k0.shape:
            raise ValueError("Native key postprocess changed the reference layout")
        outputs = []
        for index_chunk in target_indices.split(self.query_chunk):
            q = query_rotated.index_select(2, index_chunk)
            native = F.scaled_dot_product_attention(q, k0, reference_value)
            adapted = F.scaled_dot_product_attention(q, k1, v1)
            outputs.append((adapted - native).transpose(1, 2).reshape(batch, len(index_chunk), self.width))
        delta = torch.cat(outputs, dim=1) * self.gamma
        if target_gate is not None:
            if target_gate.shape != (batch, target_indices.numel()):
                raise ValueError("Target gate must match selected queries")
            delta = delta * target_gate.unsqueeze(-1)
        correction[:, target_indices, :] = delta
        return correction
