"""Independent native Q/K/V/O adapters with explicit fused matrix slices."""

import torch
from torch import nn
from torch.nn import functional as F


class SliceAdapter(nn.Module):
    def __init__(self, inputs, outputs, rank, alpha):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, inputs, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(outputs, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)
        self.scale = alpha / rank

    def forward(self, inputs):
        return (F.linear(F.linear(inputs.float(), self.a), self.b) * self.scale).to(inputs.dtype)


class SlicedLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, slices: list[tuple[int, int, int, int]], rank: int, alpha: float):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.slices = slices
        self.adapters = nn.ModuleList()
        for i0, i1, o0, o1 in slices:
            if not 0 <= i0 < i1 <= base.in_features or not 0 <= o0 < o1 <= base.out_features:
                raise ValueError("LoRA slice outside native projection")
            self.adapters.append(SliceAdapter(i1 - i0, o1 - o0, rank, alpha).to(base.weight.device))

    def forward(self, hidden):
        result = self.base(hidden)
        for (i0, i1, o0, o1), adapter in zip(self.slices, self.adapters):
            result[..., o0:o1] = result[..., o0:o1] + adapter(hidden[..., i0:i1])
        return result
