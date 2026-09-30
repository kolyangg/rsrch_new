"""Check the patched native Qwen attention insertion on a small causal layout."""

import torch
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Attention

from ba_dit.nn.reference_read_delta import ReferenceReadDelta


def test_qwen_attention_branch_zero_effect_and_first_gradient():
    torch.manual_seed(23)
    attention = QwenImage21Attention(dim=16, heads=2, dim_head=8)
    hidden = torch.randn(1, 6, 16, requires_grad=True)
    layout = {"segments": [(0, 2, True), (2, 4, False)], "key_valid": None}
    native = attention(hidden, **layout)
    native_grad = torch.autograd.grad(native[:, 4:].square().sum(), hidden)[0]

    attention.reference_branch = ReferenceReadDelta(width=16, heads=2, rank=2)
    context = (torch.tensor([2, 3]), torch.tensor([4, 5]))
    branch = attention(hidden, branch_context=context, **layout)
    assert torch.equal(native, branch)
    branch[:, 4:].square().sum().backward()
    assert torch.equal(native_grad, hidden.grad)
    assert attention.reference_branch.k_delta.b.grad.abs().sum() > 0
    assert attention.reference_branch.v_delta.b.grad.abs().sum() > 0
