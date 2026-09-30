"""Critical zero-effect, gradient, and reference-support checks."""

import torch

from ba_dit.nn.reference_read_delta import ReferenceReadDelta, stratified_reference_indices


def test_initialized_branch_matches_native_and_learns():
    torch.manual_seed(7)
    branch = ReferenceReadDelta(width=8, heads=2, rank=2, query_chunk=2)
    query = torch.randn(1, 2, 4, 4, requires_grad=True)
    key = torch.randn(1, 2, 3, 4, requires_grad=True)
    value = torch.randn(1, 2, 3, 4, requires_grad=True)
    hidden = torch.randn(1, 3, 8, requires_grad=True)
    target = torch.tensor([1, 3])
    post = lambda k: k / (k.square().mean(dim=-1, keepdim=True) + 1e-6).sqrt()
    correction = branch(query, key, value, hidden, target, post)
    assert torch.equal(correction, torch.zeros_like(correction))
    loss = (correction[:, target] * torch.randn_like(correction[:, target])).sum()
    loss.backward()
    assert branch.k_delta.b.grad is not None and torch.isfinite(branch.k_delta.b.grad).all()
    assert branch.v_delta.b.grad is not None and torch.isfinite(branch.v_delta.b.grad).all()
    assert branch.k_delta.b.grad.abs().sum() > 0
    assert branch.v_delta.b.grad.abs().sum() > 0
    assert torch.equal(correction[:, [0, 2]], torch.zeros_like(correction[:, [0, 2]]))


def test_reference_support_cap_and_empty():
    mask = torch.zeros(4, 5, dtype=torch.bool)
    assert stratified_reference_indices(mask, 3).numel() == 0
    mask[:, 1:4] = True
    selected = stratified_reference_indices(mask, 4)
    assert selected.numel() == 4
    assert mask.flatten()[selected].all()
    assert selected.unique().numel() == 4
