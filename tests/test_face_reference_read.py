"""Critical face-BA invariants: native init, query scope and trainable routing."""

import torch
from torch.nn import functional as F

from ba_dit.nn.face_reference_read import FaceReferenceRead


def test_face_read_scope_initialization_and_gradients():
    torch.manual_seed(3)
    branch = FaceReferenceRead(16, 2, rank=2)
    q, k, v = torch.randn(1, 2, 6, 8), torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8)
    targets = torch.tensor([2, 3, 4])
    args = (q, k, v, torch.randn(1, 3, 16), targets, lambda key: key)
    zero = branch(*args)
    assert torch.equal(zero, torch.zeros_like(zero))
    branch.capture = True
    branch(*args)
    delta, logits = branch.from_features(**branch.features)
    loss = (delta - 1).square().mean() + F.binary_cross_entropy_with_logits(logits, torch.tensor([[0., 1., 0.]]))
    loss.backward()
    assert branch.output_delta.b.grad.isfinite().all() and branch.output_delta.b.grad.count_nonzero()
    assert branch.face_gate.weight.grad.isfinite().all() and branch.face_gate.weight.grad.count_nonzero()
    with torch.no_grad():
        branch.output_delta.b.add_(-.1 * branch.output_delta.b.grad)
    changed = branch(*args)
    assert changed[:, targets].count_nonzero()
    assert not changed[:, [0, 1, 5]].count_nonzero()
    assert not branch(q, k, torch.zeros_like(v), args[3], targets, args[5]).count_nonzero()
    try:
        branch(*args, target_gate=torch.ones(1, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("An inference target mask was accepted")
