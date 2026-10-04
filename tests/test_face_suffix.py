"""Critical invariant: earlier BA receives gradients through frozen later blocks."""

import torch

from ba_dit.face_suffix import FaceSuffix, SupervisedFaceRead, objective, settings
from ba_dit.runtime import backend_module


def test_suffix_native_parity_and_checkpoint_gradients():
    _, config = settings("configs/flux4b_face_one_id_strong.yaml")
    backend_module(config)
    from extensions_built_in.diffusion_models.flux2.src.model import SingleStreamBlock, LastLayer

    torch.manual_seed(4)
    blocks = [SingleStreamBlock(32, 4, 3.).requires_grad_(False) for _ in range(2)]
    for block in blocks:
        block.reference_branch = SupervisedFaceRead(32, 4, rank=4)
    tail = FaceSuffix(blocks, LastLayer(32, 8).requires_grad_(False))
    case = {"x": torch.randn(2, 10, 32), "pe": torch.eye(2).repeat(2, 1, 10, 4, 1, 1),
            "shift": torch.randn(2, 1, 32), "scale": torch.randn(2, 1, 32), "gate": torch.randn(2, 1, 32),
            "vec": torch.randn(2, 32), "layout": torch.tensor([[2, 4], [2, 4]]),
            "refs": torch.tensor([[4, 6], [4, 6]]), "face_mask": torch.tensor([[0., 1., 1., 0.]]).repeat(2, 1),
            "flow_target": torch.randn(2, 4, 8)}
    native, _ = tail(case, branch=False)
    prediction, logits = tail(case)
    assert torch.equal(native, prediction)
    objective(prediction, logits, case, 1.)[0].backward()
    assert all(b.reference_branch.output_delta.b.grad.count_nonzero() for b in blocks)
    assert all(p.grad is None for p in tail.parameters() if not p.requires_grad)
    with torch.no_grad():
        for b in blocks:
            b.reference_branch.output_delta.b.normal_(std=.02)
    grads = []
    for checkpointed in (False, True):
        tail.zero_grad(set_to_none=True)
        tail.checkpoint_blocks = checkpointed
        prediction, logits = tail(case)
        objective(prediction, logits, case, 1.)[0].backward()
        grads.append({n: p.grad.clone() for n, p in tail.trainable().items()})
    for name in grads[0]:
        assert torch.equal(grads[0][name], grads[1][name]), name
        assert grads[0][name].isfinite().all()
        assert grads[0][name].count_nonzero()
