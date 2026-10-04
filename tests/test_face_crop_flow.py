"""The face crop must start as noise; only reference attention predicts flow."""

import torch
from ba_dit.nn.face_crop_flow import FaceCropFlow


def test_zero_flow_and_reference_dependence_after_update():
    torch.manual_seed(142)
    branch = FaceCropFlow(width=16, hidden=8, heads=2, channels=4)
    query, key, value = torch.randn(2, 6, 16), torch.randn(2, 5, 16), torch.randn(2, 5, 16)
    target = torch.randn(2, 6, 4)
    velocity = branch.predict(query, key, value)
    assert not velocity.count_nonzero()
    (velocity-target).square().mean().backward()
    assert branch.out.weight.grad.isfinite().all() and branch.out.weight.grad.count_nonzero()
    with torch.no_grad():
        branch.out.weight.add_(branch.out.weight.grad, alpha=-.1)
    branch.zero_grad()
    learned = branch.predict(query, key, value)
    (learned-target).square().mean().backward()
    assert all(p.grad.isfinite().all() and p.grad.count_nonzero() for p in branch.parameters())
    assert learned.count_nonzero()
    assert not branch.predict(query, key, torch.zeros_like(value)).count_nonzero()


def test_query_residual_still_starts_zero_and_read_can_be_ablated():
    torch.manual_seed(142)
    branch = FaceCropFlow(width=16, hidden=8, heads=2, channels=4, query_residual=True)
    q, k, v = torch.randn(2, 6, 16), torch.randn(2, 5, 16), torch.randn(2, 5, 16)
    assert not branch.predict(q, k, v).count_nonzero()
    with torch.no_grad():
        branch.out.weight.normal_(std=.1)
    assert not torch.equal(branch.predict(q, k, v), branch.predict(q, k, v, reference_read=False))


def test_learned_noise_skip_starts_zero_and_both_paths_receive_gradients():
    torch.manual_seed(142)
    branch = FaceCropFlow(width=16, hidden=8, heads=2, channels=4, noise_skip=True)
    args = dict(query=torch.randn(2,6,16), key=torch.randn(2,5,16), value=torch.randn(2,5,16),
                noise=torch.randn(2,6,4), timestep=torch.tensor([.2,.8]))
    result = branch.predict(**args)
    assert not result.count_nonzero()
    (result-torch.randn_like(result)).square().mean().backward()
    assert branch.out.weight.grad.count_nonzero() and branch.noisy_out.weight.grad.count_nonzero()
