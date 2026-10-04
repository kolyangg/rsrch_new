"""The refiner starts at the trained core and cannot update frozen weights."""

import torch
from ba_dit.nn.reference_refiner_flow import ReferenceRefinerFlow


def test_frozen_core_zero_delta_and_reference_ablation():
    torch.manual_seed(142)
    model=ReferenceRefinerFlow(width=16,hidden=16,heads=2,channels=4,core_hidden=8,core_heads=2)
    with torch.no_grad():
        model.core.out.weight.normal_(std=.1)
        model.core.noisy_out.weight.normal_(std=.1)
    args=dict(query=torch.randn(2,6,16),key=torch.randn(2,5,16),value=torch.randn(2,5,16),
              noise=torch.randn(2,6,4),timestep=torch.tensor([.2,.8]))
    base=model.core.predict(**args)
    assert torch.equal(model.predict(**args),base)
    assert torch.equal(model.predict(**args,core_velocity=base),base)
    frozen={n:p.detach().clone() for n,p in model.core.named_parameters()}
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=.001)
    target=torch.randn_like(base)
    for step in range(2):
        optimizer.zero_grad();(model.predict(**args)-target).square().mean().backward()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in parameters)
        if step:assert all(p.grad.count_nonzero() for p in parameters)
        optimizer.step()
    assert all(torch.equal(p,frozen[n]) and p.grad is None for n,p in model.core.named_parameters())
    assert torch.equal(model.predict(**args,reference_read=False),base)
    assert not torch.equal(model.predict(**args),base)
    assert torch.equal(model.predict(**args),model.predict(**args,core_velocity=base))
    clone=ReferenceRefinerFlow(width=16,hidden=16,heads=2,channels=4,core_hidden=8,core_heads=2)
    clone.load_state_dict(model.state_dict())
    assert torch.equal(clone.predict(**args),model.predict(**args))
