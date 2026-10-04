"""Focused invariants for the zero-initialized, time-conditioned BA head."""
import torch
from ba_dit.nn.conditioned_face_flow import ConditionedFaceFlow


def test_zero_start_gradients_reference_and_time():
    torch.manual_seed(142)
    b=ConditionedFaceFlow(width=16,hidden=16,heads=2,channels=4)
    data=dict(query=torch.randn(2,6,16),key=torch.randn(2,5,16),value=torch.randn(2,5,16),
              noise=torch.randn(2,6,4),timestep=torch.tensor([.2,.8]))
    assert not b.predict(**data).count_nonzero()
    target=torch.randn(2,6,4);opt=torch.optim.AdamW(b.parameters(),lr=.001)
    for step in range(2):
        opt.zero_grad();(b.predict(**data)-target).square().mean().backward()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in b.parameters())
        if step:assert all(p.grad.count_nonzero() for p in b.parameters())
        opt.step()
    original=b.predict(**data)
    assert not torch.equal(original,b.predict(**data,reference_read=False))
    changed={**data,'timestep':data['timestep']*.5}
    assert not torch.equal(original,b.predict(**changed))
    restored=ConditionedFaceFlow(width=16,hidden=16,heads=2,channels=4)
    restored.load_state_dict(b.state_dict())
    assert torch.equal(original,restored.predict(**data))
