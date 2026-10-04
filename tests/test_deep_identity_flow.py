"""The added reads preserve the parent and cannot couple sampled target tokens."""
import torch
from ba_dit.nn.deep_identity_flow import ReferenceReadBlock


def test_zero_residual_and_independent_queries():
    torch.manual_seed(142)
    block=ReferenceReadBlock(16,2)
    x,k,v=torch.randn(1,9,16),torch.randn(1,5,16),torch.randn(1,5,16)
    assert torch.equal(block(x,k,v),x)
    optimizer=torch.optim.AdamW(block.parameters(),lr=.001)
    for _ in range(2):
        optimizer.zero_grad();block(x,k,v).square().mean().backward();optimizer.step()
    assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.count_nonzero() for p in block.parameters())
    chosen=torch.tensor([0,4,8])
    assert torch.allclose(block(x,k,v)[:,chosen],block(x[:,chosen],k,v),atol=1e-6,rtol=1e-6)
