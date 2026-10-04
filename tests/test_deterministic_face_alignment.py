"""The replacement sampler matches native bilinear values and CPU gradients."""
import torch
from torch.nn import functional as F
from ba_dit.nn.deterministic_identity_loss import aligned_crop


def test_alignment_matches_grid_sample():
    torch.manual_seed(142)
    x=torch.randn(2,3,19,23,requires_grad=True)
    grid=torch.rand(2,11,13,2)*2.4-1.2
    expected=F.grid_sample(x,grid,mode='bilinear',padding_mode='border',align_corners=True)
    actual=aligned_crop(x,grid)
    assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)
    a=torch.autograd.grad(actual.square().sum(),x)[0]
    b=torch.autograd.grad(expected.square().sum(),x)[0]
    assert torch.allclose(a,b,atol=2e-6,rtol=2e-6)
