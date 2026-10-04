"""Face ownership must not alter text, reference or background rows at a site."""
import torch

from ba_dit.runtime import prepare_imports
from ba_dit.nn.masked_face_attention import MaskedFaceAttention
from ba_dit.nn.batched_face_attention import reference_batch

prepare_imports('flux')
from extensions_built_in.diffusion_models.flux2.src.model import (
    DoubleStreamBlock, EmbedND, FluxBranchContext, SingleStreamBlock)


def test_mask_routes_only_selected_queries_and_output_lora():
    torch.manual_seed(42)
    width,heads=16,2
    position=EmbedND(dim=8,theta=2000,axes_dim=[2,2,2,2])
    image=torch.randn(2,5,width);text=torch.randn(2,2,width)
    pe=position(torch.arange(5).repeat(4,1).T[None].float().expand(2,-1,-1))
    text_pe=position(torch.arange(2).repeat(4,1).T[None].float().expand(2,-1,-1))
    mod=tuple(torch.randn(2,1,width) for _ in range(3))
    mask=torch.tensor([[1.,0.,0.],[0.,0.,0.]])
    context=FluxBranchContext(3,torch.tensor([3,4]),2,mask)
    for single in (False,True):
        block=(SingleStreamBlock if single else DoubleStreamBlock)(width,heads,mlp_ratio=2)
        block.requires_grad_(False)
        args=(torch.cat((text,image),1),torch.cat((text_pe,pe),2),mod) if single else (
            image,text,pe,text_pe,(mod,mod),(mod,mod))
        native=block(*args)
        block.reference_branch=MaskedFaceAttention(width,heads,rank=2,alpha=2)
        routed=block(*args,context)
        native_image=native if single else native[0]
        routed_image=routed if single else routed[0]
        offset=2 if single else 0
        assert not torch.equal(native_image[0,offset],routed_image[0,offset])
        keep=torch.ones(native_image.shape[:2],dtype=torch.bool);keep[0,offset]=False
        assert torch.equal(native_image[keep],routed_image[keep])
        if not single:assert torch.equal(native[1],routed[1])
        routed_image[0,offset].square().sum().backward()
        assert all(p.grad is not None and p.grad.abs().sum()>0 for n,p in
                   block.reference_branch.named_parameters() if n.endswith('.b'))
        # Nonzero output LoRA must retain the same ownership boundary.
        with torch.no_grad():block.reference_branch.o_delta.b.fill_(.1)
        changed=block(*args,context);changed=changed if single else changed[0]
        assert torch.equal(native_image[keep],changed[keep])


def test_batch_reference_faces_are_selected_per_example():
    """Opposite reference masks must match isolated reads and gradients."""
    torch.manual_seed(71)
    width, heads = 16, 2
    position = EmbedND(dim=8, theta=2000, axes_dim=[2,2,2,2])
    hidden = torch.randn(2,5,width)
    pe = position(torch.arange(5).repeat(4,1).T[None].float().expand(2,-1,-1))
    native = torch.randn_like(hidden)
    block = DoubleStreamBlock(width, heads, mlp_ratio=2).requires_grad_(False)
    branch = MaskedFaceAttention(width, heads, rank=2, alpha=2)
    qkv = block.img_attn.qkv(hidden)
    mask = torch.tensor([[1.,0.,.5],[0.,1.,0.]])
    context = FluxBranchContext(3, torch.tensor([3,4]), 0, mask)
    refs = torch.tensor([[[True,False]], [[False,True]]])
    with reference_batch(refs, 2):
        batched, delta = branch.route_attention(block, hidden, qkv, native, pe, context, False)
    isolated, isolated_delta = [], []
    for i in range(2):
        ctx = FluxBranchContext(3, torch.tensor([3+i]), 0, mask[i:i+1])
        message, change = branch.route_attention(block, hidden[i:i+1], qkv[i:i+1],
            native[i:i+1], pe[i:i+1], ctx, False)
        isolated.append(message); isolated_delta.append(change)
    assert torch.equal(batched, torch.cat(isolated))
    assert torch.equal(delta, torch.cat(isolated_delta))
    actual = torch.autograd.grad((batched+delta).square().sum(), tuple(branch.parameters()), retain_graph=True)
    expected = torch.autograd.grad((torch.cat(isolated)+torch.cat(isolated_delta)).square().sum(), tuple(branch.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
