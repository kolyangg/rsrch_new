"""Face ownership must not alter text, reference or background rows at a site."""
import torch

from ba_dit.runtime import prepare_imports
from ba_dit.nn.masked_face_attention import MaskedFaceAttention

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
