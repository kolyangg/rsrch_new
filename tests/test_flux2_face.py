"""Critical FLUX2 invariants: erased input, one-way states and differentiable reads."""
from dataclasses import replace
import numpy as np
from PIL import Image
import torch
from ba_dit.runtime import prepare_imports
from ba_dit.nn.flux2_face import FaceBranch, predict
from ba_dit.data.flux2_memory import sanitize

prepare_imports('flux')
from extensions_built_in.diffusion_models.flux2.src.model import Flux2, Klein4BParams
from extensions_built_in.diffusion_models.flux2.src.sampling import batched_prc_img


def fixture():
    torch.set_num_threads(2);torch.manual_seed(142)
    model = Flux2(replace(Klein4BParams(),hidden_size=16,num_heads=2,axes_dim=[2,2,2,2],
                         in_channels=8,context_in_dim=8,depth=2,depth_single_blocks=2)).requires_grad_(False)
    model.reference_branch = FaceBranch(16,8,2,2)
    noisy = torch.randn(1,8,2,3)
    _,ids = batched_prc_img(noisy)
    tensors = {'target_face_mask':torch.tensor([[1,1,0,0,0,0]]),
        'flux2_context_keep':torch.tensor([[0,0,0,1,1,1]],dtype=torch.bool),
        'flux2_context':torch.randn_like(noisy),'flux2_context_noise':torch.randn_like(noisy),
        'target_ids':ids,'prompt_embeds':torch.randn(1,3,8),'text_ids':torch.zeros(1,3,4),
        'flux2_identity':torch.randn(1,512),'flux2_detail':torch.randn(1,64,384)}
    return model,tensors,noisy


def test_erased_pixels_cannot_enter_context():
    a=np.zeros((128,128,3),dtype=np.uint8);b=a.copy();b[40:65,40:65]=255
    clean_a,keep_a=sanitize(Image.fromarray(a),[40,40,65,65],16)
    clean_b,keep_b=sanitize(Image.fromarray(b),[40,40,65,65],16)
    assert np.array_equal(clean_a,clean_b) and torch.equal(keep_a,keep_b)


def test_native_face_and_excluded_context_do_not_enter_active_stream():
    model,tensors,noisy=fixture()
    for read in model.reference_branch.reads:torch.nn.init.normal_(read.out.weight,std=.01)
    before=predict(model,tensors,noisy,torch.tensor([.5]))
    changed=noisy.clone();changed.flatten(2)[:,:,2:]+=100
    other={**tensors,'flux2_context':tensors['flux2_context'].clone()}
    other['flux2_context'].flatten(2)[:,:,:3]+=100
    assert torch.equal(before,predict(model,other,changed,torch.tensor([.5])))
    assert not before.flatten(2)[:,:,2:].any()
    donor={**tensors,'flux2_identity':torch.randn(1,512)}
    assert not torch.equal(before,predict(model,donor,noisy,torch.tensor([.5])))


def test_checkpoint_gradients_and_zero_output_initialization():
    model,tensors,noisy=fixture();model.enable_gradient_checkpointing()
    params=list(model.reference_branch.parameters())
    optimizer=torch.optim.AdamW(params,lr=.001)
    for step in range(2):
        optimizer.zero_grad();loss=predict(model,tensors,noisy,torch.tensor([.4])).square().sum();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        if step:assert all(p.grad.abs().sum()>0 for p in params)
        optimizer.step()
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('reference_branch.'))


def test_checkpoint_context_offload_preserves_output_and_gradients():
    import copy
    a,tensors,noisy=fixture();b=copy.deepcopy(a);b.enable_gradient_checkpointing()
    x=predict(a,tensors,noisy,torch.tensor([.4]));y=predict(b,tensors,noisy,torch.tensor([.4]))
    assert torch.equal(x,y)
    x.square().sum().backward();y.square().sum().backward()
    assert all(torch.equal(p.grad,q.grad) for p,q in zip(a.reference_branch.parameters(),b.reference_branch.parameters()))
