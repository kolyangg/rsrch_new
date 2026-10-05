"""Information-flow, native-parity, checkpoint and spatial-ownership invariants."""
import copy
from dataclasses import replace

import torch

from ba_dit.config import ROOT, load_config
from ba_dit import adapters
from ba_dit.runtime import prepare_imports
from ba_dit.nn.isolated_reference import reference_bank
from ba_dit.nn.masked_face_flow import face_alpha, routing_token_alpha, scene_latent

prepare_imports('flux')
from extensions_built_in.diffusion_models.flux2.src.model import Flux2, Klein9BParams


def fixture():
    torch.set_num_threads(2)
    torch.manual_seed(142)
    model = Flux2(replace(Klein9BParams(), hidden_size=16, num_heads=2, axes_dim=[2,2,2,2],
                         in_channels=8, context_in_dim=8)).requires_grad_(False)
    config = load_config(ROOT/'configs/FLUX1a_vast_9b.yaml')
    config['branch'].update(rank=2, alpha=2)
    x = torch.randn(1,10,8)
    ids = torch.arange(10).view(1,10,1).expand(1,10,4).float()
    inputs = dict(x=x, x_ids=ids, timesteps=torch.tensor([.5]), ctx=torch.randn(1,3,8),
                  ctx_ids=torch.zeros(1,3,4), guidance=None)
    return model, config, inputs


def joint_reference_features(model, inputs):
    values, hooks = {}, []
    for kind in ('double','single'):
        for index, block in enumerate(getattr(model,kind+'_blocks')):
            if not hasattr(block,'reference_branch'):
                continue
            projection = block.img_attn.qkv if kind=='double' else block.linear1
            offset = 6 if kind=='double' else 9
            def hook(_module, _inputs, output, key=f'{kind}.{index}', start=offset):
                values[key] = output[:,start:,:3*model.hidden_size].detach().clone()
            hooks.append(projection.register_forward_hook(hook))
    try:
        model(**inputs)
    finally:
        for hook in hooks:
            hook.remove()
    return values


def bank_for(model, inputs):
    return reference_bank(model,inputs['x'][:,6:],inputs['x_ids'][:,6:],inputs['timesteps'],
                          torch.ones(2,2,dtype=torch.bool),4)


def routed(model, inputs, bank, mask=None):
    return model(**inputs,branch_reference_mask=torch.ones(2,2,dtype=torch.bool),branch_target_tokens=6,
                 branch_target_mask=torch.ones(1,6) if mask is None else mask,branch_reference_bank=bank)


def test_joint_bank_has_target_feedback_isolated_bank_does_not():
    model,config,inputs=fixture()
    adapters.install(model,config,'branch_only')
    changed={**inputs,'x':inputs['x'].clone()}
    changed['x'][:,:6]+=3
    first=joint_reference_features(model,inputs)
    second=joint_reference_features(model,changed)
    assert len(first)==8 and all(not torch.equal(first[k],second[k]) for k in first)
    a,b=bank_for(model,inputs),bank_for(model,changed)
    assert len(a)==8
    for key in a:
        for name in ('hidden','key','value','position'):
            assert torch.equal(getattr(a[key],name),getattr(b[key],name))
            assert not getattr(a[key],name).requires_grad
    donor={**inputs,'x':inputs['x'].clone()}
    donor['x'][:,6:]+=torch.randn_like(donor['x'][:,6:])
    c=bank_for(model,donor)
    assert all(not torch.equal(a[key].value,c[key].value) for key in a)
    # A branch-only reference swap keeps the native input byte-identical.
    assert not torch.equal(routed(model,inputs,a),routed(model,inputs,c))


def test_native_and_zero_mask_parity_with_pure_bank():
    model,config,inputs=fixture()
    native=model(**inputs).detach()
    adapters.install(model,config,'branch_only')
    bank=bank_for(model,inputs)
    assert torch.equal(native,model(**inputs))
    assert torch.equal(native,routed(model,inputs,bank,torch.zeros(1,6)))
    assert not torch.equal(native,routed(model,inputs,bank))


def test_checkpointed_two_graphs_keep_distinct_reference_banks_and_gradients():
    model,config,inputs=fixture()
    adapters.install(model,config,'branch_only')
    plain=copy.deepcopy(model)
    model.enable_gradient_checkpointing()
    second={**inputs,'x':inputs['x'].clone()}
    second['x'][:,6:]*=-2
    for candidate in (plain,model):
        outputs=[routed(candidate,item,bank_for(candidate,item)) for item in (inputs,second)]
        sum(o[:,:6].square().mean() for o in outputs).backward()
    for (name,a),(_,b) in zip(plain.named_parameters(),model.named_parameters()):
        if a.requires_grad:
            assert a.grad is not None and b.grad is not None, name
            assert torch.allclose(a.grad,b.grad,atol=2e-7,rtol=2e-5),name
            if name.endswith('.b'):
                assert a.grad.abs().sum()>0,name
        else:
            assert a.grad is None and b.grad is None


def test_binary_support_removes_native_latent_contribution_at_face_boundary():
    soft=load_config(ROOT/'configs/FLUX1a_vast_9b.yaml')
    soft['branch']['token_ownership']='soft'
    hard=load_config(ROOT/'configs/FLUX1a_vast_9b.yaml')
    alpha=face_alpha((96,96),[23,19,67,73],16)
    a,b=routing_token_alpha(alpha,soft),routing_token_alpha(alpha,hard)
    assert torch.equal(a>0,b>0) and ((a>0)&(a<1)).any()
    native=torch.ones(1,2,6,6,requires_grad=True)
    noise=torch.zeros_like(native)
    face=torch.full_like(native,2.)
    x=scene_latent(face,native,noise,.4,b)
    selected=(b>0).expand_as(x)
    assert torch.equal(x[selected],face[selected])
    grad=torch.autograd.grad(x[selected].sum(),native)[0]
    assert grad.count_nonzero()==0
    mixed=scene_latent(face,native,noise,.4,a)
    assert not torch.equal(mixed[selected],face[selected])


def test_isolated_reference_batch_matches_single_predictions():
    model,config,inputs=fixture()
    adapters.install(model,config,'branch_only')
    reference=torch.cat((inputs['x'][:,6:],-inputs['x'][:,6:]))
    ids=inputs['x_ids'][:,6:].expand(2,-1,-1)
    actual=reference_bank(model,reference,ids,torch.tensor([.3,.7]),torch.ones(2,2),4)
    for row in range(2):
        expected=reference_bank(model,reference[row:row+1],ids[row:row+1],torch.tensor([.3,.7])[row:row+1],torch.ones(2,2),4)
        for key in actual:
            assert torch.allclose(actual[key].key[row:row+1],expected[key].key,atol=2e-6,rtol=2e-5)


def test_runtime_donor_changes_only_branch_bank_and_ba_off_rejects_override():
    from ba_dit.backends.flux_runtime import predict
    model,config,inputs=fixture()
    adapters.install(model,config,'branch_only')
    tensors={'reference_tokens':inputs['x'][:,6:], 'reference_ids':inputs['x_ids'][:,6:],
             'target_ids':inputs['x_ids'][:,:6], 'reference_mask':torch.ones(2,2),
             'prompt_embeds':inputs['ctx'], 'text_ids':inputs['ctx_ids'], 'target_face_mask':torch.ones(1,6)}
    frozen={k:v.clone() for k,v in tensors.items()}
    noisy=inputs['x'][:,:6].transpose(1,2).reshape(1,8,2,3)
    own=predict(model,tensors,noisy,inputs['timesteps'],config)
    donor={**tensors,'reference_tokens':-2*tensors['reference_tokens']}
    changed=predict(model,tensors,noisy,inputs['timesteps'],config,reference_source=donor)
    assert not torch.equal(own,changed)
    assert all(torch.equal(v,frozen[k]) for k,v in tensors.items())
    expected_bank=reference_bank(model,donor['reference_tokens'],donor['reference_ids'],inputs['timesteps'],donor['reference_mask'],512)
    expected=routed(model,inputs,expected_bank)[:,:6].transpose(1,2).reshape_as(noisy)
    assert torch.equal(expected,changed)
    try:
        predict(model,tensors,noisy,inputs['timesteps'],config,False,reference_source=donor)
    except ValueError:
        pass
    else:
        raise AssertionError('BA-off must not silently apply a reference swap')


def _distributed_reference_worker(rank, rendezvous):
    import torch.distributed as dist
    from datetime import timedelta
    from torch.nn.parallel import DistributedDataParallel
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        model,config,inputs=fixture()
        adapters.install(model,config,'branch_only')
        model.enable_gradient_checkpointing()
        wrapped=DistributedDataParallel(model,broadcast_buffers=False)
        inputs={**inputs,'x':inputs['x']+rank*.2}
        for _ in range(2):
            wrapped.zero_grad(set_to_none=True)
            bank=bank_for(wrapped,inputs)
            routed(wrapped,inputs,bank)[:,:6].square().mean().backward()
            for parameter in model.parameters():
                if not parameter.requires_grad:
                    assert parameter.grad is None
                    continue
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                other=parameter.grad.clone()
                dist.broadcast(other,src=0)
                assert torch.equal(parameter.grad,other)
    finally:
        dist.destroy_process_group()


def test_two_workers_keep_reference_pass_outside_ddp_reducer(tmp_path):
    torch.multiprocessing.spawn(_distributed_reference_worker,args=(str(tmp_path/'rendezvous'),),nprocs=2,join=True)
