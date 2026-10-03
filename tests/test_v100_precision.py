"""Critical precision identity and loss-scaler resume invariants (CPU checks)."""
import copy
import random

import pytest
import torch

from ba_dit.checkpoint import load_adapters, restore_training, save_training
from ba_dit.config import ROOT, adapter_identity, load_config
from ba_dit.nn.sliced_native_lora import SlicedLoRALinear
from ba_dit.validation_masks import mask_directory


def test_mixed_precision_cannot_reuse_fp32_adapter_identity_or_masks():
    config = load_config(ROOT/'configs/clust/flux4b_2v100_amp.yaml')
    full = copy.deepcopy(config)
    del full['model']['compute_precision']
    assert adapter_identity(config) != adapter_identity(full)
    assert mask_directory(config) != mask_directory(full)


def test_scaler_optimizer_and_rng_resume_exactly(tmp_path):
    torch.manual_seed(42)
    random.seed(42)
    model = SlicedLoRALinear(torch.nn.Linear(8,8,bias=False), [(0,8,0,8)], 2,2)
    fresh = copy.deepcopy(model)
    config = load_config(ROOT/'configs/clust/flux4b_2v100_amp.yaml')

    def setup(model):
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda step:1/(step+1)), torch.amp.GradScaler('cpu', init_scale=32, growth_interval=1)

    def update(model, opt, schedule, scaler):
        opt.zero_grad(set_to_none=True)
        loss = (model(torch.randn(2,8))*random.random()).square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        scaler.step(opt); scaler.update(); schedule.step()
        return loss.detach()

    opt, schedule, scaler = setup(model)
    update(model,opt,schedule,scaler)
    checkpoint = save_training(model,opt,schedule,config,'branch_only',tmp_path,1,1,scaler=scaler)
    expected = update(model,opt,schedule,scaler)
    other_opt, other_schedule, other_scaler = setup(fresh)
    load_adapters(fresh,checkpoint,config,'branch_only')
    assert restore_training(other_opt,other_schedule,checkpoint,config,scaler=other_scaler) == (1,1)
    actual = update(fresh,other_opt,other_schedule,other_scaler)
    assert torch.equal(expected,actual)
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),fresh.parameters()))
    assert scaler.state_dict() == other_scaler.state_dict()
    assert schedule.state_dict() == other_schedule.state_dict()
    for key, state in opt.state_dict()['state'].items():
        for name, value in state.items():
            assert torch.equal(value,other_opt.state_dict()['state'][key][name])
    state = torch.load(checkpoint/'training_state.pt',weights_only=True)
    del state['grad_scaler']
    torch.save(state,checkpoint/'training_state.pt')
    with pytest.raises(ValueError, match='missing gradient-scaler'):
        restore_training(other_opt,other_schedule,checkpoint,config,scaler=other_scaler)
