"""Qwen native causal prefix and 32-block site map survive checkpoint recomputation."""

import copy

import torch
from diffusers import QwenImage21Transformer2DModel

from ba_dit import adapters
from ba_dit.config import ROOT, load_config


def test_qwen_causal_prefix_and_checkpointed_sample_contexts():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    model = QwenImage21Transformer2DModel(in_channels=8, out_channels=8, num_layers=32, attention_head_dim=8,
            num_attention_heads=2, context_in_dim=8, axes_dims_rope=(2, 2, 4))
    config = load_config(ROOT / "configs/qwen7b_80.yaml")
    config["branch"]["rank"] = config["lora"]["rank"] = 2
    inputs = dict(hidden_states=torch.randn(1, 12, 8), encoder_hidden_states=torch.randn(1, 3, 8),
                  timestep=torch.tensor([0.5]), img_shapes=[[(1, 2, 2), (1, 2, 4)]],
                  img_mask=torch.tensor([[False, True, False, True, True]]), return_dict=False)
    native = model(**inputs)[0].detach()
    adapters.install(model, config, "lora_plus_branch")
    assert torch.equal(native, model(**inputs, attention_kwargs={"branch_reference_mask": torch.ones(2, 2)})[0])
    plain = copy.deepcopy(model)
    model.enable_gradient_checkpointing()
    masks = [torch.tensor([[1, 0], [1, 1]]), torch.tensor([[0, 1], [1, 0]])]
    for candidate in (plain, model):
        outputs = [candidate(**inputs, attention_kwargs={"branch_reference_mask": mask})[0] for mask in masks]
        sum(value[:, -8:].square().mean() for value in outputs).backward()
    for (name, a), (_, b) in zip(plain.named_parameters(), model.named_parameters()):
        if a.requires_grad:
            assert a.grad is not None and torch.allclose(a.grad, b.grad, atol=1e-7, rtol=1e-5), name
            if name.endswith('.b'):
                assert a.grad.count_nonzero() > 0, name
        else:
            assert a.grad is None and b.grad is None
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'reference_branch' in name and name.endswith('.b'):
                parameter.normal_(0, 0.01)
        changed = model(**inputs, attention_kwargs={"branch_reference_mask": masks[0]})[0]
        assert torch.equal(changed[:, :4], native[:, :4])  # Clean reference prefix.
        assert not torch.equal(changed[:, -8:], native[:, -8:])
