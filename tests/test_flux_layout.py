"""Both FLUX depth/site maps: full forward, LoRA slices and checkpoint contexts."""

import copy
from dataclasses import replace

import torch

from ba_dit import adapters
from ba_dit.config import ROOT, load_config
from extensions_built_in.diffusion_models.flux2.src.model import Flux2, Klein4BParams, Klein9BParams


def test_both_flux_depths_and_checkpointed_sample_contexts():
    torch.set_num_threads(2)
    for profile, params in [("flux4b_48", Klein4BParams()), ("flux9b_80", Klein9BParams())]:
        torch.manual_seed(17)
        model = Flux2(replace(params, hidden_size=16, num_heads=2, axes_dim=[2, 2, 2, 2], in_channels=8, context_in_dim=8))
        config = load_config(ROOT / f"configs/{profile}.yaml")
        config["branch"]["rank"] = config["lora"]["rank"] = 2
        x, text = torch.randn(1, 12, 8), torch.randn(1, 3, 8)
        ids = torch.arange(12).view(1, 12, 1).expand(1, 12, 4).float()
        text_ids = torch.zeros(1, 3, 4)
        inputs = dict(x=x, x_ids=ids, timesteps=torch.tensor([0.5]), ctx=text, ctx_ids=text_ids, guidance=None)
        native = model(**inputs).detach()
        adapters.install(model, config, "lora_plus_branch")
        assert torch.equal(native, model(**inputs, branch_reference_mask=torch.ones(2, 2), branch_target_tokens=8))
        plain = copy.deepcopy(model)
        model.enable_gradient_checkpointing()
        masks = [torch.tensor([[1, 0], [1, 1]]), torch.tensor([[0, 1], [1, 0]])]
        for candidate in (plain, model):
            # Two live forward graphs expose mutable per-sample context mistakes.
            outputs = [candidate(**inputs, branch_reference_mask=mask, branch_target_tokens=8) for mask in masks]
            sum(value[:, :8].square().mean() for value in outputs).backward()
        for (name, a), (_, b) in zip(plain.named_parameters(), model.named_parameters()):
            if a.requires_grad:
                assert a.grad is not None and torch.allclose(a.grad, b.grad, atol=1e-7, rtol=1e-5), name
                if name.endswith('.b'):
                    assert a.grad.count_nonzero() > 0, name
            else:
                assert a.grad is None and b.grad is None
