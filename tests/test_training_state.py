"""Critical optimizer resume and fused LoRA scope invariants."""

import copy
import random
import tempfile
from pathlib import Path

import torch

from ba_dit.checkpoint import load_adapters, restore_training, save_training
from ba_dit.config import ROOT, load_config
from ba_dit.nn.sliced_native_lora import SlicedLoRALinear
from ba_dit.training import sample_at


def test_fused_lora_excludes_mlp_and_updates():
    torch.manual_seed(4)
    layer = SlicedLoRALinear(torch.nn.Linear(8, 40, bias=False), [(0, 8, i * 8, (i + 1) * 8) for i in range(3)], 2, 2)
    x = torch.randn(2, 3, 8)
    native = layer.base(x)
    assert torch.equal(layer(x), native)
    optimizer = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=0.01)
    layer(x).square().mean().backward()
    assert all(adapter.b.grad.abs().sum() > 0 for adapter in layer.adapters)
    assert all(adapter.a.grad.count_nonzero() == 0 for adapter in layer.adapters)
    optimizer.step()
    changed = layer(x)
    assert not torch.equal(changed[..., :24], native[..., :24])
    assert torch.equal(changed[..., 24:], native[..., 24:])
    assert layer.base.weight.grad is None


def test_resume_matches_continuous_optimizer_rng_and_sample_order():
    torch.manual_seed(42)
    random.seed(42)
    model = SlicedLoRALinear(torch.nn.Linear(8, 8, bias=False), [(0, 8, 0, 8)], 2, 2)
    initial = copy.deepcopy(model)
    config = load_config(ROOT / "configs/flux4b_48.yaml")

    def optimization(model):
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1 / (s + 1))

    def step(model, optimizer, scheduler, cursor):
        optimizer.zero_grad(set_to_none=True)
        sample = sample_at(list(range(7)), cursor, 42)
        loss = (model(torch.randn(2, 8)) * random.random() - sample).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.detach()

    optimizer, scheduler = optimization(model)
    step(model, optimizer, scheduler, 0)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = save_training(model, optimizer, scheduler, config, "lora_only", directory, 1, 1)
        continuous = step(model, optimizer, scheduler, 1)
        restored = initial
        restored_optimizer, restored_scheduler = optimization(restored)
        load_adapters(restored, checkpoint, config, "lora_only")
        update, cursor = restore_training(restored_optimizer, restored_scheduler, checkpoint, config)
        assert (update, cursor) == (1, 1)
        resumed = step(restored, restored_optimizer, restored_scheduler, cursor)
        assert torch.equal(continuous, resumed)
        assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
        assert optimizer.param_groups[0]["lr"] == restored_optimizer.param_groups[0]["lr"]
        wrong = copy.deepcopy(config)
        wrong["branch"]["rank"] += 1
        try:
            load_adapters(restored, checkpoint, wrong, "lora_only")
        except ValueError:
            pass
        else:
            raise AssertionError("Incompatible adapter identity was accepted")
