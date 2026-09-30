"""The patched FLUX blocks must preserve native output at branch initialization."""

import sys
import types
from pathlib import Path

import torch

from ba_dit.nn.reference_read_delta import ReferenceReadDelta


ROOT = Path(__file__).resolve().parents[1]
TOOLKIT = ROOT / "sources/ai-toolkit-flux"
sys.path.insert(0, str(TOOLKIT))
for package, relative in (
    ("extensions_built_in", "extensions_built_in"),
    ("extensions_built_in.diffusion_models", "extensions_built_in/diffusion_models"),
    ("extensions_built_in.diffusion_models.flux2", "extensions_built_in/diffusion_models/flux2"),
):
    module = types.ModuleType(package)
    module.__path__ = [str(TOOLKIT / relative)]
    sys.modules[package] = module

from extensions_built_in.diffusion_models.flux2.src.model import (  # noqa: E402
    DoubleStreamBlock,
    EmbedND,
    FluxBranchContext,
    SingleStreamBlock,
)


def test_double_and_single_stream_branch_seams():
    torch.manual_seed(19)
    width, heads = 16, 2
    pe = EmbedND(dim=8, theta=2000, axes_dim=[2, 2, 2, 2])
    img = torch.randn(1, 4, width, requires_grad=True)
    txt = torch.randn(1, 2, width, requires_grad=True)
    image_pe = pe(torch.arange(4).repeat(4, 1).T[None].float())
    text_pe = pe(torch.arange(2).repeat(4, 1).T[None].float())
    triple = tuple(torch.randn(1, 1, width) for _ in range(3))
    double = DoubleStreamBlock(width, heads, mlp_ratio=2)
    single = SingleStreamBlock(width, heads, mlp_ratio=2)
    context = FluxBranchContext(
        target_tokens=2, reference_indices=torch.tensor([2, 3]), text_tokens=2,
    )

    native_img, native_txt = double(img, txt, image_pe, text_pe, (triple, triple), (triple, triple))
    double.reference_branch = ReferenceReadDelta(width, heads, rank=2)
    branch_img, branch_txt = double(img, txt, image_pe, text_pe, (triple, triple), (triple, triple), context)
    assert torch.equal(native_img, branch_img)
    assert torch.equal(native_txt, branch_txt)

    joined = torch.cat((native_txt, native_img), dim=1)
    joined_pe = torch.cat((text_pe, image_pe), dim=2)
    native = single(joined, joined_pe, triple)
    single.reference_branch = ReferenceReadDelta(width, heads, rank=2)
    adapted = single(joined, joined_pe, triple, context)
    assert torch.equal(native, adapted)

    loss = branch_img[:, :2].square().sum() + adapted[:, 2:4].square().sum()
    loss.backward()
    for block in (double, single):
        assert block.reference_branch.k_delta.b.grad.abs().sum() > 0
        assert block.reference_branch.v_delta.b.grad.abs().sum() > 0
