"""Frozen image-only reference features; no target, prompt or live-stream feedback."""
from dataclasses import dataclass
from types import MappingProxyType

import torch

from ba_dit.nn.reference_read_delta import stratified_reference_indices


@dataclass(frozen=True)
class ReferenceFeatures:
    hidden: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    position: torch.Tensor


@torch.no_grad()
def reference_bank(model, reference, position_ids, sigma, face_mask, maximum):
    """Run the same frozen backbone on reference tokens alone at the current sigma.

    Capture pre-Q/K-normalization features, preserving original reference RoPE
    coordinates. Return a new immutable mapping per prediction; checkpointed
    blocks receive it in their own FluxBranchContext, never a mutable global.
    """
    # The frozen image-only pass must not enter DDP's forward/reducer lifecycle.
    # The differentiable target pass still goes through DDP in flux_runtime.
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if face_mask.ndim != 2 or face_mask.numel() != reference.shape[1]:
        raise ValueError('Isolated reference bank requires one shared reference grid')
    if any(p.requires_grad for n, p in model.named_parameters() if '.reference_branch.' not in n):
        raise ValueError('Reference bank requires a frozen native backbone')
    selected = stratified_reference_indices(face_mask.to(reference.device), maximum)
    if not selected.numel():
        raise ValueError('Isolated bank has no reference face tokens')
    positions = model.pe_embedder(position_ids).index_select(2, selected).detach().clone()
    bank, hooks = {}, []
    width = model.hidden_size

    def capture(name):
        def hook(_module, inputs, output):
            hidden = inputs[0].index_select(1, selected).detach().clone()
            kv = output[..., width:3 * width].index_select(1, selected).detach().clone()
            bank[name] = ReferenceFeatures(hidden, kv[..., :width], kv[..., width:], positions)
        return hook

    try:
        for kind in ('double', 'single'):
            for index, block in enumerate(getattr(model, kind + '_blocks')):
                if hasattr(block, 'reference_branch'):
                    projection = block.img_attn.qkv if kind == 'double' else block.linear1
                    hooks.append(projection.register_forward_hook(capture(f'{kind}.{index}')))
        if not hooks:
            raise ValueError('Install reference branches before building the bank')
        batch = reference.shape[0]
        # Zero text tokens: even target-caption information cannot enter this bank.
        text = reference.new_empty(batch, 0, model.txt_in.in_features)
        ids = position_ids.new_empty(batch, 0, position_ids.shape[-1])
        model(x=reference, x_ids=position_ids, timesteps=sigma.reshape(-1).expand(batch),
              ctx=text, ctx_ids=ids, guidance=None)
    finally:
        for hook in hooks:
            hook.remove()
    if len(bank) != len(hooks):
        raise RuntimeError('Incomplete isolated reference bank')
    return MappingProxyType(bank)
