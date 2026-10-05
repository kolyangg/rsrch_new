"""Register trainable modules once, before optimizer/checkpoint construction."""

import torch

from ba_dit.backends.flux2_native import SITES
from ba_dit.nn.sliced_native_lora import SlicedLoRALinear


MODES = ("native", "branch_only", "lora_only", "lora_plus_branch")
QWEN_SITES = (4, 8, 12, 16, 20, 24, 28, 31)


def install(model, config, mode):
    if mode not in MODES:
        raise ValueError(mode)
    model.requires_grad_(False)
    branch = config["branch"]
    if mode in {"branch_only", "lora_plus_branch"}:
        options = dict(rank=branch["rank"], alpha=branch["alpha"], gamma=branch["gamma"], query_chunk=branch["query_chunk"])
        if branch.get('kind') == 'flux2_face':
            if mode != 'branch_only':
                raise ValueError('FLUX2 trains only its face branch')
            from ba_dit.nn.flux2_face import install as install_flux2
            install_flux2(model, config)
        elif branch.get('kind') == 'masked_face_qkvo':
            if mode != 'branch_only':
                raise ValueError('Masked one-ID Q/K/V/O trains only branch-local adapters')
            from ba_dit.nn.masked_face_attention import install as install_face_attention
            install_face_attention(model, config)
        elif config["model"]["backend"] == "flux":
            from ba_dit.backends.flux2_native import install_reference_branch
            install_reference_branch(model, config["model"]["arch"], **options)
        else:
            from ba_dit.backends.qwen21 import install_reference_branch
            install_reference_branch(model, sites=QWEN_SITES, **options)
    if mode in {"lora_only", "lora_plus_branch"}:
        rank, alpha = config["lora"]["rank"], config["lora"]["alpha"]

        def wrap(parent, name, slices=None):
            layer = getattr(parent, name)
            geometry = slices or [(0, layer.in_features, 0, layer.out_features)]
            setattr(parent, name, SlicedLoRALinear(layer, geometry, rank, alpha))

        if config["model"]["backend"] == "qwen":
            for site in QWEN_SITES:
                attention = model.transformer_blocks[site].attn
                for name in ("to_q", "to_k", "to_v"):
                    wrap(attention, name)
                layer = attention.to_out[0]
                attention.to_out[0] = SlicedLoRALinear(layer, [(0, layer.in_features, 0, layer.out_features)], rank, alpha)
        else:
            width = model.hidden_size
            qkv = [(0, width, index * width, (index + 1) * width) for index in range(3)]
            for site in SITES[config["model"]["arch"]]["double"]:
                attention = model.double_blocks[site].img_attn
                wrap(attention, "qkv", qkv)
                wrap(attention, "proj")
            for site in SITES[config["model"]["arch"]]["single"]:
                block = model.single_blocks[site]
                wrap(block, "linear1", qkv)
                wrap(block, "linear2", [(0, width, 0, width)])
    from ba_dit.precision import configure_branches
    configure_branches(model, config)
    inventory = {name: list(parameter.shape) for name, parameter in model.named_parameters() if parameter.requires_grad}
    if mode != "native" and not inventory:
        raise ValueError("No trainable adapters were registered")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            raise TypeError(f"Trainable parameter is not FP32: {name}")
    return inventory


def groups(model, learning_rate):
    branch, lora = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (branch if "reference_branch" in name else lora).append(parameter)
    all_parameters = branch + lora
    if len({id(parameter) for parameter in all_parameters}) != len(all_parameters):
        raise ValueError("Duplicate optimizer parameter")
    return [{"params": group, "lr": learning_rate, "name": name} for name, group in (("branch", branch), ("lora", lora)) if group]
