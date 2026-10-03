"""The measured V100 precision policy, shared by training and validation."""
from contextlib import nullcontext
from functools import wraps

import torch


POLICY = 'amp_fp16_fp32_branch'


def mixed(config):
    return config['model'].get('compute_precision') == POLICY


def compute_context(config):
    return (torch.autocast('cuda', dtype=torch.float16, cache_enabled=False)
            if mixed(config) else nullcontext())


def fp32_inputs(module, args, kwargs):
    def cast(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.float()
        if isinstance(value, (tuple, list)):
            return type(value)(cast(item) for item in value)
        return value
    return tuple(cast(value) for value in args), {key:cast(value) for key,value in kwargs.items()}


def configure_residuals(model, config):
    if mixed(config):
        for block in [*model.double_blocks, *model.single_blocks]:
            block.register_forward_pre_hook(fp32_inputs, with_kwargs=True)
    return model


def fp32_route(route):
    @wraps(route)
    def wrapped(*args, **kwargs):
        args, kwargs = fp32_inputs(None, args, kwargs)
        with torch.autocast('cuda', enabled=False):
            return route(*args, **kwargs)
    return wrapped


def configure_branches(model, config):
    if mixed(config):
        for block in [*model.double_blocks, *model.single_blocks]:
            if hasattr(block, 'reference_branch'):
                branch = block.reference_branch
                branch.route_attention = fp32_route(branch.route_attention)


def make_scaler(config):
    return torch.amp.GradScaler('cuda', enabled=mixed(config) or config['model'].get('dtype') == 'float16',
                               init_scale=32.)


def conditioning_devices(encoder):
    devices = {torch.cuda.current_device()}
    device = encoder.text_encoder.device
    if device.type == 'cuda':
        devices.add(device.index if device.index is not None else torch.cuda.current_device())
    return sorted(devices)


def device_memory(config):
    devices = [torch.cuda.current_device()]
    if config['data'].get('encoder_device') == 'cuda:1' and 1 not in devices:
        devices.append(1)
    return {str(i): {'peak_reserved_gib':torch.cuda.max_memory_reserved(i)/2**30,
                     'reserved_fraction':torch.cuda.max_memory_reserved(i)/torch.cuda.get_device_properties(i).total_memory}
            for i in devices}
