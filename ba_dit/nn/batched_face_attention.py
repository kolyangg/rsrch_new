"""Per-example reference reads during a batched native FLUX forward/backward.

The native stream has equal, unpadded token layouts. Only the small BA reads
run per example so each image keeps its own reference-key and target masks.
The context must enclose backward too, including activation recomputation.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace

import torch

from ba_dit.nn.reference_read_delta import stratified_reference_indices

_indices = ContextVar('face_batch_reference_indices', default=None)


@contextmanager
def reference_batch(masks, maximum):
    indices = [stratified_reference_indices(mask, maximum) for mask in masks]
    token = _indices.set(indices)
    try:
        yield
    finally:
        _indices.reset(token)


def active_indices():
    return _indices.get()


def route_rows(module, block, hidden, qkv, native, pe, context, single):
    indices = active_indices()
    if len(indices) != hidden.shape[0]:
        raise ValueError('Reference-mask batch does not match hidden states')
    messages, deltas = [], []
    for i, keys in enumerate(indices):
        row_context = replace(context, reference_indices=keys+context.target_tokens,
                              target_mask=context.target_mask[i:i+1])
        row_pe = pe[i:i+1] if pe.shape[0] == hidden.shape[0] else pe
        message, delta = module.route_attention(block, hidden[i:i+1], qkv[i:i+1],
            native[i:i+1], row_pe, row_context, single)
        messages.append(message); deltas.append(delta)
    return torch.cat(messages), torch.cat(deltas)


def collate(pairs):
    keys = pairs[0].keys()
    if any(pair.keys() != keys or any(pair[k].shape != pairs[0][k].shape for k in keys) for pair in pairs):
        raise ValueError('True batches require equal native token shapes; no silent image/token padding')
    result = {k:torch.cat([p[k] for p in pairs], dim=0) for k in keys if k != 'reference_mask'}
    result['reference_masks'] = torch.stack([p['reference_mask'] for p in pairs])
    # The upstream context accepts a shared grid. Per-example BA selection above
    # replaces its indices; native joint attention remains entirely unchanged.
    result['reference_mask'] = result['reference_masks'].any(dim=0)
    return result


def training_loss(backend, model, tensors, config):
    target = tensors['target_latent']
    noisy, noise, sigma = [], [], []
    for sample in target.split(1):
        epsilon = torch.randn_like(sample)
        schedule = backend.scheduler()
        times = schedule.set_train_timesteps(1000, device=sample.device, timestep_type='sigmoid', latents=sample, patch_size=1)
        timestep = times[torch.randint(0, 999, (1,), device=sample.device)]
        noisy.append(schedule.add_noise(sample, epsilon, timestep).to(sample.dtype))
        noise.append(epsilon); sigma.append(timestep.to(sample.dtype)/1000)
    prediction = backend.predict(model, tensors, torch.cat(noisy), torch.cat(sigma), config, True)
    error = (prediction.float()-(torch.cat(noise)-target).float()).square()
    mask = tensors['target_face_mask'].reshape(target.shape[0], 1, *target.shape[-2:])
    denominator = mask.sum((1,2,3))*target.shape[1]
    if not (denominator > 0).all():
        raise ValueError('Empty face supervision in batch')
    # Match averaging individual face-normalized losses, independent of face size.
    return ((error*mask).sum((1,2,3))/denominator).mean()
