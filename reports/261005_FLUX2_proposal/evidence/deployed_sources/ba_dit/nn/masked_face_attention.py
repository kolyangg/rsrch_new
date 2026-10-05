"""Native-initialized Q/K/V/O LoRA with explicit reference-face ownership.

The face mask routes attention messages inside FLUX, before its native output
projection. The branch output LoRA is added after that projection, before the
native modulation gate. Neither path replaces the pretrained velocity head.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ba_dit.nn.reference_read_delta import LowRankProjection


def parameter_count(config):
    from ba_dit.backends.flux2_native import SITES
    arch = config['model']['arch']
    width = {'flux2_klein_4b':3072, 'flux2_klein_9b':4096}[arch]
    return sum(map(len, SITES[arch].values()))*8*width*config['branch']['rank']


class MaskedFaceAttention(nn.Module):
    def __init__(self, width, heads, rank, alpha, query_chunk=128):
        super().__init__()
        self.width, self.heads, self.query_chunk = width, heads, query_chunk
        self.q_delta = LowRankProjection(width, rank, alpha)
        self.k_delta = LowRankProjection(width, rank, alpha)
        self.v_delta = LowRankProjection(width, rank, alpha)
        self.o_delta = LowRankProjection(width, rank, alpha)

    def route_attention(self, block, hidden, qkv, native_attention, pe, context, single):
        from extensions_built_in.diffusion_models.flux2.src.model import apply_rope
        from ba_dit.nn.batched_face_attention import active_indices, route_rows

        if hidden.shape[0] > 1 and active_indices() is not None:
            return route_rows(self, block, hidden, qkv, native_attention, pe, context, single)

        mask = context.target_mask
        if mask is None or mask.shape != (hidden.shape[0], context.target_tokens):
            raise ValueError('Masked face attention needs an explicit [batch, target_tokens] mask')
        offset = context.text_tokens if single else 0
        selected = mask.gt(0).any(dim=0).nonzero(as_tuple=False).flatten()
        output_delta = torch.zeros_like(native_attention)
        if not selected.numel():
            return native_attention, output_delta
        refs = context.reference_indices + offset
        if not refs.numel():
            raise ValueError('Face branch has no reference-face keys')
        queries = selected + offset
        norm = block.norm if single else block.img_attn.norm
        split = lambda x: x.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        bank = getattr(context, 'reference_bank', None)
        if bank is not None:
            features = bank[self.site_name]
            ref_hidden, key, value, ref_pe = features.hidden, features.key, features.value, features.position
            if ref_hidden.shape[0] != hidden.shape[0]:
                raise ValueError('Isolated reference-bank batch differs from query batch')
        else:
            ref_hidden = hidden.index_select(1, refs)
            key = qkv[..., self.width:2*self.width].index_select(1, refs)
            value = qkv[..., 2*self.width:3*self.width].index_select(1, refs)
            ref_pe = pe.index_select(2, refs)
        key = key + self.k_delta(ref_hidden)
        value = value + self.v_delta(ref_hidden)
        key = norm.key_norm(split(key)).to(value.dtype)
        key = apply_rope(key, key, ref_pe)[1]
        value = split(value)
        messages = []
        for indices in queries.split(self.query_chunk):
            query = qkv[..., :self.width].index_select(1, indices) + self.q_delta(hidden.index_select(1, indices))
            query = norm.query_norm(split(query)).to(value.dtype)
            query = apply_rope(query, query, pe.index_select(2, indices))[0]
            read = F.scaled_dot_product_attention(query, key, value)
            messages.append(read.transpose(1, 2).flatten(2))
        message = torch.cat(messages, dim=1)
        gate = mask.index_select(1, selected).unsqueeze(-1)
        native = native_attention.index_select(1, queries)
        routed = torch.lerp(native.float(), message.float(), gate.float()).to(native.dtype)
        # Retain exact native values at zero-mask rows, including other batch rows.
        routed = torch.where(gate > 0, routed, native)
        attention = native_attention.index_copy(1, queries, routed)
        output_delta = output_delta.index_copy(1, queries, self.o_delta(message)*gate.to(message.dtype))
        return attention, output_delta


def install(model, config):
    from ba_dit.backends.flux2_native import SITES

    if config['model']['arch'] not in SITES or config['branch']['gamma'] != 1:
        raise ValueError('This experiment requires FLUX Klein Base and full face routing')
    model.requires_grad_(False)
    for kind, sites in SITES[config['model']['arch']].items():
        for index in sites:
            block = getattr(model, kind+'_blocks')[index]
            if hasattr(block, 'reference_branch'):
                raise ValueError('Expected a fresh native transformer')
            spec = config['branch']
            block.reference_branch = MaskedFaceAttention(model.hidden_size, model.num_heads,
                spec['rank'], spec['alpha'], spec['query_chunk']).to(next(block.parameters()).device)
            block.reference_branch.site_name = f'{kind}.{index}'


def training_mask(row, config):
    """Photo masks are training supervision, transformed exactly like VAE inputs."""
    from PIL import Image
    from ba_dit.data.geometry import target_geometry
    from ba_dit.nn.masked_face_flow import face_alpha, routing_token_alpha

    with Image.open(row['target']) as image:
        _, geometry = target_geometry(image, config['data']['target_size'], row['target_box'], geometry_only=True)
    width, height = geometry['source_wh']
    rw, rh = geometry['resize_wh']
    left, top, _, _ = geometry['crop_xyxy']
    x0, y0, x1, y1 = row['target_box']
    box = [x0*rw/width-left, y0*rh/height-top, x1*rw/width-left, y1*rh/height-top]
    h, w = config['data']['target_size']
    return routing_token_alpha(face_alpha((w, h), box, config['branch']['mask_feather_pixels']), config).flatten(1)
