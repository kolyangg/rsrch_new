"""Native-initialized Q/K/V/O LoRA with explicit reference-face ownership.

The face mask routes attention messages inside FLUX, before its native output
projection. The branch output LoRA is added after that projection, before the
native modulation gate. Neither path replaces the pretrained velocity head.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ba_dit.nn.reference_read_delta import LowRankProjection


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
        ref_hidden = hidden.index_select(1, refs)
        key = qkv[..., self.width:2*self.width].index_select(1, refs) + self.k_delta(ref_hidden)
        value = qkv[..., 2*self.width:3*self.width].index_select(1, refs) + self.v_delta(ref_hidden)
        key = norm.key_norm(split(key)).to(value.dtype)
        key = apply_rope(key, key, pe.index_select(2, refs))[1]
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

    if config['model']['arch'] != 'flux2_klein_4b' or config['branch']['gamma'] != 1:
        raise ValueError('This named experiment requires FLUX Base4B and full face routing')
    model.requires_grad_(False)
    for kind, sites in SITES['flux2_klein_4b'].items():
        for index in sites:
            block = getattr(model, kind+'_blocks')[index]
            if hasattr(block, 'reference_branch'):
                raise ValueError('Expected a fresh native transformer')
            spec = config['branch']
            block.reference_branch = MaskedFaceAttention(model.hidden_size, model.num_heads,
                spec['rank'], spec['alpha'], spec['query_chunk']).to(next(block.parameters()).device)


def training_mask(row, config):
    """Photo masks are training supervision, transformed exactly like VAE inputs."""
    from PIL import Image
    from ba_dit.data.geometry import target_geometry
    from ba_dit.nn.masked_face_flow import face_alpha, token_alpha

    with Image.open(row['target']) as image:
        _, geometry = target_geometry(image, config['data']['target_size'], row['target_box'], geometry_only=True)
    width, height = geometry['source_wh']
    rw, rh = geometry['resize_wh']
    left, top, _, _ = geometry['crop_xyxy']
    x0, y0, x1, y1 = row['target_box']
    box = [x0*rw/width-left, y0*rh/height-top, x1*rw/width-left, y1*rh/height-top]
    h, w = config['data']['target_size']
    return token_alpha(face_alpha((w, h), box, config['branch']['mask_feather_pixels'])).flatten(1)
