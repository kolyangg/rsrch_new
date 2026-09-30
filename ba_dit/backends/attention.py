"""Backend projection/RoPE seams. All added attention arithmetic lives here."""

import torch


def flux_reference_delta(block, hidden, qkv, query, value, pe, context, single: bool):
    from extensions_built_in.diffusion_models.flux2.src.model import apply_rope

    offset = context.text_tokens if single else 0
    refs = context.reference_indices + offset
    targets = torch.arange(offset, offset + context.target_tokens, device=hidden.device)
    width = block.hidden_size
    key_pre = qkv[..., width:2 * width].unflatten(-1, (block.num_heads, -1))
    key_pre = key_pre.index_select(1, refs).permute(0, 2, 1, 3)
    ref_hidden = hidden.index_select(1, refs)
    ref_value = value.index_select(2, refs)
    ref_pe = pe.index_select(2, refs)
    norm = block.norm if single else block.img_attn.norm

    def native_key_postprocess(unmodulated):
        normalized = norm.key_norm(unmodulated).to(ref_value.dtype)
        return apply_rope(normalized, normalized, ref_pe)[1]

    query_native = apply_rope(query, query, pe)[0]
    return block.reference_branch(query_native, key_pre, ref_value, ref_hidden, targets, native_key_postprocess)


def qwen_reference_delta(attn, hidden, query, value, rotary_emb, context, kv_mode):
    from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen

    if kv_mode is not None:
        raise ValueError("Reference branch requires native full attention with KV reuse disabled")
    references, targets = context
    ref_hidden = hidden.index_select(1, references)
    key_pre = attn.to_k(ref_hidden).unflatten(-1, (attn.heads, -1)).permute(0, 2, 1, 3)
    ref_value = value.index_select(1, references).permute(0, 2, 1, 3)
    ref_rotary = None if rotary_emb is None else rotary_emb.index_select(0, references)

    def native_key_postprocess(unmodulated):
        normalized = attn.norm_k(unmodulated.permute(0, 2, 1, 3)).to(ref_value.dtype)
        if ref_rotary is not None:
            normalized = apply_rotary_emb_qwen(normalized, ref_rotary, use_real=False)
        return normalized.permute(0, 2, 1, 3)

    return attn.reference_branch(query.permute(0, 2, 1, 3), key_pre, ref_value, ref_hidden, targets, native_key_postprocess)
