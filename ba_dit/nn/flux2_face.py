"""FLUX2: directed T→T, C→T/C, F→T/C/F attention and immutable ID/detail reads.

Only the face stream carries gradients. Native weights, text and sanitized context
are frozen; the face state persists through every native attention/MLP block.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class FaceRead(nn.Module):
    def __init__(self, width, rank, double):
        super().__init__()
        self.q = nn.Linear(width, rank, bias=False)
        self.k_id = nn.Linear(rank, rank, bias=False)
        self.v_id = nn.Linear(rank, rank, bias=False)
        self.k_detail = nn.Linear(rank, rank, bias=False)
        self.v_detail = nn.Linear(rank, rank, bias=False)
        self.out = nn.Linear(rank, width, bias=False)
        self.mod = nn.Linear(32, (6 if double else 3)*width, bias=False)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.mod.weight)

    def forward(self, face, identity, detail):
        q = self.q(F.layer_norm(face.float(), (face.shape[-1],))).unsqueeze(1)
        def read(k, v, memory):
            return F.scaled_dot_product_attention(q, k(memory).unsqueeze(1), v(memory).unsqueeze(1)).squeeze(1)
        return self.out(read(self.k_id, self.v_id, identity) + .5*read(self.k_detail, self.v_detail, detail)).to(face.dtype)


class FaceBranch(nn.Module):
    def __init__(self, width, rank, doubles, singles):
        super().__init__()
        self.id_tokens = nn.Linear(512, 4*rank, bias=False)
        self.detail_tokens = nn.Linear(384, rank, bias=False)
        self.detail_position = nn.Parameter(torch.randn(1, 64, rank)*.02)
        self.id_bottleneck = nn.Linear(512, 32, bias=False)
        self.reads = nn.ModuleList(FaceRead(width, rank, i < doubles) for i in range(doubles+singles))
        self.rank = rank

    def memories(self, tensors):
        identity = F.normalize(tensors['flux2_identity'].float(), dim=-1)
        return (self.id_tokens(identity).reshape(-1, 4, self.rank),
                self.detail_tokens(tensors['flux2_detail'].float()) + self.detail_position,
                self.id_bottleneck(identity))


def install(model, config):
    model.reference_branch = FaceBranch(model.hidden_size, config['branch']['rank'],
        len(model.double_blocks), len(model.single_blocks)).to(next(model.parameters()).device)


def parameter_count(config):
    width, doubles, singles = (3072, 5, 20) if config['model']['arch'].endswith('4b') else (4096, 8, 24)
    rank = config['branch']['rank']
    return (512*4*rank + 384*rank + 64*rank + 512*32 +
            (doubles+singles)*(2*width*rank+4*rank*rank) + (doubles*6+singles*3)*32*width)


def modulate(x, norm, mod):
    shift, scale, _ = mod
    return (1+scale)*norm(x)+shift


def qkv(layer, norm, x, heads, pe):
    from extensions_built_in.diffusion_models.flux2.src.model import apply_rope
    q, k, v = layer(x).reshape(x.shape[0], x.shape[1], 3, heads, -1).permute(2, 0, 3, 1, 4)
    q, k = norm(q, k, v)
    q, k = apply_rope(q, k, pe)
    return q, k, v


def bank_on(bank, device):
    # Queries are consumed immediately; retain only frozen K/V for face replay.
    return (None, bank[1].to(device), bank[2].to(device))


def attend(q, *banks):
    return F.scaled_dot_product_attention(q, torch.cat([b[1] for b in banks], 2),
        torch.cat([b[2] for b in banks], 2)).transpose(1, 2).flatten(2)


def double_state(block, x, mods, pe, text=False):
    stem = 'txt' if text else 'img'
    attention = getattr(block, stem+'_attn')
    h = modulate(x, getattr(block, stem+'_norm1'), mods[0])
    return qkv(attention.qkv, attention.norm, h, block.num_heads, pe)


def double_update(block, x, mods, message, text=False):
    stem = 'txt' if text else 'img'
    x = x + mods[0][2]*getattr(block, stem+'_attn').proj(message)
    return x + mods[1][2]*getattr(block, stem+'_mlp')(modulate(x, getattr(block, stem+'_norm2'), mods[1]))


def single_state(block, x, mod, pe):
    from extensions_built_in.diffusion_models.flux2.src.model import apply_rope
    qkv_value, mlp = block.linear1(modulate(x, block.pre_norm, mod)).split(
        [3*block.hidden_size, block.mlp_hidden_dim*block.mlp_mult_factor], dim=-1)
    q, k, v = qkv_value.reshape(x.shape[0], x.shape[1], 3, block.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k = block.norm(q, k, v)
    q, k = apply_rope(q, k, pe)
    return (q, k, v), mlp


def single_update(block, x, mod, message, mlp):
    return x + mod[2]*block.linear2(torch.cat((message, block.mlp_act(mlp)), dim=-1))


def predict(model, tensors, noisy, sigma, negative=False):
    from extensions_built_in.diffusion_models.flux2.src.model import timestep_embedding
    from extensions_built_in.diffusion_models.flux2.src.sampling import batched_prc_img
    if noisy.shape[0] != 1:
        raise ValueError('FLUX2 currently requires microbatch 1')
    selected = tensors['target_face_mask'][0].gt(0).nonzero().flatten()
    context_indices = tensors['flux2_context_keep'].flatten().nonzero().flatten()
    if not selected.numel() or torch.isin(context_indices, selected).any():
        raise ValueError('FLUX2 needs nonempty, disjoint face/context ownership')
    packed, _ = batched_prc_img(noisy)
    context = tensors['flux2_context'].to(noisy.dtype)
    context_noise = tensors['flux2_context_noise'].to(noisy.dtype)
    context, _ = batched_prc_img((1-sigma)*context + sigma*context_noise)
    prefix = 'negative_' if negative else ''
    with torch.no_grad():
        vec = model.time_in(timestep_embedding(sigma.reshape(-1), 256).to(noisy.dtype))
        imods = model.double_stream_modulation_img(vec)
        tmods = model.double_stream_modulation_txt(vec)
        smod, _ = model.single_stream_modulation(vec)
        text = model.txt_in(tensors[prefix+'prompt_embeds'].to(noisy.dtype))
        ctx = model.img_in(context.index_select(1, context_indices))
        face = model.img_in(packed.index_select(1, selected))
        pe = model.pe_embedder(tensors['target_ids'])
        pe_f, pe_c = pe.index_select(2, selected), pe.index_select(2, context_indices)
        pe_t = model.pe_embedder(tensors[prefix+'text_ids'])
    identity, detail, bottleneck = model.reference_branch.memories(tensors)
    def run(fn, *args):
        if torch.is_grad_enabled() and model.gradient_checkpointing:
            # Exact frozen context offload avoids keeping 25 K/V banks on 16GB.
            args = (*args[:-2], bank_on(args[-2], 'cpu'), bank_on(args[-1], 'cpu'))
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)
    for i, block in enumerate(model.double_blocks):
        with torch.no_grad():
            tb = double_state(block, text, tmods, pe_t, True)
            cb = double_state(block, ctx, imods, pe_c)
            text = double_update(block, text, tmods, attend(tb[0], tb), True)
            ctx = double_update(block, ctx, imods, attend(cb[0], tb, cb))
        read = model.reference_branch.reads[i]
        def face_step(f, identity, detail, bottleneck, tb, cb, block=block, read=read):
            tb, cb = bank_on(tb, f.device), bank_on(cb, f.device)
            offsets = read.mod(bottleneck).to(f.dtype).unsqueeze(1).chunk(6, -1)
            mods = tuple(tuple(base[j]+offsets[k*3+j] for j in range(3)) for k, base in enumerate(imods))
            fb = double_state(block, f, mods, pe_f)
            f = double_update(block, f, mods, attend(fb[0], tb, cb, fb))
            return f + read(f, identity, detail)
        face = run(face_step, face, identity, detail, bottleneck, tb, cb)
    for i, block in enumerate(model.single_blocks, len(model.double_blocks)):
        with torch.no_grad():
            tb, tm = single_state(block, text, smod, pe_t)
            cb, cm = single_state(block, ctx, smod, pe_c)
            text = single_update(block, text, smod, attend(tb[0], tb), tm)
            ctx = single_update(block, ctx, smod, attend(cb[0], tb, cb), cm)
        read = model.reference_branch.reads[i]
        def face_step(f, identity, detail, bottleneck, tb, cb, block=block, read=read):
            tb, cb = bank_on(tb, f.device), bank_on(cb, f.device)
            offsets = read.mod(bottleneck).to(f.dtype).unsqueeze(1).chunk(3, -1)
            mod = tuple(base+offset for base, offset in zip(smod, offsets))
            fb, mlp = single_state(block, f, mod, pe_f)
            f = single_update(block, f, mod, attend(fb[0], tb, cb, fb), mlp)
            return f + read(f, identity, detail)
        face = run(face_step, face, identity, detail, bottleneck, tb, cb)
    output = model.final_layer(face, vec)
    return torch.zeros_like(packed).index_copy(1, selected, output).transpose(1, 2).reshape_as(noisy)
