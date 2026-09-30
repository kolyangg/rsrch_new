"""Register the Qwen-Image-2.1 reference-read branch at explicit block sites."""

from torch import nn

from ba_dit.nn.reference_read_delta import ReferenceReadDelta


def install_reference_branch(
    transformer: nn.Module,
    sites: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 31),
    rank: int = 16,
    alpha: float = 16,
    gamma: float = 0.1,
    query_chunk: int = 128,
) -> list[nn.Parameter]:
    blocks = transformer.transformer_blocks
    if len(set(sites)) != len(sites) or any(site < 0 or site >= len(blocks) for site in sites):
        raise ValueError("Invalid Qwen branch site map")
    trainable = []
    for site in sites:
        attention = blocks[site].attn
        if hasattr(attention, "reference_branch"):
            raise ValueError(f"Qwen branch already installed at block {site}")
        width = attention.inner_dim
        device = next(attention.parameters()).device
        attention.reference_branch = ReferenceReadDelta(
            width=width, heads=attention.heads, rank=rank, alpha=alpha, gamma=gamma, query_chunk=query_chunk,
        ).to(device=device)
        trainable.extend(attention.reference_branch.parameters())
    return trainable
