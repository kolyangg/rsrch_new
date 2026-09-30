"""Register reference-read adapters on the pinned native FLUX.2 transformer."""

from torch import nn

from ba_dit.nn.reference_read_delta import ReferenceReadDelta


SITES = {
    "flux2_klein_4b": {"double": (1, 2, 3, 4), "single": (3, 8, 13, 18)},
    "flux2_klein_9b": {"double": (2, 4, 6, 7), "single": (4, 10, 16, 22)},
}


def install_reference_branch(
    transformer: nn.Module,
    arch: str,
    rank: int = 16,
    alpha: float = 16,
    gamma: float = 0.1,
    query_chunk: int = 128,
) -> list[nn.Parameter]:
    if arch not in SITES:
        raise ValueError(f"Unknown FLUX backbone: {arch}")
    sites = SITES[arch]
    if max(sites["double"]) >= len(transformer.double_blocks) or max(sites["single"]) >= len(transformer.single_blocks):
        raise ValueError("Branch site map does not fit loaded transformer")
    parameters = []
    for kind, blocks in (("double", transformer.double_blocks), ("single", transformer.single_blocks)):
        for site in sites[kind]:
            block = blocks[site]
            if hasattr(block, "reference_branch"):
                raise ValueError(f"Branch already installed at {kind} block {site}")
            device = next(block.parameters()).device
            block.reference_branch = ReferenceReadDelta(
                width=transformer.hidden_size, heads=transformer.num_heads,
                rank=rank, alpha=alpha, gamma=gamma, query_chunk=query_chunk,
            ).to(device=device)
            parameters.extend(block.reference_branch.parameters())
    return parameters
