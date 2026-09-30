"""Small, exact branch-adapter checkpoints independent of backbone weights."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ba_dit.nn.reference_read_delta import ReferenceReadDelta


def _branch_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {
        f"{module_name}.{name}": parameter
        for module_name, module in model.named_modules()
        if isinstance(module, ReferenceReadDelta)
        for name, parameter in module.named_parameters()
    }


def save_branch(model: torch.nn.Module, path: Path, identity: dict[str, str]) -> None:
    """Write adapters atomically with the exact backbone/source identity."""
    parameters = _branch_parameters(model)
    if not parameters or not identity.get("backend") or not identity.get("model_revision"):
        raise ValueError("Branch modules, backend, and model revision are required")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    metadata = {"schema_version": "1", "identity": json.dumps(identity, sort_keys=True)}
    tensors = {name: parameter.detach().to("cpu", dtype=torch.float32).contiguous() for name, parameter in parameters.items()}
    try:
        save_file(tensors, temporary, metadata=metadata)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_branch(model: torch.nn.Module, path: Path, expected_identity: dict[str, str]) -> None:
    """Load only when adapter names, shapes, and base identity match exactly."""
    parameters = _branch_parameters(model)
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("schema_version") != "1" or json.loads(metadata.get("identity", "{}")) != expected_identity:
            raise ValueError("Branch checkpoint belongs to another backbone or source revision")
        if set(checkpoint.keys()) != set(parameters):
            raise ValueError("Branch site map or parameter names differ from checkpoint")
        tensors = {}
        for name, parameter in parameters.items():
            tensor = checkpoint.get_tensor(name)
            if tensor.shape != parameter.shape or tensor.dtype != torch.float32:
                raise ValueError(f"Branch parameter {name} has incompatible shape or dtype")
            tensors[name] = tensor
        with torch.no_grad():
            for name, parameter in parameters.items():
                tensor = tensors[name]
                parameter.copy_(tensor.to(parameter.device))
