"""Small, exact branch-adapter checkpoints independent of backbone weights."""

import json
import random
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ba_dit.nn.reference_read_delta import ReferenceReadDelta


def trainable_parameters(model):
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def training_code_digest(config):
    import hashlib
    from ba_dit.config import ROOT
    files = ["training.py", f"backends/{config['model']['backend']}_runtime.py", "data/cache.py", "data/geometry.py"]
    return hashlib.sha256(b"".join((ROOT / "ba_dit" / name).read_bytes() for name in files)).hexdigest()


def save_training(model, optimizer, scheduler, config, mode, run_dir, step, cursor, data_digest=None):
    from ba_dit.config import adapter_identity, config_digest, portable_config
    import yaml

    run_dir = Path(run_dir)
    destination = run_dir / f"checkpoint-{step:06d}"
    if destination.exists():
        raise FileExistsError(destination)
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=run_dir))
    parameters = trainable_parameters(model)
    try:
        save_file({name: value.detach().cpu().contiguous() for name, value in parameters.items()}, temporary / "adapters.safetensors")
        manifest = {"schema": 2, "identity": adapter_identity(config), "mode": mode, "step": step,
                    "config_sha256": config_digest(config), "data_sha256": data_digest,
                    "training_code_sha256": training_code_digest(config),
                    "parameters": {name: list(value.shape) for name, value in parameters.items()}}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (temporary / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (temporary / "resume_config.yaml").write_text(yaml.safe_dump(portable_config(config), sort_keys=False))
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "cursor": cursor,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "python_rng": random.getstate()}, temporary / "training_state.pt")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def load_adapters(model, checkpoint, config, mode):
    from ba_dit.config import adapter_identity
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if manifest["schema"] != 2 or manifest["identity"] != adapter_identity(config) or manifest["mode"] != mode:
        raise ValueError("Checkpoint backbone, revision, adapter geometry, or mode is incompatible")
    parameters = trainable_parameters(model)
    tensors = load_file(checkpoint / "adapters.safetensors")
    if set(parameters) != set(tensors):
        raise ValueError("Checkpoint adapter parameter inventory differs from the model")
    if any(tensors[name].shape != parameter.shape or tensors[name].dtype != torch.float32 for name, parameter in parameters.items()):
        raise ValueError("Checkpoint tensor shape/dtype mismatch")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(tensors[name].to(parameter.device))
    return manifest


def restore_training(optimizer, scheduler, checkpoint, config, data_digest=None):
    from ba_dit.config import config_digest

    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if manifest["config_sha256"] != config_digest(config):
        raise ValueError("Resume requires the exact saved configuration; use --init-adapter for a new curriculum")
    if manifest.get("data_sha256") != data_digest:
        raise ValueError("Training images, prompts, geometry or pairing order changed since checkpoint")
    if manifest["training_code_sha256"] != training_code_digest(config):
        raise ValueError("Training/cache implementation changed; use --init-adapter for the revised experiment")
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    return manifest["step"], state["cursor"]


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
