"""Validated run configuration shared by preparation, training, and inference."""

import hashlib
import copy
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FIELDS = {
    "model": {"backend", "arch", "weights", "encoder", "vae", "revision_lock"},
    "data": {"train_manifest", "validation_manifest", "target_size", "reference_size", "cache_dir", "train_limit"},
    "branch": {"rank", "alpha", "gamma", "query_chunk", "max_reference_keys"},
    "lora": {"rank", "alpha"},
    "training": {"steps", "grad_accum", "lr", "warmup", "checkpoint_every", "validation_every", "seed", "gradient_checkpointing", "weight_decay", "gradient_clip", "max_reserved_fraction", "deterministic"},
    "validation": {"steps", "guidance", "limit", "batch_size"},
    "hardware": {"min_vram_gb"},
    "logging": {"comet_project", "enabled"},
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def portable_config(config):
    """Keep project-local paths relocatable when a checkpoint moves to a new host."""
    result = copy.deepcopy(config)
    root = Path(os.getenv("BA_ROOT", ROOT)).resolve()
    for section, fields in (("model", ("weights", "encoder", "vae", "revision_lock")),
                            ("data", ("train_manifest", "validation_manifest", "cache_dir"))):
        for field in fields:
            path = Path(result[section][field])
            if path.is_relative_to(root):
                result[section][field] = "${BA_ROOT}/" + str(path.relative_to(root))
    return result


def config_digest(config):
    return digest(portable_config(config))


def load_config(path: str | Path) -> dict:
    text = Path(path).read_text().replace("${BA_ROOT}", str(Path(os.getenv("BA_ROOT", ROOT)).resolve()))
    if "${" in text:
        raise ValueError("Unresolved configuration variable")
    config = yaml.safe_load(text)
    config["data"].setdefault("train_limit", None)
    config["training"].setdefault("deterministic", True)
    config["validation"].setdefault("batch_size", 1)
    expected = {"schema_version", "name", *FIELDS}
    if set(config) != expected or config["schema_version"] != 2:
        raise ValueError(f"Expected schema 2 and fields {sorted(expected)}")
    for section, fields in FIELDS.items():
        if set(config[section]) != fields:
            raise ValueError(f"Invalid {section} fields: missing={fields-set(config[section])}, unknown={set(config[section])-fields}")
    if config["model"]["backend"] not in {"flux", "qwen"}:
        raise ValueError("Unsupported backend")
    arches = {"flux": {"flux2_klein_4b", "flux2_klein_9b"}, "qwen": {"qwen_image_2_1"}}
    if config["model"]["arch"] not in arches[config["model"]["backend"]]:
        raise ValueError("Architecture does not belong to this backend")
    if config["data"]["train_limit"] is not None and config["data"]["train_limit"] <= 0:
        raise ValueError("train_limit must be positive or null")
    h, w = config["data"]["target_size"]
    multiple = 32 if config["model"]["backend"] == "qwen" else 16
    if min(h, w, config["data"]["reference_size"]) <= 0 or h % multiple or w % multiple or config["data"]["reference_size"] % 32:
        raise ValueError("Invalid target/reference dimensions")
    for section, keys in (("branch", ("rank", "alpha", "query_chunk", "max_reference_keys")), ("lora", ("rank", "alpha")), ("training", ("steps", "grad_accum", "lr", "checkpoint_every", "validation_every")), ("validation", ("steps", "limit", "batch_size"))):
        if any(config[section][key] <= 0 for key in keys):
            raise ValueError(f"Nonpositive {section} setting")
    if not 0 < config["training"]["max_reserved_fraction"] <= 1 or config["branch"]["gamma"] < 0:
        raise ValueError("Invalid memory gate or branch scale")
    return config


def revisions(config: dict) -> dict:
    model = config["model"]
    key = "qwen21" if model["backend"] == "qwen" else ("flux4b" if model["arch"].endswith("4b") else "flux9b")
    required = {key} if key == "qwen21" else {key, f"{key}_text", "flux_vae"}
    components = json.loads(Path(model["revision_lock"]).read_text())["components"]
    selected = {item["key"]: item["revision"] for item in components if item["key"] in required}
    if set(selected) != required:
        raise ValueError(f"Weight lock is missing {required-set(selected)}")
    return selected


def adapter_identity(config: dict) -> dict:
    from ba_dit.runtime import SOURCE_PINS

    implementation = ["adapters.py", "nn/reference_read_delta.py", "nn/sliced_native_lora.py", "backends/attention.py",
                      "backends/flux2_native.py", "backends/qwen21.py"]
    code = hashlib.sha256(b"".join((ROOT / "ba_dit" / name).read_bytes() for name in implementation)).hexdigest()
    patch_name = "flux2_reference_branch_and_offload.patch" if config["model"]["backend"] == "flux" else "qwen21_local_pairs_comet.patch"
    return {"backend": config["model"]["backend"], "arch": config["model"]["arch"],
            "revisions": revisions(config), "source_commit": SOURCE_PINS[config["model"]["backend"]],
            "source_patch_sha256": hashlib.sha256((ROOT / "patches" / patch_name).read_bytes()).hexdigest(),
            "adapter_code_sha256": code,
            "branch": config["branch"], "lora": config["lora"]}
