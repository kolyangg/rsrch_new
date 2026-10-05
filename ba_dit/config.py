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
    "validation": {"steps", "guidance", "limit"},
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
                            ("data", ("train_manifest", "validation_manifest", "cache_dir", "identity_supervision", "flux2_cache"))):
        for field in fields:
            if field not in result[section]:
                continue
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
    expected = {"schema_version", "name", *FIELDS}
    if set(config) != expected or config["schema_version"] != 2:
        raise ValueError(f"Expected schema 2 and fields {sorted(expected)}")
    for section, fields in FIELDS.items():
        optional = {"validation": {"batch_size"}, "branch": {"kind", "mask_feather_pixels", "reference_bank", "token_ownership"},
                    "data": {"conditioning", "encoder_device", "identity_supervision", "flux2_cache"},
                    "model": {"dtype", "conditioning_dtype", "compute_precision"}, "training": {"world_size", "microbatch_size", "identity_loss"}}
        allowed = fields | optional.get(section, set())
        if not fields <= set(config[section]) or set(config[section]) - allowed:
            raise ValueError(f"Invalid {section} fields: missing={fields-set(config[section])}, unknown={set(config[section])-allowed}")
    if config["model"]["backend"] not in {"flux", "qwen"}:
        raise ValueError("Unsupported backend")
    if config['data'].get('conditioning', 'cached') not in {'cached','online'}:
        raise ValueError('Unknown training conditioning mode')
    if config['data'].get('conditioning') == 'online' and config['model']['backend'] != 'flux':
        raise ValueError('Online conditioning is currently implemented for FLUX only')
    for field in ('dtype', 'conditioning_dtype'):
        if config['model'].get(field, 'bfloat16') not in {'bfloat16', 'float16', 'float32'}:
            raise ValueError(f'Unsupported {field}')
        if field in config['model'] and config['model']['backend'] != 'flux':
            raise ValueError('Explicit precision currently requires FLUX')
    if config['data'].get('encoder_device', 'cuda') not in {'cpu', 'cuda', 'cuda:1'}:
        raise ValueError('encoder_device must be cpu, cuda or cuda:1')
    world = config['training'].get('world_size', 1)
    batch = config['training'].get('microbatch_size', 1)
    if type(batch) is not int or batch < 1:
        raise ValueError('microbatch_size must be a positive integer')
    if batch > 1 and (world != 1 or config['model']['backend'] != 'flux' or
            config['branch'].get('kind') != 'masked_face_qkvo' or
            config['model'].get('dtype', 'bfloat16') != 'bfloat16' or
            config['model'].get('conditioning_dtype', 'bfloat16') != 'bfloat16'):
        raise ValueError('True microbatches currently require single-GPU BF16 masked FLUX')
    if type(world) is not int or world not in {1, 2}:
        raise ValueError('Supported world_size is 1 or 2')
    if world == 2 and (config['model']['backend'] != 'flux' or config['branch'].get('kind') != 'masked_face_qkvo'):
        raise ValueError('Two-rank training requires masked FLUX Q/K/V/O')
    if config['data'].get('encoder_device') == 'cuda:1' and world != 1:
        raise ValueError('A dedicated encoder GPU requires one training worker')
    if 'compute_precision' in config['model']:
        if (config['model']['compute_precision'] != 'amp_fp16_fp32_branch' or
                config['model']['arch'] != 'flux2_klein_4b' or
                config['model'].get('dtype') != 'float32' or
                config['model'].get('conditioning_dtype') != 'float32' or
                config['branch'].get('kind') != 'masked_face_qkvo' or
                config['training'].get('microbatch_size', 1) != 1 or
                (world == 2 and config['data'].get('conditioning') != 'cached')):
            raise ValueError('Mixed precision requires FP32-master masked FLUX4B; two workers require cached conditioning')
    arches = {"flux": {"flux2_klein_4b", "flux2_klein_9b"}, "qwen": {"qwen_image_2_1"}}
    if config["model"]["arch"] not in arches[config["model"]["backend"]]:
        raise ValueError("Architecture does not belong to this backend")
    if config["data"]["train_limit"] is not None and config["data"]["train_limit"] <= 0:
        raise ValueError("train_limit must be positive or null")
    h, w = config["data"]["target_size"]
    multiple = 32 if config["model"]["backend"] == "qwen" else 16
    if min(h, w, config["data"]["reference_size"]) <= 0 or h % multiple or w % multiple or config["data"]["reference_size"] % 32:
        raise ValueError("Invalid target/reference dimensions")
    for section, keys in (("branch", ("rank", "alpha", "query_chunk", "max_reference_keys")), ("lora", ("rank", "alpha")), ("training", ("steps", "grad_accum", "lr", "checkpoint_every", "validation_every")), ("validation", ("steps", "limit"))):
        if any(config[section][key] <= 0 for key in keys):
            raise ValueError(f"Nonpositive {section} setting")
    if not 0 < config["training"]["max_reserved_fraction"] <= 1 or config["branch"]["gamma"] < 0:
        raise ValueError("Invalid memory gate or branch scale")
    if config["validation"].get("batch_size", 1) <= 0:
        raise ValueError("Validation batch size must be positive")
    if config['branch'].get('kind', 'reference_delta') not in {'reference_delta', 'masked_face_qkvo', 'flux2_face'}:
        raise ValueError('Unknown branch kind')
    if config['branch'].get('kind') == 'masked_face_qkvo':
        if config['model']['backend'] != 'flux' or config['branch'].get('mask_feather_pixels', -1) < 0:
            raise ValueError('Masked Q/K/V/O requires FLUX Klein and an explicit nonnegative mask feather')
    if config['branch'].get('kind') == 'flux2_face':
        if (config['model']['backend'] != 'flux' or world != 1 or batch != 1 or
                config['validation'].get('batch_size', 1) != 1 or
                config['branch'].get('token_ownership') != 'binary_support' or
                not config['data'].get('flux2_cache') or config['data'].get('conditioning', 'cached') != 'cached'):
            raise ValueError('FLUX2 requires single-GPU cached conditioning and binary face ownership')
    bank = config['branch'].get('reference_bank', 'joint')
    ownership = config['branch'].get('token_ownership', 'soft')
    if bank not in {'joint', 'isolated_image'} or ownership not in {'soft', 'binary_support'}:
        raise ValueError('Unknown reference bank or token ownership policy')
    objective = config['training'].get('identity_loss', {})
    if objective and (set(objective) != {'weight', 'max_sigma', 'every'} or
            not 0 <= objective['weight'] <= 1 or not 0 < objective['max_sigma'] <= 1 or
            type(objective['every']) is not int or objective['every'] < 1):
        raise ValueError('identity_loss needs weight in [0,1], max_sigma in (0,1], integer every >= 1')
    if bank != 'joint' or ownership != 'soft' or objective:
        bf16 = (config['model'].get('dtype', 'bfloat16') == 'bfloat16' and
                config['model'].get('conditioning_dtype', 'bfloat16') == 'bfloat16')
        v100 = config['model'].get('compute_precision') == 'amp_fp16_fp32_branch'
        if config['branch'].get('kind') not in {'masked_face_qkvo', 'flux2_face'} or batch != 1 or not (bf16 or v100):
            raise ValueError('FLUX1 follow-ups require masked FLUX, microbatch 1 and BF16 or the explicit V100 precision policy')
    if objective.get('weight', 0) > 0 and not config['data'].get('identity_supervision'):
        raise ValueError('Identity supervision requires its prepared data path and the live frozen VAE')
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
    if config['branch'].get('kind') == 'masked_face_qkvo':
        implementation += ['nn/masked_face_attention.py', 'nn/masked_face_flow.py', 'nn/batched_face_attention.py']
    if config['branch'].get('reference_bank') == 'isolated_image':
        implementation.append('nn/isolated_reference.py')
    if 'compute_precision' in config['model']:
        implementation.append('precision.py')
    if config['branch'].get('kind') == 'flux2_face':
        implementation += ['nn/flux2_face.py', 'data/flux2_memory.py']
    code = hashlib.sha256(b"".join((ROOT / "ba_dit" / name).read_bytes() for name in implementation)).hexdigest()
    patch_name = "flux2_reference_branch_and_offload.patch" if config["model"]["backend"] == "flux" else "qwen21_local_pairs_comet.patch"
    identity = {"backend": config["model"]["backend"], "arch": config["model"]["arch"],
            "revisions": revisions(config), "source_commit": SOURCE_PINS[config["model"]["backend"]],
            "source_patch_sha256": hashlib.sha256((ROOT / "patches" / patch_name).read_bytes()).hexdigest(),
            "adapter_code_sha256": code,
            "branch": config["branch"], "lora": config["lora"]}
    if 'dtype' in config['model'] or 'conditioning_dtype' in config['model']:
        identity['precision'] = {k: config['model'].get(k, 'bfloat16') for k in ('dtype', 'conditioning_dtype')}
    if 'compute_precision' in config['model']:
        identity['precision']['compute_precision'] = config['model']['compute_precision']
    if config['branch'].get('kind') == 'flux2_face':
        from ba_dit.data.flux2_memory import memory_identity
        identity['flux2_memory'] = memory_identity(config)
    return identity
