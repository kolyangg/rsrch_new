"""Content-addressed tensors; load one pair at a time and retain nothing on CUDA."""

import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ba_dit.config import digest, revisions


CACHE_SCHEMA = 1


def cache_spec(config: dict, row: dict, stage: str) -> dict:
    from ba_dit.runtime import SOURCE_PINS

    backend = config["model"]["backend"]
    spec = {"schema": CACHE_SCHEMA, "backend": backend, "arch": config["model"]["arch"],
            "revisions": revisions(config), "source_commit": SOURCE_PINS[backend], "stage": stage, "dtype": "bfloat16"}
    spec['dtype'] = config['model'].get('conditioning_dtype', 'bfloat16')
    if stage == 'encoder' and 'encoder_device' in config['data']:
        spec['encoder_device'] = config['data']['encoder_device']
    if stage == "encoder":
        spec["prompt"] = row["prompt"]
    if stage == "vae" or backend == "qwen":
        spec.update(reference_hash=row["reference_hash"], reference_size=config["data"]["reference_size"],
                    reference_box=row["reference_box"], transform="native-reference-v1")
    if stage == "vae":
        spec.update(target_hash=row.get("target_hash"), target_box=row.get("target_box"),
                    target_size=config["data"]["target_size"], target_transform="face-centered-cover-v1",
                    posterior="native-flux-mean;qwen-target-fixed-sample-reference-mode", seed=config["training"]["seed"])
        if backend == "qwen" and row.get("target"):
            spec["training_normalization"] = "fp32_stats_then_bf16-v2"
            spec["posterior"] = "reference-mode;target-seed-xor-content-hash-v2"
    return spec


def cache_path(config: dict, row: dict, stage: str) -> Path:
    return Path(config["data"]["cache_dir"]) / stage / f"{digest(cache_spec(config, row, stage))}.safetensors"


def write_cache(path: Path, tensors: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = path.with_suffix(".tmp")
    save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, temporary,
              metadata={"record": json.dumps(metadata, sort_keys=True)})
    temporary.replace(path)


def load_pair(config: dict, row: dict, device="cpu", negative=False) -> tuple[dict, dict]:
    tensors, records = {}, {}
    for stage in ("encoder", "vae"):
        path = cache_path(config, row, stage)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {stage} cache for {row['sample_id']}: {path}. Run precompute first.")
        with safe_open(path, framework="pt", device="cpu") as cache:
            metadata = json.loads(cache.metadata()["record"])
        if metadata["key"] != digest(cache_spec(config, row, stage)):
            raise ValueError("Conditioning cache identity mismatch")
        tensors.update(load_file(path, device=str(device)))
        records[stage] = metadata
    if negative:
        path = cache_path(config, {**row, "prompt": ""}, "encoder")
        if not path.is_file():
            raise FileNotFoundError(f"Missing negative-prompt cache: {path}")
        tensors.update({f"negative_{key}": value for key, value in load_file(path, device=str(device)).items()})
    return tensors, records
