"""One frozen component per process; cache only exact native tensors."""

import json

import torch

from ba_dit.config import digest
from ba_dit.data.cache import cache_path, cache_spec, write_cache
from ba_dit.data.manifest import read_manifest
from ba_dit.runtime import backend_module


def prepare(config, stage, split, limit=None):
    if split == "train" and limit is None:
        limit = config["data"]["train_limit"]
    rows = read_manifest(config["data"][f"{split}_manifest"], training=split == "train", limit=limit)
    pending = {}
    for row in rows:
        candidates = [row]
        if stage == "encoder" and split == "validation" and config["validation"]["guidance"] > 1:
            candidates.append({**row, "prompt": ""})
        for candidate in candidates:
            path = cache_path(config, candidate, stage)
            if not path.is_file():
                pending[path] = candidate
    print(json.dumps({"stage": stage, "split": split, "items": len(rows), "uncached_components": len(pending)}), flush=True)
    if not pending:
        return
    backend = backend_module(config)
    component = backend.load_encoder(config) if stage == "encoder" else backend.load_vae(config)
    encode = backend.encode_text if stage == "encoder" else backend.encode_images
    for index, (path, row) in enumerate(pending.items()):
        with torch.inference_mode():
            tensors, geometry = encode(component, config, row)
        metadata = {"key": digest(cache_spec(config, row, stage)), "spec": cache_spec(config, row, stage), **geometry}
        write_cache(path, tensors, metadata)
        print(json.dumps({"stage": stage, "completed": index + 1, "total": len(pending), "sample_id": row["sample_id"],
                          "shapes": {name: list(value.shape) for name, value in tensors.items()}}), flush=True)
