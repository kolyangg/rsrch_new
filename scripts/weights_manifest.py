#!/usr/bin/env python3
"""Audit selected HF files without downloading weights; download only from a saved lock.

Requires huggingface_hub. Authentication uses HF_TOKEN or the normal HF login cache.
No architecture code is installed, no training is launched, and no files are deleted.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPECS: dict[str, dict[str, Any]] = {
    "qwen21": {
        "repo": "Qwen/Qwen-Image-2.1",
        "patterns": ["model_index.json", "LICENSE*", "README.md", "processor/*", "scheduler/*.json",
                     "text_encoder/*.json", "text_encoder/*.safetensors", "transformer/*.json",
                     "transformer/*.safetensors", "vae/*.json", "vae/*.safetensors"],
        "required_groups": [["model_index.json"], ["processor/*"], ["text_encoder/*.safetensors"],
                            ["transformer/*.safetensors"], ["vae/*.safetensors"]],
    },
    "flux4b": {
        "repo": "black-forest-labs/FLUX.2-klein-base-4B",
        "patterns": ["flux-2-klein-base-4b.safetensors", "LICENSE*", "README.md"],
        "required_groups": [["flux-2-klein-base-4b.safetensors"]],
    },
    "flux9b": {
        "repo": "black-forest-labs/FLUX.2-klein-base-9B",
        "patterns": ["flux-2-klein-base-9b.safetensors", "LICENSE*", "README.md"],
        "required_groups": [["flux-2-klein-base-9b.safetensors"]],
    },
    "flux4b_text": {
        "repo": "Qwen/Qwen3-4B",
        "patterns": ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*", "README.md"],
        "required_groups": [["*.safetensors"], ["config.json"], ["tokenizer.json", "tokenizer.model"]],
    },
    "flux9b_text": {
        "repo": "Qwen/Qwen3-8B",
        "patterns": ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*", "README.md"],
        "required_groups": [["*.safetensors"], ["config.json"], ["tokenizer.json", "tokenizer.model"]],
    },
    "flux_vae": {
        "repo": "ai-toolkit/flux2_vae",
        "patterns": ["ae.safetensors", "LICENSE*", "README.md"],
        "required_groups": [["ae.safetensors"]],
    },
}
SETS = {
    "flux48": ["flux4b", "flux4b_text", "flux_vae"],
    "flux80": ["flux9b", "flux9b_text", "flux_vae"],
    "qwen": ["qwen21"],
    "small": ["flux4b", "flux4b_text", "flux_vae", "qwen21"],
    "all": ["flux4b", "flux4b_text", "flux_vae", "qwen21", "flux9b", "flux9b_text"],
}


def matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


def bytes_from_sibling(sibling: Any) -> int:
    size = getattr(sibling, "size", None)
    if size is None:
        lfs = getattr(sibling, "lfs", None)
        size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
    if size is None or int(size) < 0:
        raise ValueError(f"No trustworthy size metadata for {sibling.rfilename}; refusing a zero estimate")
    return int(size)


def select_files(spec: dict[str, Any], siblings: list[Any]) -> list[dict[str, Any]]:
    result = [{"path": s.rfilename, "bytes": bytes_from_sibling(s)} for s in siblings
              if matches(s.rfilename, spec["patterns"])]
    names = [f["path"] for f in result]
    for group in spec["required_groups"]:
        if not any(matches(n, group) for n in names):
            raise ValueError(f"Missing required group {group} in {spec['repo']}")
    return sorted(result, key=lambda f: f["path"])


def audit(set_name: str) -> dict[str, Any]:
    from huggingface_hub import HfApi
    api = HfApi()
    components = []
    for key in SETS[set_name]:
        spec = SPECS[key]
        info = api.model_info(spec["repo"], revision="main", files_metadata=True)
        if not info.sha:
            raise ValueError(f"Could not resolve a commit for {spec['repo']}")
        files = select_files(spec, info.siblings or [])
        components.append({"key": key, "repo": spec["repo"], "revision": info.sha,
                           "files": files, "bytes": sum(f["bytes"] for f in files)})
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "set": set_name, "components": components,
            "bytes": sum(c["bytes"] for c in components)}


def validate_lock(lock: dict[str, Any]) -> None:
    if lock.get("schema_version") != 1:
        raise ValueError("Unknown lock schema")
    seen = set()
    for c in lock["components"]:
        key = c["key"]
        if key in seen or key not in SPECS or c["repo"] != SPECS[key]["repo"]:
            raise ValueError(f"Invalid or duplicate component {key}")
        seen.add(key)
        if len(c["revision"]) != 40 or any(ch not in "0123456789abcdef" for ch in c["revision"].lower()):
            raise ValueError("A full HF commit SHA is required")
        for f in c["files"]:
            p = Path(f["path"])
            if p.is_absolute() or ".." in p.parts or not matches(f["path"], SPECS[key]["patterns"]):
                raise ValueError(f"Unexpected file path {p}")
            if not isinstance(f["bytes"], int) or f["bytes"] < 0:
                raise ValueError("Invalid file size")
        names = [f["path"] for f in c["files"]]
        for group in SPECS[key]["required_groups"]:
            if not any(matches(n, group) for n in names):
                raise ValueError(f"Incomplete component {key}")
        if c["bytes"] != sum(f["bytes"] for f in c["files"]):
            raise ValueError("Component total is inconsistent")
    if lock["bytes"] != sum(c["bytes"] for c in lock["components"]):
        raise ValueError("Lock total is inconsistent")


def report(lock: dict[str, Any]) -> None:
    for c in lock["components"]:
        print(f"{c['key']:14s} {c['bytes']/1e9:9.3f} GB  {c['bytes']/2**30:9.3f} GiB  "
              f"{len(c['files']):3d} files  {c['revision']}")
    print(f"Selected-file total: {lock['bytes']/1e9:.3f} GB / {lock['bytes']/2**30:.3f} GiB")
    print("This is selected source-file size, not missing-download size, peak disk or GPU memory.")
    print("Budget separately for environments, data, conditioning caches, checkpoints and temporary space.")


def download(lock: dict[str, Any], weights_dir: Path, cache_dir: str | None) -> None:
    from huggingface_hub import snapshot_download
    weights_dir.mkdir(parents=True, exist_ok=True)
    # Do not overwrite pre-existing model directories or symlinks pointing elsewhere.
    for c in lock["components"]:
        destination = weights_dir / c["key"]
        expected_suffix = Path("snapshots") / c["revision"]
        if os.path.lexists(destination):
            if not destination.is_symlink() or not str(destination.resolve()).endswith(str(expected_suffix)):
                raise FileExistsError(f"Refusing to replace {destination}")
    for c in lock["components"]:
        snapshot = Path(snapshot_download(repo_id=c["repo"], revision=c["revision"],
                        allow_patterns=[f["path"] for f in c["files"]], cache_dir=cache_dir))
        for f in c["files"]:
            path = snapshot / f["path"]
            if not path.is_file() or path.stat().st_size != f["bytes"]:
                raise RuntimeError(f"Missing file or size mismatch: {path}")
        destination = weights_dir / c["key"]
        if os.path.lexists(destination):
            if not destination.is_symlink() or destination.resolve() != snapshot.resolve():
                raise FileExistsError(f"Refusing to replace {destination}")
        else:
            destination.symlink_to(snapshot.resolve(), target_is_directory=True)
        print(f"Verified files; local alias: {destination} -> {snapshot}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "download"])
    parser.add_argument("--set", choices=SETS, default="small")
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--cache-dir", default=os.getenv("HF_HUB_CACHE"))
    args = parser.parse_args()
    if args.action == "audit":
        if args.lock.exists():
            raise FileExistsError("Use a new lock filename; existing locks are immutable")
        lock = audit(args.set)
        validate_lock(lock)
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        with args.lock.open("x", encoding="utf-8") as handle:
            json.dump(lock, handle, indent=2)
            handle.write("\n")
        report(lock)
    else:
        if args.weights_dir is None:
            parser.error("download requires --weights-dir")
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        validate_lock(lock)
        report(lock)
        download(lock, args.weights_dir, args.cache_dir)


if __name__ == "__main__":
    main()
