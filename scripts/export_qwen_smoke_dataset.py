#!/usr/bin/env python3
"""Embed the private paired smoke images in a local HF Dataset for Qwen's trainer."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/train_pairs_smoke.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/qwen_smoke_dataset")
    args = parser.parse_args()
    from datasets import Dataset, Features, Image, Value

    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for line in args.manifest.read_text().splitlines():
        record = json.loads(line)
        if len(record["reference_images"]) != 1:
            raise ValueError("Native paired Qwen smoke accepts one reference per target")
        target = ROOT / "data" / record["target_image"]
        reference = ROOT / "data" / record["reference_images"][0]
        if target.resolve() == reference.resolve():
            raise ValueError("A target cannot be its own reference")
        rows.append({
            "image": {"path": None, "bytes": target.read_bytes()},
            "cond_image": {"path": None, "bytes": reference.read_bytes()},
            "caption": record["prompt"],
        })
    if not rows:
        raise ValueError("Empty paired dataset")
    dataset = Dataset.from_list(rows, features=Features({"image": Image(), "cond_image": Image(), "caption": Value("string")}))
    dataset.save_to_disk(str(args.output))
    print(f"Saved {len(dataset)} local cross-view pairs to {args.output}")


if __name__ == "__main__":
    main()
