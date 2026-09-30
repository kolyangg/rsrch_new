#!/usr/bin/env python3
"""Export cross-view pairs in the native Toolkit target/control folder contract."""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/train_pairs_smoke.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/pairs_export")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    targets, references = args.output / "target", args.output / "reference"
    targets.mkdir(parents=True)
    references.mkdir()
    mapping = []
    for line in args.manifest.read_text().splitlines():
        row = json.loads(line)
        if len(row["reference_images"]) != 1:
            raise ValueError("Toolkit upstream smoke accepts one reference per pair")
        stem = row["sample_id"]
        if not stem or "/" in stem or ".." in stem:
            raise ValueError(f"Invalid pair basename: {stem}")
        target, reference = ROOT / "data" / row["target_image"], ROOT / "data" / row["reference_images"][0]
        if not target.is_file() or not reference.is_file() or target.resolve() == reference.resolve():
            raise ValueError(f"Invalid pair: {stem}")
        target_link = targets / f"{stem}{target.suffix}"
        reference_link = references / f"{stem}{reference.suffix}"
        target_link.symlink_to(os.path.relpath(target, targets))
        reference_link.symlink_to(os.path.relpath(reference, references))
        (targets / f"{stem}.txt").write_text(row["prompt"] + "\n")
        mapping.append({"sample_id": stem, "target": str(target), "reference": str(reference)})
    (args.output / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    print(f"Exported {len(mapping)} cross-view pairs to {args.output}")


if __name__ == "__main__":
    main()
