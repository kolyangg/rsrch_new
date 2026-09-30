#!/usr/bin/env python3
"""Package the two original metadata files for private transfer to a new machine."""

import argparse
import json
import tarfile
from pathlib import Path

from ba_dit.config import ROOT
from ba_dit.data.manifest import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-data", type=Path, required=True, help="Old project's dataset_full directory")
    parser.add_argument("--output", type=Path, default=ROOT / "data/bundles/training-metadata.tar.gz")
    args = parser.parse_args()
    presets = json.loads((ROOT / "locks/datasets.json").read_text())
    paths = [(args.legacy_data / preset["legacy_metadata"], preset) for preset in presets.values()]
    for path, preset in paths:
        if file_hash(path) != preset["metadata_sha256"]:
            raise ValueError(f"Metadata differs from the original release: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "x:gz") as archive:
        for path, preset in paths:
            archive.add(path, arcname=preset["metadata"], recursive=False)
    print(f"{args.output}\nSHA256 {file_hash(args.output)}")


if __name__ == "__main__":
    main()
