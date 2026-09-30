#!/usr/bin/env python3
"""Build a local transfer bundle; source photographs never enter the public repo."""

import argparse
import hashlib
import tarfile
from pathlib import Path

from ba_dit.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/bundles/validation_and_smoke.tar.gz")
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    paths = [ROOT / "data/validation"]
    if args.include_smoke:
        paths += [ROOT / "data/train_smoke", ROOT / "data/train_pairs_smoke.jsonl"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "x:gz") as bundle:
        for path in paths:
            bundle.add(path, arcname=str(path.relative_to(ROOT / "data")))
    with args.output.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    print(f"{args.output}\nSHA256 {checksum}")


if __name__ == "__main__":
    main()
