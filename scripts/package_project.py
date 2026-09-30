#!/usr/bin/env python3
"""Package the current working source, including uncommitted code, for a new host."""

import argparse
import hashlib
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=ROOT / "data/bundles/rsrch_new-code.tar.gz")
args = parser.parse_args()
files = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT).decode().split("\0")
args.output.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(args.output, "x:gz") as archive:
    for name in sorted(set(files) - {""}):
        path = ROOT / name
        if path.is_file() or path.is_symlink():
            archive.add(path, arcname=f"rsrch_new/{name}", recursive=False)
with args.output.open("rb") as stream:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    checksum = digest.hexdigest()
print(f"{args.output}\nSHA256 {checksum}")
