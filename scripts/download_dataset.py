#!/usr/bin/env python3
"""Download a Drive/HTTPS archive and safely extract it, as in the old Vast workflow."""

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path


def extract(archive, destination):
    if destination.exists():
        raise FileExistsError(f"Use a new extraction directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".extract-", dir=destination.parent))
    try:
        def check(name, size):
            if not (staging / name).resolve().is_relative_to(staging.resolve()):
                raise ValueError(f"Archive path escapes destination: {name}")
            return size

        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as source:
                required = sum(check(item.filename, item.file_size) for item in source.infolist())
                if required > shutil.disk_usage(staging).free:
                    raise RuntimeError("Insufficient disk space for the unpacked dataset")
                source.extractall(staging)
        else:
            with tarfile.open(archive) as source:
                members = source.getmembers()
                if any(not (item.isfile() or item.isdir()) for item in members):
                    raise ValueError("Dataset archives may contain only regular files and directories")
                required = sum(check(item.name, item.size) for item in members)
                if required > shutil.disk_usage(staging).free:
                    raise RuntimeError("Insufficient disk space for the unpacked dataset")
                source.extractall(staging, members=members, filter="data")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def download(url, download_dir):
    download_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(url.encode()).hexdigest()[:16]
    archive = download_dir / f"dataset-{name}.archive"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        if "drive.google.com" in url or "://" not in url:
            import gdown
            options = {"url": url, "fuzzy": True} if "://" in url else {"id": url}
            result = gdown.download(output=str(temporary), quiet=False, resume=True, **options)
            if not result:
                raise RuntimeError("Drive download failed; check the archive link and access")
        else:
            subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-", "--output", str(temporary), url], check=True)
        temporary.rename(archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="HTTPS URL, Google Drive share link, or Drive file ID")
    source.add_argument("--archive", type=Path, help="Use an archive already on this machine")
    parser.add_argument("--download-dir", type=Path, default=Path("data/downloads"))
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--sha256", help="Optional expected archive hash")
    args = parser.parse_args()
    archive = download(args.url, args.download_dir) if args.url else args.archive
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if args.sha256 and checksum != args.sha256.lower():
        raise ValueError(f"Archive hash differs from expected: {checksum}")
    extract(archive, args.destination)
    print(f"Extracted to {args.destination}; archive SHA256 {checksum}")


if __name__ == "__main__":
    main()
