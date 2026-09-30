#!/usr/bin/env python3
"""Download/reuse either original training dataset and import a separate paired manifest."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ba_dit.config import ROOT
from ba_dit.data.manifest import file_hash
from download_dataset import download, extract

DATASETS = json.loads((ROOT / "locks/datasets.json").read_text())


def image_root(extracted, directory, kind):
    # Original tar files preserve different top-level folders; never flatten images.
    prefixes = ("", "dataset_full", "cosmic_large", "dataset_full/cosmic_large",
                "large_dataset_adj", "dataset_full/large_dataset_adj")
    matches = [extracted / prefix / directory for prefix in prefixes
               if (extracted / prefix / directory).is_dir()]
    if len(matches) != 1:
        raise ValueError(f"Expected one {directory} under {extracted}; use --images-root/--reference-root for another layout")
    return matches[0] if kind == "large" else matches[0].parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=tuple(DATASETS))
    parser.add_argument("--show", action="store_true", help="Show pinned sources without downloading")
    parser.add_argument("--url", help="Large archive URL/file ID; defaults to LARGE_DATASET_URL or the saved source")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/datasets")
    parser.add_argument("--download-dir", type=Path, default=ROOT / "data/downloads")
    parser.add_argument("--metadata", type=Path, help="Original JSON; otherwise use the private training-metadata bundle")
    parser.add_argument("--images-root", type=Path, help="Reuse extracted images instead of downloading")
    parser.add_argument("--reference-root", type=Path, help="Cosmic face-bank root when reusing separate image folders")
    parser.add_argument("--output", type=Path, help="Defaults to data/train_pairs_<dataset>.jsonl")
    parser.add_argument("--identity-aliases", type=Path)
    parser.add_argument("--exclude-identities", type=Path)
    parser.add_argument("--max-pairs", type=int, help="Explicitly limit a pilot import")
    parser.add_argument("--sample-pairs", type=int, help="Stable hash sample across identities for a disk-bounded training pilot")
    args = parser.parse_args()
    preset = DATASETS[args.dataset]
    if args.show:
        print(json.dumps(preset, indent=2))
        return
    if args.url and args.dataset != "large":
        parser.error("Cosmic has two pinned archives; --url is for Large only")
    if args.reference_root and not args.images_root:
        parser.error("--reference-root requires --images-root")
    if args.max_pairs and args.sample_pairs:
        parser.error("Choose --max-pairs or --sample-pairs")
    if any(value is not None and value <= 0 for value in (args.max_pairs, args.sample_pairs)):
        parser.error("pair counts must be positive")
    output = args.output or ROOT / "data" / f"train_pairs_{args.dataset}.jsonl"
    if output.exists():
        raise FileExistsError(f"Choose a new manifest path: {output}")
    metadata = args.metadata or args.data_dir / "metadata" / preset["metadata"]
    if not metadata.is_file():
        parser.error(f"Extract training-metadata.tar.gz into {args.data_dir}/metadata, or pass --metadata")
    if file_hash(metadata) != preset["metadata_sha256"]:
        raise ValueError(f"{metadata} is not the pinned {args.dataset} release; use import_training_dataset.py for a different release")
    provenance = []
    images, references = args.images_root, args.reference_root or args.images_root
    if images is None:
        roots = {}
        for source in preset["archives"]:
            url = args.url or os.getenv(source.get("url_env", "")) or source["url"]
            if not url:
                parser.error(f"Original {source['filename']} link is not recorded; pass --url or set {source['url_env']}, or reuse --images-root")
            destination = args.data_dir / args.dataset / source["name"]
            receipt = destination.with_suffix(".source.json")
            if destination.exists():
                if not receipt.is_file() or json.loads(receipt.read_text())["url"] != url:
                    raise ValueError(f"No matching source receipt for {destination}; use a new --data-dir or explicit --images-root")
                entry = json.loads(receipt.read_text())
            else:
                archive = download(url, args.download_dir)
                entry = {"name": source["name"], "url": url, "archive_sha256": file_hash(archive)}
                extract(archive, destination)
                receipt.write_text(json.dumps(entry, indent=2) + "\n")
            roots[source["name"]] = image_root(destination, source["image_directory"], preset["format"])
            provenance.append(entry)
        images = roots["images"] if args.dataset == "large" else roots["targets"]
        references = images if args.dataset == "large" else roots["references"]
    command = [sys.executable, str(ROOT / "scripts/import_training_dataset.py"),
               "--format", preset["format"], "--metadata", str(metadata),
               "--images-root", str(images), "--reference-root", str(references), "--output", str(output)]
    for name in ("identity_aliases", "exclude_identities", "max_pairs", "sample_pairs"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    subprocess.run(command, check=True)
    audit_path = output.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text())
    audit.update(dataset=args.dataset, archives=provenance, pinned_metadata_sha256=preset["metadata_sha256"])
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Ready: --train-manifest {output}")


if __name__ == "__main__":
    main()
