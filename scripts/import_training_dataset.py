#!/usr/bin/env python3
"""Convert legacy Large/BigCelebs or Cosmic metadata into deterministic paired JSONL."""

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from ba_dit.config import ROOT, digest
from ba_dit.data.geometry import face_mask
from ba_dit.data.manifest import assert_disjoint, file_hash, read_manifest


def records(metadata, kind):
    if kind == "large":
        for identity, views in sorted(metadata.items()):
            names = sorted(views, key=lambda value: (int(value) if value.isdigit() else 0, value))
            if len(names) < 2:
                raise ValueError(f"No distinct reference for {identity}")
            for index, target in enumerate(names):
                reference = names[(index + 1) % len(names)]
                target_meta, reference_meta = views[target], views[reference]
                yield dict(identity=str(identity), target=f"{identity}/{target}.jpg", reference=f"{identity}/{reference}.jpg",
                    prompt=target_meta["text"], target_box=target_meta["new_face_crop"], reference_box=reference_meta["new_face_crop"],
                    target_crop=target_meta.get("body_crop"), reference_crop=reference_meta.get("body_crop"))
    else:
        for target, record in sorted(metadata.items()):
            candidates = sorted(set(record.get("face_paths", [])) - {target})
            boxes = record.get("face_bboxes", {})
            candidates = [path for path in candidates if path in boxes or path.lstrip("/") in boxes]
            if not candidates:
                raise ValueError(f"No distinct reference with a face box for {target}")
            reference = candidates[0]
            identity = record.get("identity_id") or record.get("person_id") or record.get("id") or str(Path(reference).parent)
            prompt = ", ".join(str(record.get(key) or "").strip() for key in ("facial_caption", "pose_caption", "background_caption") if record.get(key))
            yield dict(identity=str(identity), target=target, reference=reference, prompt=prompt or "person img",
                target_box=record["face_crop_new"], reference_box=boxes.get(reference, boxes.get(reference.lstrip("/"))),
                target_crop=record.get("body_crop"), reference_crop=None)


def prepare_image(path, box, crop, kind, prepared):
    with Image.open(path) as source:
        image = source.convert("RGB")
        if crop is not None and image.size != (1024, 1024):
            if kind == "large":
                left, right, top, bottom = crop
                crop = [left, top, right, bottom]
            image = image.crop(tuple(map(int, crop)))
            if image.size != (1024, 1024):
                raise ValueError(f"Legacy body crop is not 1024x1024: {path}")
            prepared.mkdir(parents=True, exist_ok=True)
            output = prepared / f"{digest([file_hash(path), crop])}.png"
            if not output.exists():
                image.save(output)
            path = output
        face_mask(image, box)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("large", "cosmic"), required=True, help="BigCelebs uses large")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, help="Separate Cosmic reference archive root; defaults to images-root")
    parser.add_argument("--output", type=Path, default=ROOT / "data/train_pairs.jsonl")
    parser.add_argument("--identity-aliases", type=Path, help="JSON mapping dataset IDs to validation/canonical names")
    parser.add_argument("--exclude-identities", type=Path, help="JSON list of additional held-out dataset IDs")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-pairs", type=int, help="First N eligible pairs, for quick importer checks")
    selection.add_argument("--sample-pairs", type=int, help="N eligible pairs chosen by stable hash across the full release")
    args = parser.parse_args()
    if any(value is not None and value <= 0 for value in (args.max_pairs, args.sample_pairs)):
        parser.error("pair counts must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    aliases = json.loads(args.identity_aliases.read_text()) if args.identity_aliases else {}
    excluded = set(json.loads(args.exclude_identities.read_text())) if args.exclude_identities else set()
    validation = read_manifest(ROOT / "data/validation/manual_val_96.jsonl")
    heldout = {row["identity_id"] for row in validation}
    rows, skipped = [], 0
    duplicate_pairs = validation_hash_pairs = 0
    validation_hashes = {row["reference_hash"] for row in validation}
    metadata = json.loads(args.metadata.read_text())
    selected = records(metadata, args.format)
    if args.sample_pairs:
        candidates = list(selected)
        eligible = [record for record in candidates
                    if aliases.get(record["identity"], record["identity"]) not in heldout
                    and record["identity"] not in excluded]
        skipped = len(candidates) - len(eligible)
        selected = sorted(eligible, key=lambda record: digest([record["identity"], record["target"], record["reference"]]))
    for record in selected:
        identity = aliases.get(record["identity"], record["identity"])
        if identity in heldout or record["identity"] in excluded:
            skipped += 1
            continue
        paths = {}
        for role in ("reference", "target"):
            root = args.reference_root if role == "reference" and args.reference_root else args.images_root
            path = (root / record[role]).resolve()
            paths[role] = prepare_image(path, record[f"{role}_box"], record[f"{role}_crop"], args.format,
                                        args.output.parent / "datasets/prepared")
        reference_hash, target_hash = file_hash(paths["reference"]), file_hash(paths["target"])
        if reference_hash == target_hash:
            duplicate_pairs += 1
            continue
        if reference_hash in validation_hashes or target_hash in validation_hashes:
            validation_hash_pairs += 1
            continue
        rows.append(dict(sample_id=digest([record["identity"], record["reference"], record["target"]])[:20], identity_id=identity,
            reference_images=[os.path.relpath(paths["reference"], args.output.parent)], reference_face_boxes=[record["reference_box"]],
            target_image=os.path.relpath(paths["target"], args.output.parent), target_face_box=record["target_box"],
            prompt=record["prompt"], split="train"))
        if (args.max_pairs or args.sample_pairs) and len(rows) >= (args.max_pairs or args.sample_pairs):
            break
    temporary = args.output.with_suffix(".tmp.jsonl")
    try:
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        assert_disjoint(read_manifest(temporary, training=True), validation)
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    audit = {"pairs": len(rows), "excluded_pairs": skipped, "duplicate_content_pairs": duplicate_pairs,
             "validation_image_overlap_pairs": validation_hash_pairs, "metadata_sha256": file_hash(args.metadata),
             "manifest_sha256": file_hash(args.output), "identity_aliases": aliases, "excluded_identities": sorted(excluded),
             "format": args.format, "images_root": str(args.images_root.resolve()),
             "reference_root": str((args.reference_root or args.images_root).resolve()),
             "selection": {"method": "stable-hash" if args.sample_pairs else "first-eligible",
                           "requested_pairs": args.sample_pairs or args.max_pairs},
             "pairing": "deterministic next view (large) / first sorted eligible reference (cosmic)"}
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
