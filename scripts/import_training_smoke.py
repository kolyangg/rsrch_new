#!/usr/bin/env python3
"""Build a bounded cross-view training smoke set from the old three-identity sample."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / "data/train_smoke")
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    metadata_files = list(source.glob("filtered_ids3_adj_sample_3ids.json"))
    if len(metadata_files) != 1:
        raise FileNotFoundError("Expected the old three-identity sample metadata")
    metadata = json.loads(metadata_files[0].read_text())
    validation = Path(__file__).resolve().parents[1] / "data/validation/manual_val_96.jsonl"
    val_ids = {json.loads(line)["identity_id"] for line in validation.read_text().splitlines()}
    if val_ids.intersection(metadata):
        raise ValueError("Training and validation identities overlap")
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for identity, views in sorted(metadata.items()):
        ids = sorted(views, key=lambda value: int(value))
        if len(ids) < 2:
            raise ValueError(f"Identity {identity} has fewer than two views")
        for index, target_id in enumerate(ids):
            reference_id = ids[(index + 1) % len(ids)]
            target, reference = source / identity / f"{target_id}.jpg", source / identity / f"{reference_id}.jpg"
            if not target.is_file() or not reference.is_file():
                raise FileNotFoundError(f"Missing image for {identity}")
            output_dir = destination / identity
            output_dir.mkdir(exist_ok=True)
            for image in (target, reference):
                output = output_dir / image.name
                if output.exists() and sha(output) != sha(image):
                    raise FileExistsError(f"Different existing image {output}")
                shutil.copy2(image, output)
            reference_meta, target_meta = views[reference_id], views[target_id]
            rows.append({
                "sample_id": f"{identity}_{reference_id}_to_{target_id}",
                "identity_id": identity,
                "reference_images": [f"train_smoke/{identity}/{reference_id}.jpg"],
                "reference_face_boxes": [reference_meta["new_face_crop"]],
                "target_image": f"train_smoke/{identity}/{target_id}.jpg",
                "target_face_box": target_meta["new_face_crop"],
                "prompt": target_meta["text"],
                "split": "train_smoke",
            })
    manifest = destination.parent / "train_pairs_smoke.jsonl"
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if manifest.exists() and manifest.read_text() != content:
        raise FileExistsError(f"Different existing manifest: {manifest}")
    manifest.write_text(content)
    print(f"Imported {len(metadata)} disjoint identities and {len(rows)} cross-view smoke pairs; SHA256 {sha(manifest)}")


if __name__ == "__main__":
    main()
