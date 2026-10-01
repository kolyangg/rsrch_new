"""Normalize existing PhotoMaker panel and cross-view training manifests."""

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: str | Path, training: bool = False, limit: int | None = None) -> list[dict]:
    path = Path(path).resolve()
    rows = []
    # Repeated identities/prompts share image files; hash each image once per read.
    image_hash = lru_cache(maxsize=None)(file_hash)
    for line in path.read_text().splitlines():
        raw = json.loads(line)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", raw["sample_id"]):
            raise ValueError("sample_id must contain only letters, digits, underscore or hyphen")
        references = raw.get("reference_images", [raw.get("reference_image")])
        boxes = raw.get("reference_face_boxes", [raw.get("reference_face_bbox")])
        if len(references) != 1 or not references[0] or len(boxes) != 1 or boxes[0] is None:
            raise ValueError("The initial branch supports one reference with an explicit face box")
        reference = (path.parent / references[0]).resolve()
        target = (path.parent / raw["target_image"]).resolve() if raw.get("target_image") else None
        if not reference.is_file() or (target is not None and not target.is_file()):
            raise FileNotFoundError(f"Missing image in {raw['sample_id']}")
        if training and (target is None or reference == target or raw.get("target_face_box") is None):
            raise ValueError("Training requires a distinct reference/target pair")
        row = {"sample_id": raw["sample_id"], "identity_id": raw["identity_id"], "prompt": raw["prompt"],
               "reference": str(reference), "reference_box": boxes[0], "reference_hash": image_hash(reference),
               "seed": raw.get("seed", 0)}
        if "split_policy" in raw:
            if raw["split_policy"] != "one_id_diagnostic":
                raise ValueError("Unknown explicit split policy")
            row["split_policy"] = raw["split_policy"]
        if raw.get("reference_sha256", row["reference_hash"]) != row["reference_hash"]:
            raise ValueError(f"Reference bytes changed in the fixed validation panel: {raw['sample_id']}")
        if target is not None:
            row.update(target=str(target), target_box=raw.get("target_face_box"), target_hash=image_hash(target))
            if training and row["reference_hash"] == row["target_hash"]:
                raise ValueError("Training reference and target have identical contents")
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Empty manifest or duplicate sample IDs")
    return rows


def assert_disjoint(train: list[dict], validation: list[dict]) -> None:
    rows = train + validation
    if any(row.get("split_policy") == "one_id_diagnostic" for row in rows):
        # AICODE-NOTE: The explicitly named one-ID wiring experiment reuses the
        # original validation reference seen in training. Never infer this exception.
        if not train or not validation or not all(row.get("split_policy") == "one_id_diagnostic" for row in rows):
            raise ValueError("Both complete manifests must opt into the one-ID diagnostic")
        if len({row["identity_id"] for row in rows}) != 1:
            raise ValueError("The one-ID diagnostic must contain exactly one shared identity")
        return
    identities = {row["identity_id"] for row in train} & {row["identity_id"] for row in validation}
    train_images = {row[key] for row in train for key in ("reference_hash", "target_hash")}
    if identities or train_images & {row["reference_hash"] for row in validation}:
        raise ValueError(f"Training and validation overlap: {sorted(identities)}")
