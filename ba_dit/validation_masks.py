"""Backbone-specific output face masks from native baseline validation images."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ba_dit.config import ROOT, load_config, revisions
from ba_dit.data.manifest import file_hash


def signature(config):
    return {"arch": config["model"]["arch"], "revisions": revisions(config),
            "target_size": config["data"]["target_size"], "reference_size": config["data"]["reference_size"],
            "steps": config["validation"]["steps"], "guidance": config["validation"]["guidance"],
            "panel_sha256": file_hash(config["data"]["validation_manifest"])}


def mask_directory(config):
    height, width = config["data"]["target_size"]
    geometry = f"{height}x{width}_ref{config['data']['reference_size']}_steps{config['validation']['steps']}_cfg{config['validation']['guidance']:g}"
    return ROOT / "data/validation/output_masks" / config["model"]["arch"] / geometry


def mask_images(size, box):
    mask = Image.new("L", size)
    if box is not None:
        ImageDraw.Draw(mask).rectangle((box[0], box[1], box[2]-1, box[3]-1), fill=255)
    pixels = np.asarray(mask) > 0
    height, width = pixels.shape
    tokens = pixels.reshape(height//16, 16, width//16, 16).any(axis=(1, 3))
    return mask, Image.fromarray(tokens.astype(np.uint8) * 255)


def ensure_assets(directory, row, size):
    # Frozen boxes can restore omitted PNGs on a new host without rerunning detection.
    images = None
    for index, field in enumerate(("pixel_mask", "token_mask")):
        path = directory / row[field]
        if not path.exists():
            images = images or mask_images(size, row["face_bbox"])
            images[index].save(path)
        if file_hash(path) != row[field + "_sha256"]:
            raise ValueError(f"Output mask changed: {path}")


def load_masks(config, sample_ids):
    directory = mask_directory(config)
    manifest = directory / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Run native validation for this profile to prepare its output masks: {manifest}")
    record = json.loads(manifest.read_text())
    if record["signature"] != signature(config):
        raise ValueError("Output masks belong to another backbone, revision, geometry, or validation protocol")
    if missing := set(sample_ids) - set(record["samples"]):
        raise ValueError(f"Missing native baseline masks for {sorted(missing)}; run native validation for this profile first")
    for sample_id in sample_ids:
        ensure_assets(directory, record["samples"][sample_id], tuple(reversed(config["data"]["target_size"])))
    return record


def prepare(directory, overrides=None):
    from ba_dit.metrics import LegacyFaces

    directory = Path(directory)
    config = load_config(directory / "resolved_config.yaml")
    report = json.loads((directory / "validation.json").read_text())
    if report.get("checkpoint") is not None:
        raise ValueError("Freeze output masks from the native/zero-initialized baseline before scoring trained adapters")
    destination = mask_directory(config)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "manifest.json"
    record = json.loads(manifest.read_text()) if manifest.exists() else {"schema": 1, "signature": signature(config),
        "selection": "largest detected face; >=.95 area tie is ambiguous; no identity-score selection",
        "usage": "validation scoring only", "samples": {}}
    if record["signature"] != signature(config):
        raise ValueError("Existing baseline masks have another validation signature")
    manual = json.loads(Path(overrides).read_text()) if overrides else {}
    if unknown := set(manual) - {row["sample_id"] for row in report["samples"]}:
        raise ValueError(f"Manual boxes refer to samples absent from this validation: {sorted(unknown)}")
    pending = [row for row in report["samples"] if row["sample_id"] not in record["samples"] or row["sample_id"] in manual]
    detector = LegacyFaces() if pending else None
    for sample in pending:
        image_path = directory / sample["image"]
        image = Image.open(image_path).convert("RGB")
        faces = detector(image)
        boxes = [box for box, _ in faces]
        ranked = sorted(enumerate(boxes), key=lambda value: (-(value[1][2]-value[1][0])*(value[1][3]-value[1][1]), value[0]))
        box, status = None, "no_face"
        if sample["sample_id"] in manual:
            box, status = manual[sample["sample_id"]], "manual"
        elif ranked:
            area = lambda bbox: (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
            if len(ranked) > 1 and area(ranked[1][1]) >= .95 * area(ranked[0][1]):
                status = "ambiguous"
            else:
                box, status = ranked[0][1], "largest_face"
        if box is not None:
            box = [max(0, math.floor(box[0])), max(0, math.floor(box[1])), min(image.width, math.ceil(box[2])), min(image.height, math.ceil(box[3]))]
            if not (box[0] < box[2] and box[1] < box[3]):
                raise ValueError(f"Invalid output face box: {box}")
        mask, tokens = mask_images(image.size, box)
        pixel_path = destination / f"{sample['sample_id']}.png"
        token_path = destination / f"{sample['sample_id']}.tokens.png"
        mask.save(pixel_path)
        tokens.save(token_path)
        record["samples"][sample["sample_id"]] = dict(identity_id=sample["identity_id"], prompt=sample["prompt"], seed=sample["seed"],
            face_bbox=box, status=status, detected_face_count=len(faces), detected_boxes=boxes,
            baseline_image_sha256=file_hash(image_path), pixel_mask=pixel_path.name, token_mask=token_path.name,
            pixel_mask_sha256=file_hash(pixel_path), token_mask_sha256=file_hash(token_path))
    for sample in report["samples"]:
        ensure_assets(destination, record["samples"][sample["sample_id"]], tuple(reversed(config["data"]["target_size"])))
    record["coverage"] = {"prepared": len(record["samples"]),
        "usable": sum(row["face_bbox"] is not None for row in record["samples"].values()),
        "needs_review": [key for key, row in record["samples"].items() if row["status"] in {"no_face", "ambiguous"}]}
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest)
    print(json.dumps({"output_masks": str(destination), **record["coverage"]}), flush=True)
    return destination
