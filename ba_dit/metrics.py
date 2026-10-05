"""Legacy PhotoMaker identity/CLIP definitions, applied only after image generation."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ba_dit.config import ROOT, load_config
from ba_dit.data.manifest import file_hash, read_manifest
from ba_dit.logging import connect


def cosine(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def bbox_iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union > 0 else 0.0


class LegacyFaces:
    def __init__(self, recognition=True):
        import os
        import onnxruntime
        from unittest.mock import patch
        from insightface.app import FaceAnalysis
        from insightface.model_zoo.model_zoo import ModelRouter
        modules = ["detection", "recognition"] if recognition else ["detection"]
        if os.getenv('BA_ORT_THREADS'):
            options=onnxruntime.SessionOptions()
            options.intra_op_num_threads=int(os.environ['BA_ORT_THREADS'])
            options.inter_op_num_threads=1
            original=ModelRouter.get_model
            with patch.object(ModelRouter,'get_model',lambda router,**kw:original(router,sess_options=options,**kw)):
                self.detector=FaceAnalysis(name='buffalo_l',providers=['CPUExecutionProvider'],allowed_modules=modules)
        else:
            self.detector = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=modules)
        self.detector.prepare(ctx_id=-1, det_size=(640, 640))

    def detect(self, image):
        pixels = np.asarray(image.convert("RGB"))[:, :, ::-1]
        # Same progressively smaller detector sizes as rsrch_apr_test/id_utils.py.
        for size in [None, *range(640, 256, -64), 256]:
            if size is not None:
                self.detector.det_model.input_size = (size, size)
            faces = self.detector.get(pixels)
            if faces:
                return faces
        return []

    def __call__(self, image):
        bounds = [image.width, image.height] * 2
        return [(np.clip(face.bbox.astype(np.int32), 0, bounds).tolist(), face.embedding) for face in self.detect(image)]


@torch.no_grad()
def evaluate(directory, ownership_boxes=None, with_clip=True, no_comet=False, log_dir=None, step=0, device="cpu"):
    directory = Path(directory)
    config = load_config(directory / "resolved_config.yaml")
    if no_comet:
        config["logging"]["enabled"] = False
    report = json.loads((directory / "validation.json").read_text())
    panel = {row["sample_id"]: row for row in read_manifest(config["data"]["validation_manifest"])}
    identities = {}
    hashes = {}
    one_id = all(row.get("split_policy") == "one_id_diagnostic" for row in panel.values())
    for name in ("id_embeds_manual_val", "id_embeds_manual_val_subject_v2"):
        path = (Path(config["data"]["validation_manifest"]).parent / "id_embeds_one_id.pth"
                if one_id else ROOT / "data/validation" / f"{name}.pth")
        if not path.exists():
            raise FileNotFoundError(f"Import the original validation embeddings: {path}")
        identities[name] = torch.load(path, map_location="cpu", weights_only=True)
        hashes[path.stem] = file_hash(path)
    owned = json.loads(Path(ownership_boxes).read_text()) if ownership_boxes else None
    mask_manifest = None
    if owned is None:
        from ba_dit.validation_masks import load_masks, mask_directory
        masks = load_masks(config, [row["sample_id"] for row in report["samples"]])
        mask_manifest = mask_directory(config) / "manifest.json"
        owned = {key: Image.open(mask_directory(config) / row["pixel_mask"]).getbbox() for key, row in masks["samples"].items()}
    detector = LegacyFaces()
    if with_clip:
        import clip
        clip_model, preprocess = clip.load("ViT-L/14@336px", device="cpu")
        clip_model.float().to(device).eval()  # Preserve CPU FP32 weights/scoring on GPU.
    import time
    from ba_dit.progress import stage_progress
    started = time.monotonic()
    rows = []
    for sample in report["samples"]:
        identity, sample_id = sample["identity_id"], sample["sample_id"]
        image = Image.open(directory / sample["image"]).convert("RGB")
        faces = detector(image)
        for lookup in identities.values():
            if identity not in lookup:
                references = detector(Image.open(panel[sample_id]["reference"]))
                if not references:
                    raise ValueError(f"No reference face for {identity}")
                lookup[identity] = references[0][1]
        values = {"sample_id": sample_id, "identity_id": identity,
            "id_sim_legacy_best": max((cosine(embedding, identities["id_embeds_manual_val"][identity]) for _, embedding in faces), default=0.0),
            "id_sim_face_count": len(faces), "id_sim_no_face": float(not faces)}
        if owned is not None:
            # Frozen baseline boxes use this backbone's output coordinate system.
            # Historical PhotoMaker output boxes must not be reused for a new backbone.
            box = owned[sample_id]
            ranked = sorted(((bbox_iou(bbox, box), index, embedding) for index, (bbox, embedding) in enumerate(faces)), key=lambda item: (-item[0], item[1])) if box else []
            iou = ranked[0][0] if ranked else 0.0
            values.update(id_sim=cosine(ranked[0][2], identities["id_embeds_manual_val_subject_v2"][identity]) if iou >= 0.05 else 0.0,
                          id_sim_mask_iou=iou, id_sim_unowned=float(iou < 0.05), output_mask_missing=float(box is None),
                          id_sim_ambiguous=float(len(ranked) > 1 and ranked[1][0] >= 0.05 and abs(iou - ranked[1][0]) <= 0.02))
        if with_clip:
            _, logits = clip_model(preprocess(image).unsqueeze(0).to(device), clip.tokenize([sample["prompt"]], truncate=True).to(device))
            values["text_sim"] = float(logits.mean())
        rows.append(values)
        stage_progress('validation/ID+CLIP',len(rows),len(report['samples']),started)
        print(json.dumps(values), flush=True)
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key not in {"sample_id", "identity_id"}}
    with (directory / "quality_per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "quality_summary.json").write_text(json.dumps({"metrics": summary, "identity_embedding_sha256": hashes,
        "ownership_boxes_sha256": file_hash(ownership_boxes) if ownership_boxes else None,
        "output_mask_set_sha256": file_hash(mask_manifest) if mask_manifest else None,
        "clip_device": device, "clip_dtype": "float32",
        "definitions": "rsrch_apr_test IDSimBest, IDSimMaskMatched (IoU .05/margin .02), CLIP ViT-L/14@336px logits"}, indent=2) + "\n")
    experiment = connect(config, log_dir or directory)
    if experiment:
        experiment.log_metrics({f"validation/{key}": value for key, value in summary.items()}, step=step)
        experiment.log_asset(str(directory / "quality_per_image.csv"))
        experiment.log_asset(str(directory / "quality_summary.json"))
        experiment.end()
    return summary
