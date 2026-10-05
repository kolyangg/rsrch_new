"""The previous project's seven face-quality curves, with the same crop policy."""

import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from ba_dit.config import load_config
from ba_dit.logging import connect
from ba_dit.metrics import LegacyFaces


MODELS = {"topiq_nr-face": "topiq_face", "topiq_nr": "topiq", "musiq": "musiq", "maniqa-pipal": "maniqa"}
DEFAULT_THREADS = min(8, os.cpu_count() or 1)


def square_face_crop(image, bbox):
    # Same 25% padding per side, clipping, rounding and Lanczos resize as rsrch_apr_test.
    x0, y0, x1, y1 = bbox
    side = min(max(2, round(max(x1-x0, y1-y0) * 1.5)), *image.size)
    left = min(max(round((x0+x1-side)/2), 0), image.width-side)
    top = min(max(round((y0+y1-side)/2), 0), image.height-side)
    box = [left, top, left+side, top+side]
    crop = image.crop(box)
    if crop.size != (512, 512):
        crop = crop.resize((512, 512), Image.Resampling.LANCZOS)
    return crop, box


@torch.inference_mode()
def evaluate(directory, no_comet=False, log_dir=None, step=0, threads=DEFAULT_THREADS, device="cpu"):
    import pyiqa

    if importlib.metadata.version("pyiqa") != "0.1.15":
        raise ValueError("Comparable face-quality scoring requires PyIQA 0.1.15")
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    np.random.seed(0)
    directory = Path(directory)
    config = load_config(directory / "resolved_config.yaml")
    if no_comet:
        config["logging"]["enabled"] = False
    report = json.loads((directory / "validation.json").read_text())
    detector = LegacyFaces(recognition=False)
    models = {name: pyiqa.create_metric(name, device=device).eval() for name in MODELS}
    rows, crops = [], []
    for sample in report["samples"]:
        image = Image.open(directory / sample["image"]).convert("RGB")
        faces = detector.detect(image)
        row = {"step": step, "sample_id": sample["sample_id"], "identity_id": sample["identity_id"],
               "image": sample["image"], "face_detected": int(bool(faces)), "face_count": len(faces),
               "face_bbox": None, "crop_bbox": None, "det_score": None}
        if faces:
            face = max(faces, key=lambda value: max(0, float(value.bbox[2]-value.bbox[0])) * max(0, float(value.bbox[3]-value.bbox[1])))
            box = [min(max(float(value), 0), (image.width, image.height)[index % 2]) for index, value in enumerate(face.bbox)]
            crop, crop_box = square_face_crop(image, box)
            row.update(face_bbox=box, crop_bbox=crop_box, det_score=float(face.det_score), _crop_index=len(crops))
            crops.append(pil_to_tensor(crop).float().div_(255))
        rows.append(row)
    model_seconds = {}
    for name, field in MODELS.items():
        started = perf_counter()
        scores = []
        batch_size = 1 if name == "topiq_nr-face" else 8
        for start in range(0, len(crops), batch_size):
            batch = torch.stack(crops[start:start+batch_size]).to(device)
            try:
                values = models[name](batch).detach().float().reshape(-1).tolist()
            except Exception as error:
                if name != "topiq_nr-face":
                    raise
                print(f"TOPIQ face alignment failed for crop {start}: {error}", flush=True)
                values = [None]
            if len(values) != len(batch):
                raise ValueError(f"{name} returned the wrong score count")
            scores.extend(value if value is not None and math.isfinite(value) else None for value in values)
            from ba_dit.progress import stage_progress
            stage_progress('validation/'+name,len(scores),len(crops),started)
            if len(scores) % 8 == 0 or len(scores) == len(crops):
                print(json.dumps({"face_quality_model": name, "processed": len(scores), "total_crops": len(crops),
                                  "seconds": round(perf_counter()-started, 2)}), flush=True)
        model_seconds[name] = perf_counter()-started
        for row in rows:
            row[field] = scores[row["_crop_index"]] if "_crop_index" in row else None
        print(json.dumps({"face_quality_model": name, "scored": sum(value is not None for value in scores), "images": len(rows)}), flush=True)
    summary = {"face_detection_rate": sum(row["face_detected"] for row in rows)/len(rows)}
    for field in MODELS.values():
        values = [row[field] for row in rows if row[field] is not None]
        summary[field + "_mean"] = float(np.mean(values)) if values else None
        if field == "topiq_face":
            summary["topiq_face_p10"] = float(np.quantile(values, .1)) if values else None
            summary["topiq_face_coverage"] = len(values)/len(rows)
    for row in rows:
        row.pop("_crop_index", None)
    csv_path = directory / f"face_quality_details__manual_val__step_{step:06d}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"metrics": summary, "images": len(rows), "pyiqa_version": "0.1.15", "torch_version": torch.__version__,
              "crop_policy": "largest detected face; 25% padding per side; square clipped to image; 512px Lanczos; all four models use this same crop",
              "models": MODELS, "device": device, "cpu_threads": threads, "model_seconds": model_seconds}
    if device == 'cuda':
        result['peak_reserved_gib'] = torch.cuda.max_memory_reserved()/2**30
        result['reserved_fraction'] = torch.cuda.max_memory_reserved()/torch.cuda.get_device_properties(0).total_memory
        assert result['reserved_fraction'] < .9, 'GPU scoring exceeded the admitted memory budget'
    json_path = directory / "face_quality_summary.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    experiment = connect(config, log_dir or directory)
    if experiment:
        experiment.log_metrics({f"face_quality/{key}": value for key, value in summary.items() if value is not None}, step=step)
        experiment.log_asset(str(csv_path))
        experiment.log_asset(str(json_path))
        experiment.end()
    return result
