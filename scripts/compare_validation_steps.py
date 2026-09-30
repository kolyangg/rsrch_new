#!/usr/bin/env python3
"""Compare the same named validation panel at step zero and a later checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def metric_delta(before, after, filename):
    left = json.loads((before / filename).read_text())["metrics"]
    right = json.loads((after / filename).read_text())["metrics"]
    return {key: {"step_0": left[key], "later": right[key],
                  "delta": right[key] - left[key] if left[key] is not None and right[key] is not None else None}
            for key in sorted(left.keys() & right.keys())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--later-step", type=int, default=2000)
    args = parser.parse_args()
    before = args.run_dir / "validation-000000"
    after = args.run_dir / f"validation-{args.later_step:06d}"
    first = json.loads((before / "validation.json").read_text())
    last = json.loads((after / "validation.json").read_text())
    ids = [row["sample_id"] for row in first["samples"]]
    if ids != [row["sample_id"] for row in last["samples"]] or first["panel_sha256"] != last["panel_sha256"]:
        raise ValueError("Validation panels or order differ")
    images = []
    for left, right in zip(first["samples"], last["samples"]):
        with Image.open(before / left["image"]) as source, Image.open(after / right["image"]) as trained:
            a, b = np.asarray(source.convert("RGB")), np.asarray(trained.convert("RGB"))
        if a.shape != b.shape:
            raise ValueError(f"Image geometry changed for {left['sample_id']}")
        difference = np.abs(a.astype(np.int16) - b.astype(np.int16))
        images.append({"sample_id": left["sample_id"], "changed": bool(np.any(difference)),
                       "mean_abs_pixel": float(difference.mean()), "max_abs_pixel": int(difference.max())})
    result = {"panel_sha256": first["panel_sha256"], "later_step": args.later_step,
              "image_count": len(images), "changed_images": sum(row["changed"] for row in images),
              "images": images,
              "legacy_metrics": metric_delta(before, after, "quality_summary.json"),
              "face_quality_metrics": metric_delta(before, after, "face_quality_summary.json")}
    output = args.run_dir / f"validation_change_000000_to_{args.later_step:06d}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(output), "changed_images": result["changed_images"],
                      "image_count": len(images), "legacy_metrics": result["legacy_metrics"],
                      "face_quality_metrics": result["face_quality_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
