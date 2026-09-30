#!/usr/bin/env python3
"""Copy the fixed PhotoMaker manual_val panel without changing its order or prompts."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / "data/validation")
    args = parser.parse_args()
    src, dst = args.source.resolve(), args.destination.resolve()
    required = ["prompts_10.txt", "classes_ref.json", "ref_bboxes.json", "pm96_bboxes_new.json"]
    for name in required:
        if not (src / name).is_file():
            raise FileNotFoundError(src / name)
    images = sorted(p for p in (src / "references").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    prompts = [s.strip() for s in (src / "prompts_10.txt").read_text().splitlines() if s.strip()]
    if len(images) < 8 or len(prompts) != 12:
        raise ValueError(f"Expected at least 8 references and 12 prompts; found {len(images)} and {len(prompts)}")
    classes = json.loads((src / "classes_ref.json").read_text())
    boxes = json.loads((src / "ref_bboxes.json").read_text())
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "references").mkdir(exist_ok=True)
    for name in required:
        target = dst / name
        if target.exists() and sha256(target) != sha256(src / name):
            raise FileExistsError(f"Different existing validation metadata: {target}")
        shutil.copy2(src / name, target)
    for image in images:
        target = dst / "references" / image.name
        if target.exists() and sha256(target) != sha256(image):
            raise FileExistsError(f"Different existing reference image: {target}")
        shutil.copy2(image, target)
    records = []
    # AICODE-NOTE: Legacy manual_val stops after the first 96 items in sorted-image,
    # prompt, seed order. Never infer the panel from all available references.
    for image in images[:8]:
        person = image.stem
        if person not in classes or image.name not in boxes:
            raise ValueError(f"Missing class or reference box for {image.name}")
        cls = classes[person]
        for prompt_index, raw_prompt in enumerate(prompts):
            prompt = raw_prompt.replace("<class>", f"{cls} img")
            records.append({
                "sample_id": f"{len(records):02d}",
                "identity_id": person,
                "reference_image": f"references/{image.name}",
                "reference_sha256": sha256(image),
                "reference_face_bbox": boxes[image.name]["face_crop_new"],
                "prompt": prompt,
                "prompt_index": prompt_index,
                "seed": 0,
                "split": "validation",
            })
    manifest = dst / "manual_val_96.jsonl"
    content = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    if manifest.exists() and manifest.read_text() != content:
        raise FileExistsError(f"Different existing validation manifest: {manifest}")
    manifest.write_text(content)
    print(f"Imported {len(images)} references and {len(records)} fixed validation items; manifest SHA256 {sha256(manifest)}")


if __name__ == "__main__":
    main()
