#!/usr/bin/env python3
"""Project original reference face boxes through each backend's native resize path."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flux_geometry(width: int, height: int, size: int) -> tuple[int, int, tuple[int, int, int, int]]:
    # Same cap_pixels + center_crop_to_multiple_of_x sequence as pinned Toolkit.
    pixel_count = width * height
    if pixel_count > size * size:
        scale = math.sqrt(size * size / pixel_count)
        width, height = int(width * scale), int(height * scale)
    final_width, final_height = width // 16 * 16, height // 16 * 16
    if min(final_width, final_height) <= 0:
        raise ValueError("Reference becomes empty in FLUX preprocessing")
    left, top = (width - final_width) // 2, (height - final_height) // 2
    return width, height, (left, top, left + final_width, top + final_height)


def qwen_geometry(width: int, height: int, size: int) -> tuple[int, int]:
    # Same calculate_dimensions area/aspect rounding as pinned Qwen pipeline.
    ratio = width / height
    output_width = round(math.sqrt(size * size * ratio) / 32) * 32
    output_height = round((math.sqrt(size * size * ratio) / ratio) / 32) * 32
    if min(output_width, output_height) <= 0:
        raise ValueError("Reference becomes empty in Qwen preprocessing")
    return output_width, output_height


def token_support(mask: Image.Image) -> Image.Image:
    if mask.width % 16 or mask.height % 16:
        raise ValueError("Encoded dimensions must be divisible by 16")
    output = Image.new("L", (mask.width // 16, mask.height // 16), 0)
    pixels = output.load()
    for y in range(output.height):
        for x in range(output.width):
            pixels[x, y] = 255 if mask.crop((x * 16, y * 16, (x + 1) * 16, (y + 1) * 16)).getbbox() else 0
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path(__file__).resolve().parents[1] / "data/validation")
    parser.add_argument("--reference-sizes", type=int, nargs="+", default=[512, 768])
    args = parser.parse_args()
    panel = args.panel.resolve()
    boxes = json.loads((panel / "ref_bboxes.json").read_text())
    root = panel / "masks"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for image_path in sorted((panel / "references").iterdir()):
        if image_path.name not in boxes:
            continue
        with Image.open(image_path) as source:
            width, height = source.size
        bbox = boxes[image_path.name]["face_crop_new"]
        if not (0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height):
            raise ValueError(f"Invalid reference face box: {image_path.name}: {bbox}")
        native = Image.new("L", (width, height), 0)
        ImageDraw.Draw(native).rectangle(tuple(bbox), fill=255)
        for size in args.reference_sizes:
            for backend in ("flux2", "qwen21"):
                if backend == "flux2":
                    resize_width, resize_height, crop = flux_geometry(width, height, size)
                    prepared = native.resize((resize_width, resize_height), Image.Resampling.NEAREST).crop(crop)
                else:
                    resize_width, resize_height = qwen_geometry(width, height, size)
                    prepared = native.resize((resize_width, resize_height), Image.Resampling.NEAREST)
                tokens = token_support(prepared)
                if not tokens.getbbox():
                    raise ValueError(f"Face box disappeared after {backend} transform: {image_path.name}")
                target_dir = root / backend / f"ref{size}"
                target_dir.mkdir(parents=True, exist_ok=True)
                pixel_path = target_dir / f"{image_path.stem}.png"
                token_path = target_dir / f"{image_path.stem}.tokens.png"
                prepared.save(pixel_path)
                tokens.save(token_path)
                records.append({
                    "backend": backend, "reference_size": size, "identity_id": image_path.stem,
                    "reference_image": str(image_path.relative_to(panel)),
                    "reference_sha256": sha256(image_path), "source_size_wh": [width, height],
                    "source_face_bbox": bbox, "encoded_size_wh": list(prepared.size),
                    "token_grid_hw": [tokens.height, tokens.width],
                    "pixel_mask": str(pixel_path.relative_to(panel)),
                    "token_mask": str(token_path.relative_to(panel)),
                    "token_mask_sha256": sha256(token_path),
                    "geometry": "toolkit_default_prep_cap_crop16" if backend == "flux2" else "qwen_pipeline_calculate_dimensions_resize",
                })
    manifest = panel / "reference_masks.jsonl"
    manifest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    print(f"Built {len(records)} backend/size reference masks; manifest SHA256 {sha256(manifest)}")


if __name__ == "__main__":
    main()
