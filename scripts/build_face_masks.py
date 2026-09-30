#!/usr/bin/env python3
"""Export the same reference masks and geometry used by both runtime backends."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ba_dit.data.geometry import face_mask, reference_geometry
from ba_dit.data.manifest import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=Path(__file__).resolve().parents[1] / 'data/validation')
    parser.add_argument('--reference-sizes', type=int, nargs='+', default=[512, 768])
    args = parser.parse_args()
    panel = args.panel.resolve()
    boxes = json.loads((panel / 'ref_bboxes.json').read_text())
    records = []
    for path in sorted((panel / 'references').iterdir()):
        if path.name not in boxes:
            continue
        with Image.open(path) as source:
            image = source.convert('RGB')
        box = boxes[path.name]['face_crop_new']
        for size in args.reference_sizes:
            for backend, native in [('flux2', 'flux'), ('qwen21', 'qwen')]:
                _, tokens, geometry = reference_geometry(image, box, native, size)
                mask = face_mask(image, box).resize(tuple(geometry['resize_wh']), Image.Resampling.NEAREST).crop(geometry['crop_xyxy'])
                destination = panel / 'masks' / backend / f'ref{size}'
                destination.mkdir(parents=True, exist_ok=True)
                pixel_path, token_path = destination / f'{path.stem}.png', destination / f'{path.stem}.tokens.png'
                mask.save(pixel_path)
                Image.fromarray(tokens.astype(np.uint8) * 255).save(token_path)
                records.append(dict(backend=backend, reference_size=size, identity_id=path.stem,
                    reference_image=str(path.relative_to(panel)), reference_sha256=file_hash(path),
                    source_size_wh=list(image.size), source_face_bbox=box, encoded_size_wh=geometry['encoded_wh'],
                    token_grid_hw=geometry['token_hw'], pixel_mask=str(pixel_path.relative_to(panel)),
                    token_mask=str(token_path.relative_to(panel)), token_mask_sha256=file_hash(token_path),
                    geometry=geometry, box_convention='xyxy-exclusive'))
    manifest = panel / 'reference_masks.jsonl'
    manifest.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in records))
    print(f'Built {len(records)} masks; manifest SHA256 {file_hash(manifest)}')


if __name__ == '__main__':
    main()
