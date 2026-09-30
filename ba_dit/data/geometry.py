"""Explicit pixel transforms and reference face support on native token grids."""

import math

import numpy as np
from PIL import Image, ImageDraw


def face_mask(image: Image.Image, box: list[int]) -> Image.Image:
    x0, y0, x1, y1 = box
    if not 0 <= x0 < x1 <= image.width or not 0 <= y0 < y1 <= image.height:
        raise ValueError(f"Face box {box} does not fit image {image.size}")
    mask = Image.new("L", image.size)
    ImageDraw.Draw(mask).rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
    return mask


def reference_geometry(image: Image.Image, box: list[int], backend: str, size: int):
    mask = face_mask(image, box)
    original = image.size
    if backend == "flux":
        scale = min(1.0, math.sqrt(size * size / (image.width * image.height)))
        width, height = int(image.width * scale), int(image.height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
        left, top = (width % 16) // 2, (height % 16) // 2
        crop = (left, top, left + width // 16 * 16, top + height // 16 * 16)
        image, mask = image.crop(crop), mask.crop(crop)
    else:
        width = round(math.sqrt(size * size * image.width / image.height) / 32) * 32
        height = round(math.sqrt(size * size * image.height / image.width) / 32) * 32
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
        crop = (0, 0, width, height)
    if min(image.size) < 32 or not mask.getbbox():
        raise ValueError("Reference preprocessing lost the face or produced an invalid image")
    pixels = np.array(mask) > 0
    h, w = pixels.shape
    tokens = pixels.reshape(h // 16, 16, w // 16, 16).any(axis=(1, 3))
    return image, tokens, {"source_wh": original, "resize_wh": [width, height], "crop_xyxy": crop,
                           "encoded_wh": image.size, "token_hw": list(tokens.shape)}


def target_geometry(image: Image.Image, size: list[int], box: list[int] | None):
    height, width = size
    scale = max(width / image.width, height / image.height)
    resized = (round(image.width * scale), round(image.height * scale))
    if box:
        face_mask(image, box)
        cx, cy = (box[0] + box[2]) * scale / 2, (box[1] + box[3]) * scale / 2
    else:
        cx, cy = resized[0] / 2, resized[1] / 2
    left = max(0, min(resized[0] - width, round(cx - width / 2)))
    top = max(0, min(resized[1] - height, round(cy - height / 2)))
    crop = (left, top, left + width, top + height)
    if box:
        transformed = [box[0] * resized[0] / image.width - left, box[1] * resized[1] / image.height - top,
                       box[2] * resized[0] / image.width - left, box[3] * resized[1] / image.height - top]
        if not (0 <= transformed[0] < transformed[2] <= width and 0 <= transformed[1] < transformed[3] <= height):
            raise ValueError("Target cover/crop would cut the supplied face; prepare an identity-preserving source crop")
    output = image.resize(resized, Image.Resampling.LANCZOS).crop(crop)
    return output, {"source_wh": image.size, "resize_wh": resized, "crop_xyxy": crop, "encoded_wh": output.size}
