"""Two-pass spatial routing: generated native scene outside, BA flow inside.

AICODE-NOTE: On 2026-10-01 the user explicitly authorized native-output masks
for inference in this named experiment. These are never target-photo masks.
"""

import numpy as np
import torch
from PIL import Image


def face_alpha(size, box, feather=16):
    """Full weight inside the detected box, a feathered outer ring, zero beyond."""
    width, height = size
    if box is None or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError("A reviewed native-backbone face box is required")
    x, y = np.arange(width), np.arange(height)
    dx = np.maximum(box[0]-x, x-(box[2]-1))
    dy = np.maximum(box[1]-y, y-(box[3]-1))
    distance = np.maximum(dy[:, None], dx[None, :])
    alpha = np.clip(1-distance/max(feather, 1), 0, 1).astype(np.float32)
    return torch.from_numpy(alpha)[None, None]


def token_alpha(alpha):
    return torch.nn.functional.avg_pool2d(alpha, 16)


def scene_latent(face, native_clean, initial_noise, sigma, mask):
    """The native generated image supplies the known background at every sigma."""
    background = (1-sigma)*native_clean + sigma*initial_noise
    return torch.lerp(background.float(), face.float(), mask.float()).to(face.dtype)


def preserve_background(generated, native, alpha):
    """VAE decoding mixes nearby pixels; enforce exact outside-support pixels."""
    original = np.asarray(native.convert("RGB"))
    changed = np.asarray(generated.convert("RGB"))
    mask = alpha.squeeze().cpu().numpy()[..., None]
    output = np.rint(original*(1-mask) + changed*mask).clip(0, 255).astype(np.uint8)
    outside = mask[..., 0] == 0
    if not np.array_equal(output[outside], original[outside]):
        raise RuntimeError("Background preservation failed")
    return Image.fromarray(output)
