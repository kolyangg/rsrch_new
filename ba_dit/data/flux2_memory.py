"""Verified frozen reference features and RGB-erased context; no native face latents."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from ba_dit.data.manifest import file_hash
from ba_dit.nn.masked_face_flow import face_alpha, token_alpha


def sanitize(image, box, feather):
    # Erase every original feather pixel, plus a 32-pixel halo, BEFORE VAE encoding.
    exclusion = face_alpha(image.size, box, feather+32).gt(0)
    pixels = np.asarray(image.convert('RGB')).copy()
    pixels[exclusion.squeeze().numpy()] = 127
    keep = token_alpha(exclusion.float()).eq(0).flatten(1)
    return Image.fromarray(pixels), keep


@lru_cache(maxsize=8)
def manifest(root, manifest_hash):
    return json.loads((Path(root)/'manifest.json').read_text())


def memory_identity(config):
    root = Path(config['data']['flux2_cache'])
    data = manifest(str(root), file_hash(root/'manifest.json'))
    if data['target_size'] != config['data']['target_size'] or data['feather'] != config['branch']['mask_feather_pixels']:
        raise ValueError('FLUX2 context geometry changed')
    for split in ('train', 'validation'):
        if data[split+'_sha256'] != file_hash(config['data'][split+'_manifest']):
            raise ValueError('FLUX2 cache belongs to different data')
    return {'manifest_sha256':file_hash(root/'manifest.json'), 'dino_revision':data['dino_revision'],
            'arcface_sha256':data['arcface_sha256']}


def load_memory(config, row, device='cpu'):
    root = Path(config['data']['flux2_cache'])
    memory_identity(config)
    record = manifest(str(root), file_hash(root/'manifest.json'))['samples'][row['sample_id']]
    if record['reference_hash'] != row['reference_hash'] or record.get('target_hash') != row.get('target_hash'):
        raise ValueError('FLUX2 conditioning image changed')
    path = root/record['file']
    if file_hash(path) != record['sha256']:
        raise ValueError('FLUX2 frozen conditioning hash changed')
    return load_file(path, device=device)
