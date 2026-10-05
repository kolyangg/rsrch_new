"""Training-target aligned identity supervision for the full-denoiser flow objective."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ba_dit.data.manifest import file_hash
from ba_dit.nn.arcface_identity import FrozenOnnxArcFace
from ba_dit.nn.deterministic_identity_loss import aligned_crop


def alignment_grid(inverse, height, width, device):
    """Fixed inverse affine: aligned 112px coordinates to full training image."""
    y, x = torch.meshgrid(torch.arange(112, device=device, dtype=torch.float32),
                          torch.arange(112, device=device, dtype=torch.float32), indexing='ij')
    coordinates = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    pixels = coordinates @ torch.as_tensor(inverse, device=device, dtype=torch.float32).T
    return (pixels * pixels.new_tensor([2 / (width - 1), 2 / (height - 1)]) - 1).unsqueeze(0)


def supervision_identity(config):
    root = Path(config['data']['identity_supervision'])
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['train_manifest_sha256'] != file_hash(config['data']['train_manifest']):
        raise ValueError('Identity supervision belongs to a different training manifest')
    if manifest['target_size'] != config['data']['target_size'] or not manifest['complete']:
        raise ValueError('Incomplete identity supervision or mismatched training geometry')
    root_source = Path(__file__).resolve().parents[2]
    if manifest['geometry_sha256'] != file_hash(root_source / 'ba_dit/data/geometry.py'):
        raise ValueError('Training geometry changed since identity preparation')
    if manifest['preparation_code_sha256'] != file_hash(root_source / 'scripts/prepare_flux1_identity.py'):
        raise ValueError('Identity preparation code changed; prepare a fresh supervision directory')
    for name, expected in manifest['files'].items():
        if file_hash(root / name) != expected:
            raise ValueError(f'Identity supervision hash mismatch: {name}')
    if manifest['accepted_fraction'] < .9:
        raise ValueError('Fewer than 90% of training targets have usable identity supervision')
    if file_hash(manifest['arcface_path']) != manifest['arcface_sha256']:
        raise ValueError('Prepared ArcFace weights changed')
    return {'manifest_sha256': file_hash(root / 'manifest.json'), 'files': manifest['files'],
            'arcface_sha256': manifest['arcface_sha256'], 'accepted_fraction': manifest['accepted_fraction']}


class OnlineIdentityObjective:
    def __init__(self, config, vae):
        self.config, self.vae = config, vae
        supervision_identity(config)
        root = Path(config['data']['identity_supervision'])
        manifest = json.loads((root / 'manifest.json').read_text())
        self.rows = {r['sample_id']: r for r in map(json.loads, (root / 'records.jsonl').read_text().splitlines())}
        self.embeddings = np.load(root / 'target_embeddings.npy', mmap_mode='r')
        self.recognizer = FrozenOnnxArcFace(manifest['arcface_path'],
            expected_sha256=manifest['arcface_sha256']).to(vae.device).eval()
        self.last_metrics = {}

    def loss(self, prediction, noisy, sigma, row, step, force=False):
        spec = self.config['training']['identity_loss']
        rec = self.rows[row['sample_id']]
        if rec['target_sha256'] != row['target_hash']:
            raise ValueError('Identity target differs from the training pair')
        valid = rec['accepted']
        active = valid and (force or (step % spec['every'] == 0 and float(sigma.max()) <= spec['max_sigma']))
        self.last_metrics = {'train/identity_active': int(active), 'train/identity_target_valid': int(valid)}
        if not active:
            return prediction.new_zeros((), dtype=torch.float32)
        clean = noisy.float() - sigma.float().reshape(-1, 1, 1, 1) * prediction.float()
        # Freeze VAE weights, retaining the generated-image gradient. Checkpoint
        # its activations rather than detaching or using the ONNX metric path.
        decoded = checkpoint(self.vae.decode, clean.to(self.vae.dtype), use_reentrant=False).float()
        grid = alignment_grid(rec['aligned_to_image'], *decoded.shape[-2:], decoded.device)
        face = aligned_crop(decoded, grid).clamp(-1, 1)
        embedding = F.normalize(self.recognizer(face), dim=-1)
        target = torch.as_tensor(np.array(self.embeddings[rec['embedding_index']], copy=True),
                                 device=embedding.device).unsqueeze(0)
        loss = (1 - (embedding * target).sum(-1)).mean()
        self.last_metrics['train/identity_loss'] = float(loss.detach())
        return loss * spec['weight']
