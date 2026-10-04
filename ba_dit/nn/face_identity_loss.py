"""Identity supervision through frozen FLUX VAE and ArcFace, on training photos."""
import torch
from torch.nn import functional as F


def aligned_prediction(vae, velocity, case):
    # FLUX rectified flow is v = noise - clean, hence x0 = x_sigma - sigma*v.
    clean = case['noise'].float() - case['timestep'].float().reshape(-1,1,1)*velocity
    clean = clean.transpose(1,2).reshape_as(case['clean'])
    # Packed FLUX2 latents already include the native BN normalization. decode
    # reverses it and unpacks; no SDXL scaling/shift is appropriate here.
    decoded = vae.decode(clean.to(vae.dtype)).float()
    return F.grid_sample(decoded, case['grid'], mode='bilinear',
                         padding_mode='border', align_corners=True).clamp(-1,1)


def identity_loss(vae, recognizer, velocity, case, reference_embedding):
    aligned = aligned_prediction(vae, velocity, case)
    embedding = F.normalize(recognizer(aligned), dim=-1)
    return (1-(embedding*reference_embedding).sum(-1)).mean()
