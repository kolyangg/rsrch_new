import numpy as np
import torch
from PIL import Image
from ba_dit.nn.masked_face_flow import face_alpha, token_alpha, scene_latent, preserve_background


def test_zero_face_flow_keeps_noise_and_native_background():
    alpha = face_alpha((128,128),[32,32,96,96])
    mask = token_alpha(alpha)
    noise, native = torch.randn(1,4,8,8), torch.randn(1,4,8,8)
    result = scene_latent(noise,native,noise,0.,mask)
    assert torch.equal(result[mask.expand_as(result)==1],noise[mask.expand_as(noise)==1])
    assert torch.equal(result[mask.expand_as(result)==0],native[mask.expand_as(native)==0])
    assert torch.equal(scene_latent(noise,native,noise,0.,torch.zeros_like(mask)),native)
    original = Image.new('RGB',(128,128),(30,60,80))
    changed = Image.new('RGB',(128,128),(150,10,20))
    output = np.asarray(preserve_background(changed,original,alpha))
    assert np.array_equal(output[0,0],np.array(original)[0,0])
    assert np.array_equal(output[64,64],np.array(changed)[64,64])
