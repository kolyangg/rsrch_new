"""Streaming decode preserves pixels/RNG and commits only complete images."""
import json
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np
from PIL import Image
import torch
from safetensors.torch import save_file

from ba_dit.data.manifest import file_hash
from ba_dit.nn.masked_face_flow import face_alpha,preserve_background
from scripts.clust_stream_decode import StreamingDecoder,infer


def test_stream_decode_preserves_native_pixels_rng_and_resume(tmp_path):
    native=tmp_path/'native';native.mkdir()
    panel=tmp_path/'validation-002000';panel.mkdir()
    Image.new('RGB',(32,32),(10,20,30)).save(native/'00.png')
    (tmp_path/'routing_masks.json').write_text(json.dumps({'samples':{'00':{
        'baseline_image_sha256':file_hash(native/'00.png'),'face_bbox':[8,8,24,24]}}}))
    save_file({'latent':torch.zeros(1,3,4,4)},panel/'00.safetensors',metadata={'checkpoint_sha256':'checkpoint'})
    def load(config):
        torch.randn(7)
        return object()
    raw=Image.new('RGB',(32,32),(100,200,50))
    backend=SimpleNamespace(load_vae=Mock(side_effect=load),decode=Mock(return_value=raw))
    module=SimpleNamespace(__file__=__file__,backend_module=lambda c:backend,
                           face_alpha=face_alpha,preserve_background=preserve_background,memory=Mock())
    config={'branch':{'mask_feather_pixels':2}}
    report={'checkpoint_sha256':'checkpoint','samples':[{'sample_id':'00','image':'00.png'}]}
    rng=torch.get_rng_state()
    with patch('torch.cuda.is_available',return_value=False):
        decoder=StreamingDecoder(module,tmp_path,config,2000);decoder.update(report)
        decoder=StreamingDecoder(module,tmp_path,config,2000);decoder.update(report)
    assert torch.equal(rng,torch.get_rng_state())
    assert backend.decode.call_count==1
    expected=preserve_background(raw,Image.open(native/'00.png').convert('RGB'),face_alpha((32,32),[8,8,24,24],2))
    assert np.array_equal(np.asarray(expected),np.asarray(Image.open(panel/'00.png')))
    assert not list(panel.glob('*.tmp'))
    assert json.loads((tmp_path/'background_audit_2000.json').read_text())[0]['background_max_abs']==0


def test_stream_hook_restored_after_generation_failure(tmp_path):
    original=Mock()
    module=SimpleNamespace(write=original,infer=Mock(side_effect=RuntimeError('generation failed')))
    with patch('scripts.clust_stream_decode.StreamingDecoder'):
        try:infer(module,tmp_path,{},2000)
        except RuntimeError:pass
        else:raise AssertionError('Generation failure was hidden')
    assert module.write is original
