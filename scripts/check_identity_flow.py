"""Measured admission: real-cache parity, frozen ArcFace gradients and throughput."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ba_dit.config import load_config
from ba_dit.data.manifest import file_hash
from ba_dit.nn.arcface_identity import FrozenOnnxArcFace
from ba_dit.nn.deep_identity_flow import DeepIdentityFlow
from ba_dit.nn.face_identity_loss import aligned_prediction, identity_loss
from ba_dit.nn.reference_refiner_flow import ReferenceRefinerFlow
from ba_dit.runtime import backend_module
from scripts.face_crop_flow import cases, prediction, memory, write


def check(parent,aux):
    torch.set_num_threads(8)
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(142)
    config=load_config(parent/'resolved_config.yaml')
    checkpoint=Path(json.loads((parent/'best_checkpoint.json').read_text())['file'])
    b=DeepIdentityFlow().cuda()
    missing,unexpected=b.load_state_dict(load_file(checkpoint),strict=False)
    assert all(n.startswith('reads.') for n in missing) and not unexpected
    original=ReferenceRefinerFlow().cuda();original.load_state_dict(load_file(checkpoint))
    groups=cases(parent)
    parity=[]
    with torch.no_grad():
        for case in (groups['fit'][0],groups['fit'][-1],groups['probe'][0],groups['probe'][-1]):
            before,after=prediction(original,case),prediction(b,case)
            assert torch.equal(before,after)
            parity.append(float((before-after).abs().max()))
        for group in groups.values():
            for case in group:case['core_velocity']=prediction(b,case,False)
    del original;gc.collect();torch.cuda.empty_cache()
    fit={k:torch.cat([c[k] for c in groups['fit']]) for k in groups['fit'][0]}
    del groups;gc.collect();torch.cuda.empty_cache()
    geometry=json.loads((aux/'geometry.json').read_text())
    arc=FrozenOnnxArcFace(geometry['arcface_path'],expected_sha256=geometry['arcface_sha256']).cuda()
    pair=np.load(aux/'arcface_parity.npz')
    pixels=torch.from_numpy(pair['input']).cuda().requires_grad_()
    expected=torch.from_numpy(pair['embedding']).cuda()
    actual=arc(pixels)
    cosine=float(torch.nn.functional.cosine_similarity(actual,expected).item())
    relative=float(((actual-expected).square().mean().sqrt()/expected.square().mean().sqrt()).detach())
    assert cosine>.99999 and relative<.0002,(cosine,relative)
    actual.square().mean().backward()
    assert pixels.grad.isfinite().all() and pixels.grad.count_nonzero()
    arc_parity={'cosine':cosine,'relative_rms':relative,'max_abs':float((actual-expected).abs().max()),
                'input_gradient_norm':float(pixels.grad.norm()),'weights_frozen':True}
    del pixels,expected,actual
    vae=backend_module(config).load_vae(config)
    reference=torch.from_numpy(np.load(aux/'reference_embedding.npy')).cuda()
    records=json.loads((aux/'cache_manifest.json').read_text())['records']
    parameters=[p for p in b.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=5e-5,weight_decay=.01,fused=True)
    norms=[]
    for index in (1,30,50):
        record=records[index];assert file_hash(aux/record['file'])==record['sha256']
        case={k:v.cuda() for k,v in load_file(aux/record['file']).items()}
        optimizer.zero_grad(set_to_none=True)
        velocity=prediction(b,case)
        flow=(velocity-case['flow_target'].float()).square().mean();flow.backward()
        flow_norm=float(torch.nn.utils.clip_grad_norm_(parameters,float('inf'),error_if_nonfinite=True))
        optimizer.zero_grad(set_to_none=True)
        ident=identity_loss(vae,arc,prediction(b,case),case,reference);ident.backward()
        id_norm=float(torch.nn.utils.clip_grad_norm_(parameters,float('inf'),error_if_nonfinite=True))
        assert id_norm>0 and not any(p.grad is not None for p in vae.parameters())
        norms.append({'row':record['row'],'sigma':record['sigma'],'flow_loss':float(flow.detach()),
            'identity_loss':float(ident.detach()),'flow_gradient_norm':flow_norm,'id_gradient_norm':id_norm})
        print(json.dumps(norms[-1]),flush=True)
    # One identity update per two flow updates: target comparable average norms.
    weight=float(np.clip(np.median([2*n['flow_gradient_norm']/n['id_gradient_norm'] for n in norms]),.01,.5))
    benchmarks=[]
    # Include the largest 704px decoder region, not just a conveniently small face.
    largest={k:v.cuda() for k,v in load_file(aux/records[50]['file']).items()}
    initial={n:p.cpu().detach().clone() for n,p in b.state_dict().items()}
    for size in (64,128,256):
        b.load_state_dict(initial);optimizer.state.clear();gc.collect();torch.cuda.empty_cache()
        batch={k:v[:size] for k,v in fit.items()}
        torch.cuda.reset_peak_memory_stats()
        elapsed=[]
        for step in range(12):
            began=time.monotonic();optimizer.zero_grad(set_to_none=True)
            (prediction(b,batch)-batch['flow_target'].float()).square().mean().backward()
            if step%2:
                (weight*identity_loss(vae,arc,prediction(b,largest),largest,reference)).backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
            torch.cuda.synchronize()
            if step>=4:elapsed.append(time.monotonic()-began)
        result={'batch_size':size,'seconds_per_update_with_largest_ID_crop':sum(elapsed)/len(elapsed),
                'flow_cases_per_second':size/(sum(elapsed)/len(elapsed)),**memory()}
        benchmarks.append(result);print(json.dumps(result),flush=True)
    write(aux/'admission.json',{'parent_prediction_max_abs':parity,'arcface_onnx_parity':arc_parity,
        'gradient_calibration':norms,'suggested_identity_weight':weight,'benchmarks':benchmarks,
        'trainable_parameters':sum(p.numel() for p in parameters),
        'frozen_BA_parameters':sum(p.numel() for p in b.parameters() if not p.requires_grad),
        'check_source_sha256':file_hash(Path(__file__))})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--aux',type=Path,required=True)
    a=p.parse_args();check(a.parent.resolve(),a.aux.resolve())
