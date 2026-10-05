"""Reproducible structural and saved-image audits; no training or GPU work."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def structure():
    import runpy
    import torch
    from ba_dit import adapters
    scope = runpy.run_path(str(ROOT/'tests/test_flux1_isolated_reference.py'))
    model, config, inputs = scope['fixture']()
    adapters.install(model, config, 'branch_only')
    changed = {**inputs, 'x': inputs['x'].clone()}
    changed['x'][:, :6] += 3
    original = scope['joint_reference_features'](model, inputs)
    altered = scope['joint_reference_features'](model, changed)
    isolated = scope['bank_for'](model, inputs)
    isolated_altered = scope['bank_for'](model, changed)
    differences = {}
    for key in original:
        differences[key] = {'joint_qkv_mean_abs_change':float((original[key]-altered[key]).abs().mean()),
            'isolated_kv_max_abs_change':max(float((getattr(isolated[key],field)-getattr(isolated_altered[key],field)).abs().max())
                                           for field in ('key','value'))}
    passed = []
    for name, function in scope.items():
        if name.startswith('test_') and callable(function):
            function()
            passed.append(name)
    return {'evidence': 'CPU random-weight full-depth Klein9B topology, width16; not pretrained quality validation',
            'seed':142, 'perturbation':'target inputs only +3; reference, sigma and text held fixed',
            'sites':differences, 'critical_tests_passed':passed}


def masks():
    import torch
    from ba_dit.nn.masked_face_flow import face_alpha, token_alpha
    from ba_dit.config import load_config
    source = ROOT/'runs/flux9b_8000_fixed96_20261004'
    config = load_config(source/'resolved_config.yaml')
    width,height = reversed(config['data']['target_size'])
    records=json.loads((source/'routing_masks.json').read_text())['samples']
    output=[]
    for key,row in records.items():
        x0,y0,x1,y1=map(int,row['face_bbox'])
        alpha=face_alpha((width,height),row['face_bbox'],config['branch']['mask_feather_pixels'])
        tokens=token_alpha(alpha)
        expanded=tokens.repeat_interleave(16,-2).repeat_interleave(16,-1)
        inside=expanded[0,0,y0:y1,x0:x1]
        output.append({'sample_id':key,'identity_id':row['identity_id'],
            'fraction_face_pixels_in_fractional_tokens':float((inside<1).float().mean()),
            'mean_native_clean_coefficient_in_face_tokens_at_sigma0':float((1-inside).mean()),
            'fully_native_tokens_inside_face':int((inside==0).sum()),
            'final_pixel_copy_coefficient_inside_face':float((1-alpha[0,0,y0:y1,x0:x1]).max()),
            'soft_effective_tokens':float(tokens.sum()),'binary_support_tokens':int((tokens>0).sum())})
    return {'evidence':'Exact saved full96 mask geometry; coefficients are not measured identity influence',
        'samples':output,'mean_fraction_face_pixels_in_fractional_tokens':float(np.mean([r['fraction_face_pixels_in_fractional_tokens'] for r in output])),
        'mean_native_clean_coefficient_at_sigma0':float(np.mean([r['mean_native_clean_coefficient_in_face_tokens_at_sigma0'] for r in output])),
        'direct_final_pixel_copy_inside_face_max':max(r['final_pixel_copy_coefficient_inside_face'] for r in output)}


def faces(output):
    from PIL import Image
    import torch
    from ba_dit.metrics import LegacyFaces, bbox_iou, cosine
    torch.set_num_threads(4)
    source = ROOT/'runs/flux9b_8000_fixed96_20261004'
    masks=json.loads((source/'routing_masks.json').read_text())['samples']
    report=json.loads((source/'validation-008000/validation.json').read_text())
    detector=LegacyFaces()
    records=[]
    for row in report['samples']:
        key=row['sample_id']; box=masks[key]['face_bbox']; embeddings=[];pixels=[]
        for directory in ('native','validation-008000'):
            image=Image.open(source/directory/row['image']).convert('RGB')
            detected=detector(image)
            ordered=sorted(detected,key=lambda item:bbox_iou(item[0],box),reverse=True)
            if not ordered or bbox_iou(ordered[0][0],box)<.05:
                raise ValueError(f'No matched face: {key} {directory}')
            embeddings.append(ordered[0][1]);pixels.append(np.asarray(image.crop(tuple(map(int,box)))).astype(float))
        records.append({'sample_id':key,'identity_id':row['identity_id'],
                        'native_trained_identity_cosine':cosine(*embeddings),
                        'face_box_rgb_mae_255':float(np.abs(pixels[1]-pixels[0]).mean())})
        if len(records)%12==0:
            print(f'Face comparisons {len(records)}/96',flush=True)
    values=[r['native_trained_identity_cosine'] for r in records]
    return {'samples':records,'mean_native_trained_identity_cosine':float(np.mean(values)),
            'median_native_trained_identity_cosine':float(np.median(values)),
            'min_native_trained_identity_cosine':float(np.min(values)),
            'max_native_trained_identity_cosine':float(np.max(values)),
            'fraction_above_0_8':float(np.mean(np.asarray(values)>.8)),
            'per_identity':{name:float(np.mean([r['native_trained_identity_cosine'] for r in records if r['identity_id']==name]))
                            for name in sorted({r['identity_id'] for r in records})}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('structure','masks','faces'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    value=faces(args.output) if args.action=='faces' else globals()[args.action]()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='samples'},indent=2))
