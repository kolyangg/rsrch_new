"""Named eight-image causal diagnostic; never changes fixed96 evaluation or Comet curves."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file, save_file

from ba_dit import adapters
from ba_dit.checkpoint import load_adapters
from ba_dit.config import ROOT
from ba_dit.data.cache import load_pair
from ba_dit.data.manifest import file_hash, read_manifest
from ba_dit.nn.masked_face_flow import face_alpha, routing_token_alpha, scene_latent, preserve_background
from ba_dit.runtime import backend_module
from scripts.online_face_ba import verify, write, memory


@torch.no_grad()
def generate(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    config, _ = verify(args.run)
    if config['branch'].get('reference_bank') != 'isolated_image':
        raise ValueError('This probe requires a new FLUX1a/b/c isolated-bank run')
    lock = (ROOT/'runs/face_flow_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Stable first occurrence per identity in the original order: eight named
    # diagnostics, never a substitute for the complete 96-image comparison.
    selected = {}
    for row in read_manifest(config['data']['validation_manifest']):
        selected.setdefault(row['identity_id'], row)
    rows = list(selected.values())
    assert len(rows) == 8 and all('target' not in row for row in rows)
    args.output.mkdir(parents=True, exist_ok=False)
    backend = backend_module(config)
    from extensions_built_in.diffusion_models.flux2.src.sampling import get_schedule
    model = backend.load_transformer(config)
    adapters.install(model, config, 'branch_only')
    checkpoint = args.run/f'checkpoint-{args.step:06d}'
    load_adapters(model, checkpoint, config, 'branch_only')
    masks = json.loads((args.run/'routing_masks.json').read_text())['samples']
    height, width = config['data']['target_size']
    times = get_schedule(config['validation']['steps'], height*width//256)
    samples = []
    for index, row in enumerate(rows):
        donor = rows[(index+1) % len(rows)]
        key = row['sample_id']
        own, _ = load_pair(config, row, 'cuda', negative=True)
        other, _ = load_pair(config, donor, 'cuda', negative=True)
        assert 'target_latent' not in own and 'target_latent' not in other
        native_path = args.run/'native'/f'{key}.safetensors'
        assert file_hash(native_path) == masks[key]['latent_sha256']
        assert file_hash(args.run/'native'/f'{key}.png') == masks[key]['baseline_image_sha256']
        native = load_file(native_path)['latent'].cuda()
        alpha = routing_token_alpha(face_alpha((width,height), masks[key]['face_bbox'],
                                    config['branch']['mask_feather_pixels']), config).cuda()
        own['target_face_mask'] = alpha.flatten(1)
        noise = torch.randn(native.shape, generator=torch.Generator().manual_seed(row['seed']), dtype=native.dtype).cuda()
        for arm in ('ba_off', 'own_bank', 'donor_bank'):
            face = noise.clone()
            for current, following in zip(times[:-1], times[1:]):
                sigma = face.new_tensor([current])
                mixed = scene_latent(face,native,noise,current,alpha)
                kwargs = {'branch':arm != 'ba_off', 'reference_source':other if arm == 'donor_bank' else None}
                positive = backend.predict(model,own,mixed,sigma,config,**kwargs)
                negative = backend.predict(model,own,mixed,sigma,config,negative=True,**kwargs)
                face = face+(following-current)*(negative+config['validation']['guidance']*(positive-negative))
            latent = scene_latent(face,native,noise,0.,alpha)
            assert torch.isfinite(latent).all()
            outside = alpha.expand_as(latent)==0
            assert torch.equal(latent[outside],native[outside])
            save_file({'latent':latent.cpu().contiguous()},args.output/f'{key}_{arm}.safetensors')
        samples.append({'sample_id':key, 'identity_id':row['identity_id'], 'donor_id':donor['identity_id'],
            'donor_sample_id':donor['sample_id'], 'seed':row['seed'], 'prompt':row['prompt'],
            'native_reference_sha256':row['reference_hash'], 'bank_donor_reference_sha256':donor['reference_hash'],
            'face_bbox':masks[key]['face_bbox']})
        print(f'Causal diagnostic: {len(samples)}/8 completed',flush=True)
    measured = memory()
    del model, own, other, positive, negative, face, latent, native
    gc.collect(); torch.cuda.empty_cache()
    vae = backend.load_vae(config)
    for row in samples:
        key = row['sample_id']
        original = Image.open(args.run/'native'/f'{key}.png').convert('RGB')
        alpha = face_alpha(original.size,row['face_bbox'],config['branch']['mask_feather_pixels'])
        for arm in ('ba_off', 'own_bank', 'donor_bank'):
            latent = load_file(args.output/f'{key}_{arm}.safetensors')['latent'].cuda()
            image = backend.decode(vae,latent,config)
            preserve_background(image,original,alpha).save(args.output/f'{key}_{arm}.png')
    write(args.output/'probe.json',{'name':'branch_reference_swap8', 'run':str(args.run), 'step':args.step,
        'checkpoint_sha256':file_hash(checkpoint/'adapters.safetensors'), 'samples':samples,
        'native_inputs_fixed':True, 'target_photos_loaded':False, 'first_sample_per_identity':True,
        'official_fixed96_metrics':False, 'source_sha256':file_hash(__file__), **measured})


def score(args):
    from ba_dit.metrics import LegacyFaces, bbox_iou, cosine
    report = json.loads((args.output/'probe.json').read_text())
    prototypes = torch.load(ROOT/'data/validation/id_embeds_manual_val_subject_v2.pth',map_location='cpu',weights_only=True)
    detector = LegacyFaces()
    results = []
    for row in report['samples']:
        scores = {}
        for arm in ('ba_off', 'own_bank', 'donor_bank'):
            faces = detector(Image.open(args.output/f"{row['sample_id']}_{arm}.png"))
            faces.sort(key=lambda f:bbox_iou(f[0],row['face_bbox']),reverse=True)
            valid = bool(faces and bbox_iou(faces[0][0],row['face_bbox'])>=.05)
            scores[arm] = {'owner_found':valid,
                'own_id':cosine(faces[0][1],prototypes[row['identity_id']]) if valid else 0.,
                'donor_id':cosine(faces[0][1],prototypes[row['donor_id']]) if valid else 0.}
        results.append({'sample_id':row['sample_id'], 'identity_id':row['identity_id'], 'scores':scores,
            'own_bank_gain_over_off':scores['own_bank']['own_id']-scores['ba_off']['own_id'],
            'donor_swap_donor_gain':scores['donor_bank']['donor_id']-scores['own_bank']['donor_id'],
            'donor_swap_own_drop':scores['own_bank']['own_id']-scores['donor_bank']['own_id']})
    summary = {key:float(np.mean([r[key] for r in results])) for key in
               ('own_bank_gain_over_off','donor_swap_donor_gain','donor_swap_own_drop')}
    write(args.output/'probe_scores.json',{'samples':results,'means':summary,
        'definitions':'Frozen subject-v2 prototypes; generated-mask owner match IoU >= .05; missing owner scored zero',
        'subject_prototypes_sha256':file_hash(ROOT/'data/validation/id_embeds_manual_val_subject_v2.pth'),
        'interpretation':'Positive donor gain together with own-ID drop supports branch-reference causality; inspect all faces for artifacts.'})
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('generate','score'))
    parser.add_argument('--run',type=Path)
    parser.add_argument('--step',type=int,default=2000)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action == 'generate':
        if args.run is None:
            parser.error('generate requires --run')
        args.run=args.run.resolve()
        generate(args)
    else:
        score(args)
