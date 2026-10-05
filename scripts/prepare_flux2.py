"""Prepare frozen reference-only ArcFace/DINO features and face-erased RGB context."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ba_dit.config import ROOT, load_config
from ba_dit.data.geometry import target_geometry
from ba_dit.data.manifest import file_hash, read_manifest

DINO_REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'


def main(args):
    config = load_config(args.config)
    root = Path(config['data']['flux2_cache'])
    rows = read_manifest(config['data']['train_manifest'], training=True) + read_manifest(config['data']['validation_manifest'])
    refs = {r['reference_hash']:r for r in rows}
    if args.references:
        from scripts import prepare_flux1_identity as prep
        from ba_dit.metrics import bbox_iou
        prep.initialize_detector(2)
        data = {}
        for key, row in refs.items():
            pixels = np.asarray(Image.open(row['reference']).convert('RGB'))[:, :, ::-1].copy()
            faces = []
            for size in (640,576,512,448,384,320,256):
                prep.APP.det_model.input_size = (size,size)
                faces = sorted(prep.APP.get(pixels), key=lambda f:bbox_iou(f.bbox,row['reference_box']), reverse=True)
                if faces and bbox_iou(faces[0].bbox,row['reference_box']) >= .3:
                    break
            if not faces or bbox_iou(faces[0].bbox,row['reference_box']) < .3:
                raise ValueError('Reference face ownership failed: '+row['sample_id'])
            e = faces[0].embedding.astype(np.float32)
            data[key] = (e/np.linalg.norm(e)).tolist()
        data['arcface_sha256'] = file_hash(prep.APP.models['recognition'].model_file)
        (root/'reference_identity.json').write_text(json.dumps(data))
        return
    import torch
    from torch.nn import functional as F
    from safetensors.torch import save_file
    from transformers import AutoImageProcessor, Dinov2Model
    from ba_dit.data.flux2_memory import sanitize
    from ba_dit.runtime import backend_module
    torch.set_num_threads(8)
    model_dir = ROOT/'weights/flux2_features/dinov2-small'
    from huggingface_hub import snapshot_download
    snapshot_download('facebook/dinov2-small', revision=DINO_REVISION, local_dir=model_dir,
                      allow_patterns=['config.json','preprocessor_config.json','model.safetensors'])
    dino = Dinov2Model.from_pretrained(model_dir).eval().requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(model_dir)
    identities = json.loads((root/'reference_identity.json').read_text())
    details = {}
    with torch.no_grad():
        for key, row in refs.items():
            image = Image.open(row['reference']).convert('RGB').crop(row['reference_box']).resize((224,224),Image.Resampling.BICUBIC)
            inputs = processor(images=image, return_tensors='pt', do_resize=False, do_center_crop=False)
            patch = dino(**inputs).last_hidden_state[:,1:].transpose(1,2).reshape(1,384,16,16)
            details[key] = F.adaptive_avg_pool2d(patch,(8,8)).flatten(2).transpose(1,2).contiguous()
    del dino
    backend = backend_module(config)
    vae = backend.load_vae(config)
    from extensions_built_in.diffusion_models.flux2.src.sampling import default_images_prep
    native = ROOT/'runs/flux4b_deep_identity1024_det_20261001'
    masks = json.loads((native/'routing_masks.json').read_text())['samples']
    record = {'dino_revision':DINO_REVISION,'arcface_sha256':identities['arcface_sha256'],
              'target_size':config['data']['target_size'],'feather':config['branch']['mask_feather_pixels'],
              'train_sha256':file_hash(config['data']['train_manifest']),
              'validation_sha256':file_hash(config['data']['validation_manifest']),'samples':{}}
    with torch.no_grad():
        for row in rows:
            key = row['sample_id']
            if 'target' in row:
                image, geo = target_geometry(Image.open(row['target']).convert('RGB'),config['data']['target_size'],row['target_box'])
                scale = np.asarray(geo['resize_wh'])/np.asarray(geo['source_wh'])
                left,top = geo['crop_xyxy'][:2]
                box = np.asarray(row['target_box'])*[*scale,*scale]-[left,top,left,top]
                source_hash = row['target_hash']
            else:
                path = native/'native'/f'{key}.png'
                source_hash = file_hash(path)
                assert source_hash == masks[key]['baseline_image_sha256']
                image = Image.open(path).convert('RGB');box=masks[key]['face_bbox']
            clean,keep = sanitize(image,box,config['branch']['mask_feather_pixels'])
            context = vae.encode(default_images_prep(clean).unsqueeze(0).to(vae.device,vae.dtype)).cpu()
            tensors = {'flux2_context':context,'flux2_context_keep':keep,
                       'flux2_identity':torch.tensor(identities[row['reference_hash']]).unsqueeze(0),
                       'flux2_detail':details[row['reference_hash']]}
            path = root/f'{key}.safetensors'
            save_file(tensors,path)
            record['samples'][key] = {'file':path.name,'sha256':file_hash(path),
                'reference_hash':row['reference_hash'],'target_hash':row.get('target_hash'),
                'context_rgb_source_sha256':source_hash,'erase_box':list(map(float,box)),
                'context_tokens':int(keep.sum()),'context_source':'training photo' if 'target' in row else 'frozen native generation'}
            print(f'Prepared {key}: {int(keep.sum())} context tokens',flush=True)
    (root/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--references',action='store_true')
    main(p.parse_args())
