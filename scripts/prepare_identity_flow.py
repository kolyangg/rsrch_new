"""Training-only aligned face regions for differentiable identity supervision."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ba_dit.config import load_config
from ba_dit.data.geometry import target_geometry
from ba_dit.data.manifest import file_hash, read_manifest


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def align(source, destination):
    import cv2
    from insightface.app import FaceAnalysis
    from insightface.utils.face_align import estimate_norm, norm_crop
    config = load_config(source/'resolved_config.yaml')
    identity = json.loads((source/'identity.json').read_text())
    destination.mkdir(parents=True, exist_ok=True)
    assert not (destination/'geometry.json').exists(), 'Completed geometry is immutable'
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'],
                       allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=-1, det_size=(640,640))

    def select(image, box):
        bgr = np.asarray(image)[:,:,::-1].copy()
        faces = []
        for size in (640,576,512,448,384,320,256):
            app.det_model.input_size=(size,size)
            faces=app.get(bgr)
            if faces:break
        if not faces:
            raise ValueError('No training face found for identity alignment')
        def iou(face):
            a=np.asarray(face.bbox); b=np.asarray(box)
            inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
            return inter/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter)
        face=max(faces,key=iou)
        assert iou(face)>.3, 'Training face ownership failed'
        return face,bgr

    reference=identity['training_reference']
    face,bgr=select(Image.open(reference['reference']).convert('RGB'),reference['reference_box'])
    aligned=norm_crop(bgr,face.kps,112)
    normalized=(aligned[:,:,::-1].astype(np.float32).transpose(2,0,1)[None]-127.5)/127.5
    recognition=app.models['recognition']
    embedding=recognition.session.run(None,{recognition.input_name:normalized})[0]
    np.savez(destination/'arcface_parity.npz',input=normalized,embedding=embedding)
    np.save(destination/'reference_embedding.npy',embedding/np.linalg.norm(embedding,axis=-1,keepdims=True))
    Image.fromarray(aligned[:,:,::-1]).save(destination/'reference_aligned.png')
    records=[]
    for i,row in enumerate(read_manifest(config['data']['train_manifest'],training=True)):
        assert row['identity_id']==reference['identity_id']
        original=Image.open(row['target']).convert('RGB')
        image,geometry=target_geometry(original,config['data']['target_size'],row['target_box'])
        sx,sy=np.asarray(geometry['resize_wh'])/np.asarray(geometry['source_wh'])
        left,top=geometry['crop_xyxy'][:2]
        box=(np.asarray(row['target_box'])*[sx,sy,sx,sy]-[left,top,left,top]).tolist()
        face,bgr=select(image,box)
        matrix=estimate_norm(face.kps,image_size=112)
        inverse=cv2.invertAffineTransform(matrix)
        yy,xx=np.mgrid[:112,:112]
        pixels=np.stack((xx,yy,np.ones_like(xx)),-1)@inverse.T
        # Include decoder context beyond the aligned face, on the packed 16px grid.
        low=np.maximum(0,np.floor((pixels.min((0,1))-48)/16)*16).astype(int)
        high=np.minimum(768,np.ceil((pixels.max((0,1))+49)/16)*16).astype(int)
        assert (high>low).all()
        grid=2*(pixels-low)/(high-low-1)-1
        assert np.abs(grid).max()<=1, 'Aligned crop falls outside decoded training region'
        record={'row':i,'sample_id':row['sample_id'],'target_sha256':row['target_hash'],
                'pixel_crop_xyxy':[*low.tolist(),*high.tolist()],
                'landmarks':face.kps.tolist(),'source_to_aligned':matrix.tolist()}
        np.save(destination/f'grid_{i:02d}.npy',grid.astype(np.float32)[None])
        Image.fromarray(norm_crop(bgr,face.kps,112)[:,:,::-1]).save(destination/f'aligned_{i:02d}.png')
        records.append(record)
    model=Path(recognition.model_file)
    write(destination/'geometry.json',{'records':records,'reference':reference,
          'reference_embedding_sha256':file_hash(destination/'reference_embedding.npy'),
          'arcface_path':str(model),'arcface_sha256':file_hash(model),
          'train_manifest_sha256':file_hash(config['data']['train_manifest']),
          'validation_images_used':False,'alignment':'five training-photo landmarks, bilinear differentiable grid'})
    print(json.dumps({'faces':len(records),'crop_sizes':[[r['pixel_crop_xyxy'][2]-r['pixel_crop_xyxy'][0],
          r['pixel_crop_xyxy'][3]-r['pixel_crop_xyxy'][1]] for r in records]}),flush=True)


def cache(source, destination):
    import torch
    from safetensors.torch import save_file
    from ba_dit.data.cache import load_pair
    from ba_dit.nn.face_crop_flow import capture, install
    from ba_dit.runtime import backend_module
    from scripts.masked_face_flow import with_fixed_reference
    from scripts.face_crop_flow import memory
    config=load_config(source/'resolved_config.yaml')
    geometry=json.loads((destination/'geometry.json').read_text())
    assert file_hash(config['data']['train_manifest'])==geometry['train_manifest_sha256']
    backend=backend_module(config); model=backend.load_transformer(config)
    branch=install(model,True,True,False)
    reference,_=load_pair(config,geometry['reference'],'cuda')
    rows=read_manifest(config['data']['train_manifest'],training=True)
    (destination/'cases').mkdir(exist_ok=True)
    records=[]; began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for rec in geometry['records']:
            i=rec['row'];row=rows[i]
            assert row['target_hash']==rec['target_sha256']
            tensors=with_fixed_reference(config,row,reference);target=tensors['target_latent']
            x0,y0,x1,y1=[v//16 for v in rec['pixel_crop_xyxy']]
            yy,xx=torch.meshgrid(torch.arange(y0,y1),torch.arange(x0,x1),indexing='ij')
            indices=(yy*48+xx).flatten().cuda()
            for j,s in enumerate((.2,.4,.6,.8)):
                sigma=torch.tensor([s],dtype=target.dtype,device='cuda')
                noise=torch.randn(target.shape,dtype=target.dtype,generator=torch.Generator().manual_seed(62100+i*10+j)).cuda()
                noisy=(1-sigma)*target+sigma*noise
                features=capture(model,branch,backend,tensors,noisy,sigma,config)
                data={k:v.cpu().contiguous() for k,v in features.items()}
                for key in ('query','noise'):data[key]=data[key][:,indices.cpu()].contiguous()
                data.update(clean=target[:,:,y0:y1,x0:x1].cpu().contiguous(),
                            grid=torch.from_numpy(np.load(destination/f'grid_{i:02d}.npy')),
                            flow_target=(noise-target).flatten(2).transpose(1,2)[:,indices].cpu().contiguous())
                path=destination/'cases'/f'face_{i:02d}_{j}.safetensors'
                save_file(data,path)
                records.append({'file':str(path.relative_to(destination)),'sha256':file_hash(path),'row':i,'sigma':float(sigma)})
            print(f'Identity region cache {i+1}/19; {time.monotonic()-began:.0f}s',flush=True)
    write(destination/'cache_manifest.json',{'records':records,'geometry_sha256':file_hash(destination/'geometry.json'),
          'builder_sha256':file_hash(Path(__file__)),'seconds':time.monotonic()-began,**memory()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('align','cache'))
    p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();globals()[a.action](a.source.resolve(),a.destination.resolve())
