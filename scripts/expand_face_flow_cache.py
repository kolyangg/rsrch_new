"""Add fresh full-backbone noise/sigma cases; preserve the original noise probe."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from ba_dit.config import load_config
from ba_dit.data.cache import load_pair
from ba_dit.data.manifest import read_manifest,file_hash
from ba_dit.face_diagnostic import target_face_tokens
from ba_dit.nn.face_crop_flow import install,capture
from ba_dit.runtime import backend_module
from scripts.masked_face_flow import with_fixed_reference
from scripts.face_crop_flow import write,memory


@torch.no_grad()
def expand(source,destination,seeds=4):
    config=load_config(source/'resolved_config.yaml')
    identity=json.loads((source/'identity.json').read_text())
    destination.mkdir(parents=True,exist_ok=True);(destination/'cases').mkdir(exist_ok=True)
    records=json.loads((source/'cache_manifest.json').read_text())['records']
    for row in records:
        assert file_hash(source/row['file'])==row['sha256']
        if not (destination/row['file']).exists():os.link(source/row['file'],destination/row['file'])
    backend=backend_module(config);model=backend.load_transformer(config);branch=install(model,True,True,False)
    reference,_=load_pair(config,identity['training_reference'],'cuda')
    rows=read_manifest(config['data']['train_manifest'],training=True)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for i,row in enumerate(rows):
        tensors=with_fixed_reference(config,row,reference);target=tensors['target_latent']
        weights=target_face_tokens(row,config).flatten()
        for seed in range(342,342+seeds):
            for bin_ in range(6):
                generator=torch.Generator().manual_seed(seed*10000+i*100+bin_)
                # Stratified continuous sigma, independent of the old probe seeds.
                sigma=torch.tensor([(bin_+torch.rand((),generator=generator).item())/6],dtype=target.dtype,device='cuda')
                noise=torch.randn(target.shape,dtype=target.dtype,generator=generator).cuda()
                noisy=(1-sigma)*target+sigma*noise
                indices=torch.multinomial(weights,64,replacement=True,generator=generator).cuda()
                path=destination/'cases'/f'fit_fresh_{i}_{seed}_{bin_}.safetensors'
                if not path.exists():
                    features=capture(model,branch,backend,tensors,noisy,sigma,config)
                    cached={k:v.cpu().contiguous() for k,v in features.items()}
                    for key in ('query','noise'):cached[key]=cached[key][:,indices.cpu()].contiguous()
                    cached.update(flow_target=(noise-target).flatten(2).transpose(1,2)[:,indices].cpu().contiguous(),
                                  query_indices=indices.cpu()[None],noisy=noisy.cpu(),sigma=sigma.cpu())
                    save_file(cached,path)
                records.append({'file':str(path.relative_to(destination)),'sha256':file_hash(path),
                                'split':'fit','row':i,'sigma':float(sigma),'noise_seed':seed})
        print(f'Expanded full-scene cache {i+1}/{len(rows)}; {time.monotonic()-started:.0f}s',flush=True)
    # Keep a probe record last for the existing exact cached/live replay check.
    records.sort(key=lambda r:r['split'])
    write(destination/'cache_manifest.json',{'records':records,'seconds':time.monotonic()-started,
        'source':str(source),'source_manifest_sha256':file_hash(source/'cache_manifest.json'),
        'builder_sha256':file_hash(Path(__file__)),**memory()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--seeds',type=int,default=4)
    a=p.parse_args();expand(a.source,a.destination,a.seeds)
