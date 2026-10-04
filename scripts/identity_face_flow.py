"""Deeper reference-only face flow, with training-photo ArcFace supervision."""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file, save_file

from ba_dit.config import ROOT, digest, load_config
from ba_dit.data.manifest import file_hash
from ba_dit.logging import connect, log_metrics
from ba_dit.nn.arcface_identity import FrozenOnnxArcFace
from ba_dit.nn.deep_identity_flow import DeepIdentityFlow
from ba_dit.nn.face_identity_loss import identity_loss
from ba_dit.nn.reference_refiner_flow import ReferenceRefinerFlow
from ba_dit.runtime import backend_module
from scripts.face_crop_flow import cases, evaluate, memory, prediction, write
from scripts.masked_face_flow import verify, infer, decode

NEW_SOURCES=('ba_dit/nn/deep_identity_flow.py','ba_dit/nn/arcface_identity.py',
             'ba_dit/nn/face_identity_loss.py','scripts/prepare_identity_flow.py',
             'scripts/identity_face_flow.py','scripts/run_identity_flow.py',
             'scripts/check_identity_flow.py')


def make(identity):
    return DeepIdentityFlow(hidden=identity['head']['width'], heads=identity['head']['heads'],
                            depth=identity['head']['extra_reference_reads'])


def experiment(config, run):
    identity=json.loads((run/'identity.json').read_text())
    exp=connect(config,run,name=run.name)
    exp.log_parameters({'head':identity['head'],'model_identity':digest(identity),
        'trainable_parameters':identity['trainable_parameters'],
        'frozen_BA_parameters':identity['frozen_BA_parameters'],
        'training/batch_size':identity['optimizer_batch_size'],
        'identity_loss':identity['identity_objective'],'native_velocity_used':False,
        'validation_samples':24,'validation_targets':False,'trainable_backbone_parameters':0,
        'ablation_semantics':'all trainable BA refiner reads off; frozen 512 core still reads reference'})
    return exp


def initialize(run, parent, aux, batch_size, id_weight, lr, total):
    config,parent_identity=verify(parent)
    best=json.loads((parent/'best_checkpoint.json').read_text())
    checkpoint=Path(best['file']);assert file_hash(checkpoint)==best['sha256']
    assert parent_identity['head']['kind']=='refiner'
    aux_manifest=json.loads((aux/'cache_manifest.json').read_text())
    geometry=json.loads((aux/'geometry.json').read_text())
    assert geometry['train_manifest_sha256']==parent_identity['train_manifest_sha256']
    assert aux_manifest['geometry_sha256']==file_hash(aux/'geometry.json')
    assert aux_manifest['builder_sha256']==file_hash(ROOT/'scripts/prepare_identity_flow.py')
    assert file_hash(geometry['arcface_path'])==geometry['arcface_sha256']
    admission=json.loads((aux/'admission.json').read_text())
    assert id_weight==admission['suggested_identity_weight']
    measured=next(r for r in admission['benchmarks'] if r['batch_size']==batch_size)
    assert measured['hardware/reserved_fraction']<.9
    run.mkdir(parents=True,exist_ok=False)
    config['name']=run.name
    config['training'].update(lr=lr,steps=total,weight_decay=.01,checkpoint_every=2000,validation_every=2000)
    (run/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    load_config(run/'resolved_config.yaml')
    for name in ('routing_masks.json','ownership_boxes.json','mask_overlays.png','cache_manifest.json'):
        shutil.copyfile(parent/name,run/name)
    for name in ('native','cases'):
        shutil.copytree(parent/name,run/name,copy_function=os.link)
    identity={**parent_identity,'variant':'masked_deep_identity_flow_v5',
        'config_sha256':digest(config),'optimizer_batch_size':batch_size,
        'head':{'kind':'deep_identity','width':1024,'heads':16,'extra_reference_reads':2,'weight_decay':.01},
        'initialization':'best refiner step 500; two new zero-output reference-read blocks',
        'parent_checkpoint':{'path':str(checkpoint),'sha256':best['sha256'],'step':best['step'],'id_sim':best['id_sim']},
        'identity_objective':{'weight':id_weight,'every':2,'batch_size':1,'loss':'1 - ArcFace cosine to fixed training reference',
            'alignment':'training-photo five landmarks','clean_estimate':'noisy - sigma * velocity',
            'decoder':'frozen FLUX2 VAE; cropped packed latent with 48px context',
            'cache':str(aux),'cache_manifest_sha256':file_hash(aux/'cache_manifest.json'),
            'arcface_path':geometry['arcface_path'],'arcface_sha256':geometry['arcface_sha256'],
            'reference_embedding_sha256':file_hash(aux/'reference_embedding.npy'),
            'validation_data_used':False,'optimizer_reset':True,'cudnn_tf32':False,
            'admission_sha256':file_hash(aux/'admission.json')},
        'loss':'native sampled-face-token flow MSE + periodic decoded face identity loss',
        'source_sha256':{p:file_hash(ROOT/p) for p in (*parent_identity['source_sha256'],*NEW_SOURCES)}}
    torch.manual_seed(142)
    branch=make(identity)
    missing,unexpected=branch.load_state_dict(load_file(checkpoint),strict=False)
    assert missing and all(n.startswith('reads.') for n in missing) and not unexpected
    identity['trainable_parameters']=sum(p.numel() for p in branch.parameters() if p.requires_grad)
    identity['frozen_BA_parameters']=sum(p.numel() for p in branch.parameters() if not p.requires_grad)
    save_file(branch.state_dict(),run/'initial.safetensors')
    identity['initial_sha256']=file_hash(run/'initial.safetensors')
    write(run/'identity.json',identity)
    shutil.copyfile(aux/'admission.json',run/'admission.json')
    for p in identity['source_sha256']:
        dst=run/'source_snapshot'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,dst)
    exp=experiment(config,run)
    for name in ('identity.json','resolved_config.yaml','cache_manifest.json','admission.json'):exp.log_asset(str(run/name))
    exp.end()


def auxiliary(config, identity):
    spec=identity['identity_objective'];source=Path(spec['cache'])
    assert file_hash(source/'cache_manifest.json')==spec['cache_manifest_sha256']
    assert file_hash(source/'reference_embedding.npy')==spec['reference_embedding_sha256']
    records=json.loads((source/'cache_manifest.json').read_text())['records']
    data=[]
    for record in records:
        path=source/record['file'];assert file_hash(path)==record['sha256']
        data.append(load_file(path))
    vae=backend_module(config).load_vae(config)
    recognizer=FrozenOnnxArcFace(spec['arcface_path'],expected_sha256=spec['arcface_sha256']).cuda()
    embedding=torch.from_numpy(np.load(source/'reference_embedding.npy')).cuda()
    assert not any(p.requires_grad for p in (*vae.parameters(),*recognizer.parameters()))
    return data,vae,recognizer,embedding


def train(run, config, identity, resume, until):
    torch.set_num_threads(8)
    torch.backends.cudnn.allow_tf32=False
    b=make(identity).cuda()
    path=run/f'checkpoint-{resume:06d}/branch.safetensors' if resume else run/'initial.safetensors'
    if resume:
        manifest=json.loads((path.parent/'manifest.json').read_text())
        assert manifest['identity']==identity and manifest['sha256']==file_hash(path)
    b.load_state_dict(load_file(path),strict=True)
    named={n:p for n,p in b.named_parameters() if p.requires_grad};parameters=list(named.values())
    optimizer=torch.optim.AdamW(parameters,lr=config['training']['lr'],weight_decay=.01,fused=True)
    if resume:optimizer.load_state_dict(torch.load(path.parent/'optimizer.pt',weights_only=True))
    groups=cases(run)
    with torch.no_grad():
        for group in groups.values():
            for case in group:case['core_velocity']=prediction(b,case,False)
    fit={k:torch.cat([c[k] for c in groups['fit']]) for k in groups['fit'][0]}
    groups['fit']=[{k:v[i:i+1] for k,v in fit.items()} for i in range(len(groups['fit']))]
    data,vae,recognizer,embedding=auxiliary(config,identity)
    with torch.no_grad():
        for case in data:
            case['core_velocity']=prediction(b,{k:v.cuda() for k,v in case.items()},False).cpu()
    order=torch.randint(len(groups['fit']),(config['training']['steps'],identity['optimizer_batch_size']),
                        generator=torch.Generator().manual_seed(142)).cuda()
    aux_order=torch.randperm(len(data),generator=torch.Generator().manual_seed(6142)).tolist()
    spec=identity['identity_objective']
    exp=experiment(config,run)
    probes=json.loads((run/'probes.json').read_text()) if resume else [{'step':0,**evaluate(b,groups)}]
    torch.cuda.reset_peak_memory_stats()

    def update(step):
        for group in optimizer.param_groups:group['lr']=config['training']['lr']*min(1.,step/100)
        optimizer.zero_grad(set_to_none=True)
        batch={k:v.index_select(0,order[step-1]) for k,v in fit.items()}
        flow=(prediction(b,batch)-batch['flow_target'].float()).square().mean()
        flow.backward()
        value=0.;weighted=0.
        if step%spec['every']==0:
            index=aux_order[(step//spec['every']-1)%len(data)]
            case={k:v.cuda() for k,v in data[index].items()}
            ident=identity_loss(vae,recognizer,prediction(b,case),case,embedding)
            (spec['weight']*ident).backward()
            value=float(ident.detach());weighted=spec['weight']*value
        assert all(p.grad is not None for p in parameters)
        if step in (1,2):
            gradients={n:float(p.grad.norm()) for n,p in named.items()}
            write(run/f'first_gradients_{step}.json',gradients)
            if step==2:assert all(gradients.values()),'Some branch parameters have no gradient'
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        assert torch.isfinite(flow) and np.isfinite(value)
        optimizer.step()
        return {'train/loss':float(flow.detach())+weighted,'train/flow_mse':float(flow.detach()),
                'train/identity_loss':value,'train/identity_active':int(step%spec['every']==0),
                'train/gradient_norm':float(norm)}

    try:
        for step in range(resume+1,until+1):
            began=time.monotonic();metrics=update(step);torch.cuda.synchronize()
            expected=run/f'expected_update_{step}.safetensors'
            if expected.exists():
                check=load_file(expected)
                errors={n:float((v.cpu()-check[n]).abs().max()) for n,v in b.state_dict().items()}
                assert all(torch.allclose(v.cpu(),check[n],atol=2e-7,rtol=2e-5) for n,v in b.state_dict().items()),max(errors.values())
                write(run/'resume_check.json',{'step':step,'max_abs':max(errors.values()),'atol':2e-7,'rtol':2e-5})
            log_metrics(exp if step==1 or step%50==0 else None,run,
                        {**metrics,'train/seconds':time.monotonic()-began,**memory()},step)
            if step%1000==0 or step==until:
                probes.append({'step':step,**evaluate(b,groups)});write(run/'probes.json',probes)
                exp.log_metrics(probes[-1],step=step)
            if step==until or step%2000==0:
                folder=run/f'checkpoint-{step:06d}';folder.mkdir()
                save_file({n:v.cpu().contiguous() for n,v in b.state_dict().items()},folder/'branch.safetensors')
                torch.save(optimizer.state_dict(),folder/'optimizer.pt')
                write(folder/'manifest.json',{'step':step,'identity':identity,'sha256':file_hash(folder/'branch.safetensors')})
        initial=load_file(run/'initial.safetensors')
        updates={n:float((p.cpu()-initial[n]).norm()) for n,p in named.items()}
        assert all(updates.values())
        frozen={n:torch.equal(p.cpu(),initial[n]) for n,p in b.named_parameters() if not p.requires_grad}
        assert all(frozen.values())
        write(run/'training_summary.json',{'steps':until,'all_gradients_finite':True,'updates':updates,
              'frozen_parameters_exact':frozen,**memory()})
        if until<config['training']['steps']:
            # Verify two resumed updates, including one with the identity path.
            for step in (until+1,until+2):
                update(step)
                save_file({n:v.cpu().contiguous() for n,v in b.state_dict().items()},run/f'expected_update_{step}.safetensors')
    finally:exp.end()


def install(model,identity):
    model.requires_grad_(False)
    assert not any(hasattr(b,'reference_branch') for b in (*model.double_blocks,*model.single_blocks))
    branch=make(identity).to(model.device);model.single_blocks[-1].reference_branch=branch
    assert all('reference_branch.' in n for n,p in model.named_parameters() if p.requires_grad)
    return branch


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('init','train','infer','decode','score'))
    p.add_argument('--run',type=Path,required=True);p.add_argument('--parent',type=Path)
    p.add_argument('--aux',type=Path);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--id-weight',type=float,default=.1);p.add_argument('--lr',type=float,default=.00005)
    p.add_argument('--total',type=int,default=100000);p.add_argument('--resume',type=int,default=0)
    p.add_argument('--step',type=int,default=500)
    a=p.parse_args();run=a.run.resolve()
    if a.action=='init':return initialize(run,a.parent.resolve(),a.aux.resolve(),a.batch_size,a.id_weight,a.lr,a.total)
    config,identity=verify(run)
    if a.action=='train':train(run,config,identity,a.resume,a.step)
    elif a.action=='infer':infer(run,config,identity,a.step,branch_factory=lambda model:install(model,identity))
    elif a.action=='decode':decode(run,config,identity,a.step,experiment_factory=experiment)
    else:
        subprocess.run([str(ROOT/'envs/metrics/bin/python'),str(ROOT/'scripts/evaluate_metrics.py'),
            '--validation',str(run/f'validation-{a.step:06d}'),'--ownership-boxes',str(run/'ownership_boxes.json'),
            '--log-dir',str(run),'--global-step',str(a.step)],check=True)


if __name__=='__main__':main()
