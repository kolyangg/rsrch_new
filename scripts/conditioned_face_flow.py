"""Controlled head-capacity experiments, sharing the prompted-scene protocol."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

from ba_dit.config import ROOT, digest, load_config
from ba_dit.data.manifest import file_hash
from ba_dit.logging import connect, log_metrics
from ba_dit.nn.face_crop_flow import FaceCropFlow
from ba_dit.nn.conditioned_face_flow import ConditionedFaceFlow
from ba_dit.nn.reference_refiner_flow import ReferenceRefinerFlow
from scripts.face_crop_flow import cases, evaluate, memory, prediction, write
from scripts.masked_face_flow import SOURCES, verify, infer, decode

DONOR = ROOT/'runs/flux4b_masked_face_flow_24_stable_20261001'
EXTRA = ('scripts/conditioned_face_flow.py', 'ba_dit/nn/conditioned_face_flow.py',
         'ba_dit/nn/reference_refiner_flow.py', 'scripts/run_reference_refiner.py',
         'scripts/benchmark_reference_refiner.py')


def make(identity):
    settings = identity['head']
    if settings['kind'] == 'refiner':
        return ReferenceRefinerFlow(hidden=settings['width'], heads=settings['heads'])
    if settings['kind'] == 'conditioned':
        return ConditionedFaceFlow(hidden=settings['width'], heads=settings['heads'])
    return FaceCropFlow(hidden=settings['width'], heads=settings['heads'],
                        query_residual=True, noise_skip=True, timestep_scaling=False)


def experiment(config, run):
    identity = json.loads((run/'identity.json').read_text())
    records = json.loads((run/'cache_manifest.json').read_text())['records']
    exp = connect(config, run, name=run.name)
    exp.log_parameters({'model_identity': digest(identity), 'head': identity['head'],
        'BA_width': identity['head']['width'], 'BA_heads': identity['head']['heads'],
        'native_attention_lora_enabled': False, 'native_velocity_used': False,
        'trainable_parameters': identity['trainable_parameters'], 'training/batch_size': identity.get('optimizer_batch_size',8),
        'cache_fit_cases': sum(r['split']=='fit' for r in records),
        'cache_probe_cases': sum(r['split']=='probe' for r in records), 'validation_samples': 24,
        'face_cfg': 1., 'background_cfg': 4., 'validation_targets': False,
        'live_metric_stride': 100, 'identity_loss': False,
        'frozen_BA_parameters': identity.get('frozen_BA_parameters',0),
        'initialization': identity.get('initialization','zero face velocity'),
        'ablation_semantics': identity.get('ablation','reference attention read off')})
    return exp


def initialize(run, kind, width, lr, total, weight_decay, cache=None, core_checkpoint=None, batch_size=8):
    parent = json.loads((DONOR/'identity.json').read_text())
    cache = DONOR if cache is None else cache.resolve()
    manifest = json.loads((cache/'cache_manifest.json').read_text())
    if cache != DONOR:
        assert manifest['source_manifest_sha256'] == file_hash(DONOR/'cache_manifest.json')
    config = load_config(DONOR/'resolved_config.yaml')
    config['name'] = run.name
    if batch_size<1:
        raise ValueError('Batch size must be positive')
    config['training'].update(steps=total, lr=lr, weight_decay=weight_decay,
                              checkpoint_every=2000, validation_every=2000)
    run.mkdir(parents=True, exist_ok=False)
    (run/'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    load_config(run/'resolved_config.yaml')
    for name in ('routing_masks.json','ownership_boxes.json','mask_overlays.png'):
        shutil.copyfile(DONOR/name,run/name)
    shutil.copyfile(cache/'cache_manifest.json',run/'cache_manifest.json')
    shutil.copytree(DONOR/'native',run/'native',copy_function=os.link)
    (run/'cases').mkdir()
    for record in manifest['records']:
        assert file_hash(cache/record['file'])==record['sha256']
        os.link(cache/record['file'],run/record['file'])
    sources = (*SOURCES,*EXTRA,'scripts/expand_face_flow_cache.py')
    identity = {**parent, 'variant': 'masked_face_flow_capacity_v2' if cache==DONOR else 'masked_face_flow_diverse_noise_v3',
        'config_sha256': digest(config),
        'head': {'kind': kind, 'width': width, 'heads': width//64, 'weight_decay': weight_decay},
        'source_sha256': {p:file_hash(ROOT/p) for p in sources},
        'cache_source': {'run':str(cache), 'manifest_sha256':file_hash(cache/'cache_manifest.json')},
        'probe_every': 1000, 'live_metric_every': 100,'optimizer_batch_size':batch_size,
        'frozen_core_cache':kind=='refiner','batch_gather':'stacked CUDA tensors'}
    if cache != DONOR:
        identity['sigmas']='original six values plus fresh stratified continuous sigmas; see cache manifest'
        identity['comparison']='bundled architecture, noise diversity and optimizer experiment'
    torch.manual_seed(142)
    branch = make(identity)
    if kind == 'refiner':
        if core_checkpoint is None:
            raise ValueError('Reference refiner requires a verified trained core checkpoint')
        core_checkpoint=core_checkpoint.resolve()
        manifest=json.loads((core_checkpoint.parent/'manifest.json').read_text())
        assert file_hash(core_checkpoint)==manifest['sha256']
        assert manifest['identity']['head']['kind']=='conditioned' and manifest['identity']['head']['width']==512
        core_source='ba_dit/nn/conditioned_face_flow.py'
        assert manifest['identity']['source_sha256'][core_source]==file_hash(ROOT/core_source)
        for field in ('base','train_manifest_sha256','validation_manifest_sha256','routing_masks_sha256'):
            assert manifest['identity'][field]==identity[field],field
        branch.core.load_state_dict(load_file(core_checkpoint),strict=True)
        identity.update(variant='masked_face_reference_refiner_v4',initial_flow_is_zero=False,
            initialization='frozen trained 512-wide BA core plus zero-output 1024-wide reference refiner',
            core_checkpoint={'path':str(core_checkpoint),'sha256':file_hash(core_checkpoint),
                             'manifest_sha256':file_hash(core_checkpoint.parent/'manifest.json'),'step':manifest['step']},
            ablation='new refiner off; frozen core retains its own reference conditioning')
    identity['trainable_parameters'] = sum(p.numel() for p in branch.parameters() if p.requires_grad)
    identity['frozen_BA_parameters'] = sum(p.numel() for p in branch.parameters() if not p.requires_grad)
    save_file(branch.state_dict(),run/'initial.safetensors')
    identity['initial_sha256'] = file_hash(run/'initial.safetensors')
    write(run/'identity.json',identity)
    for p in identity['source_sha256']:
        dst=run/'source_snapshot'/p; dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,dst)
    exp=experiment(config,run)
    for name in ('identity.json','resolved_config.yaml','cache_manifest.json'):exp.log_asset(str(run/name))
    exp.end()


def train(run, config, identity, resume, until):
    b=make(identity).cuda()
    path=run/f'checkpoint-{resume:06d}/branch.safetensors' if resume else run/'initial.safetensors'
    if resume:
        manifest=json.loads((path.parent/'manifest.json').read_text())
        assert manifest['identity']==identity and manifest['sha256']==file_hash(path)
    b.load_state_dict(load_file(path))
    named_trainable={n:p for n,p in b.named_parameters() if p.requires_grad}
    parameters=list(named_trainable.values())
    optimizer=torch.optim.AdamW(parameters,lr=config['training']['lr'],
                              weight_decay=identity['head']['weight_decay'],fused=True)
    if resume:optimizer.load_state_dict(torch.load(path.parent/'optimizer.pt',weights_only=True))
    groups=cases(run)
    # The core is frozen and each input is fixed: compute its velocities once.
    if identity['head']['kind']=='refiner':
        with torch.no_grad():
            for group in groups.values():
                for case in group:
                    case['core_velocity']=prediction(b,case,reference_read=False)
    fit={k:torch.cat([c[k] for c in groups['fit']]) for k in groups['fit'][0]}
    # Views keep probe evaluation unchanged without storing a second fit cache.
    groups['fit']=[{k:v[i:i+1] for k,v in fit.items()} for i in range(len(groups['fit']))]
    order=torch.randint(len(groups['fit']),(config['training']['steps'],identity.get('optimizer_batch_size',8)),
                        generator=torch.Generator().manual_seed(142)).cuda()
    exp=experiment(config,run)
    probes=json.loads((run/'probes.json').read_text()) if resume else [{'step':0,**evaluate(b,groups)}]
    torch.cuda.reset_peak_memory_stats()
    def update(step):
        batch={k:v.index_select(0,order[step-1]) for k,v in fit.items()}
        for group in optimizer.param_groups:
            group['lr']=config['training']['lr']*min(1.,step/100)
        optimizer.zero_grad(set_to_none=True)
        loss=(prediction(b,batch)-batch['flow_target'].float()).square().mean()
        loss.backward()
        assert all(p.grad is not None for p in parameters)
        if step in (1,2):
            write(run/f'first_gradients_{step}.json',{n:float(p.grad.norm()) for n,p in named_trainable.items()})
            if step==2: assert all(p.grad.count_nonzero() for p in parameters)
        # The global norm is finite iff every gradient element is finite. This
        # avoids a GPU/CPU synchronization for every parameter on every update.
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        assert torch.isfinite(loss)
        optimizer.step()
        return float(loss.detach()),float(norm)
    try:
        if not resume:exp.log_metrics(probes[0],step=0)
        for step in range(resume+1,until+1):
            began=time.monotonic();loss,norm=update(step);torch.cuda.synchronize()
            expected=run/f'expected_update_{step}.safetensors'
            if expected.exists():
                check=load_file(expected)
                errors={n:float((v.cpu()-check[n]).abs().max()) for n,v in b.state_dict().items()}
                # Different CUDA allocation layouts may change FP32 GEMM by one ULP.
                assert all(torch.allclose(v.cpu(),check[n],atol=1e-8,rtol=1e-6) for n,v in b.state_dict().items())
                write(run/'resume_check.json',{'step':step,'fresh_process_exact':max(errors.values())==0,
                    'max_abs_by_tensor':errors,'atol':1e-8,'rtol':1e-6})
            log_metrics(exp if step==1 or step%100==0 else None,run,
                {'train/loss':loss,'train/gradient_norm':norm,'train/seconds':time.monotonic()-began,**memory()},step)
            if step%1000==0 or step==until:
                probes.append({'step':step,**evaluate(b,groups)});write(run/'probes.json',probes)
                exp.log_metrics(probes[-1],step=step)
            if step==until or step%2000==0:
                folder=run/f'checkpoint-{step:06d}';folder.mkdir()
                save_file({n:v.cpu().contiguous() for n,v in b.state_dict().items()},folder/'branch.safetensors')
                torch.save(optimizer.state_dict(),folder/'optimizer.pt')
                write(folder/'manifest.json',{'step':step,'identity':identity,'sha256':file_hash(folder/'branch.safetensors')})
        initial=load_file(run/'initial.safetensors')
        updates={n:float((p.cpu()-initial[n]).norm()) for n,p in named_trainable.items()}
        assert all(updates.values())
        frozen={n:torch.equal(p.cpu(),initial[n]) for n,p in b.named_parameters() if not p.requires_grad}
        assert all(frozen.values())
        write(run/'training_summary.json',{'steps':until,'all_gradients_finite':True,'updates':updates,
              'frozen_parameters_exact':frozen,**memory()})
        if until<config['training']['steps']:
            update(until+1)
            save_file({n:v.cpu().contiguous() for n,v in b.state_dict().items()},run/f'expected_update_{until+1}.safetensors')
    finally:exp.end()


def install(model,identity):
    model.requires_grad_(False)
    assert not any(hasattr(b,'reference_branch') for b in (*model.double_blocks,*model.single_blocks))
    branch=make(identity).to(model.device)
    model.single_blocks[-1].reference_branch=branch
    assert all('reference_branch.' in n for n,p in model.named_parameters() if p.requires_grad)
    return branch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('init','train','infer','decode','score'))
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--kind',choices=('original','conditioned','refiner'),default='conditioned')
    p.add_argument('--width',type=int,default=512)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--weight-decay',type=float,default=0.)
    p.add_argument('--total',type=int,default=10000)
    p.add_argument('--step',type=int,default=2000)
    p.add_argument('--resume',type=int,default=0)
    p.add_argument('--cache',type=Path)
    p.add_argument('--core-checkpoint',type=Path)
    p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args();run=args.run.resolve()
    if args.action=='init':return initialize(run,args.kind,args.width,args.lr,args.total,args.weight_decay,args.cache,args.core_checkpoint,args.batch_size)
    config,identity=verify(run)
    if args.action=='train':train(run,config,identity,args.resume,args.step)
    elif args.action=='infer':infer(run,config,identity,args.step,branch_factory=lambda model:install(model,identity))
    elif args.action=='decode':decode(run,config,identity,args.step,experiment_factory=experiment)
    else:
        subprocess.run([str(ROOT/'envs/metrics/bin/python'),str(ROOT/'scripts/evaluate_metrics.py'),
            '--validation',str(run/f'validation-{args.step:06d}'),'--ownership-boxes',str(run/'ownership_boxes.json'),
            '--log-dir',str(run),'--global-step',str(args.step)],check=True)

if __name__=='__main__':main()
