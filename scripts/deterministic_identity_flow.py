"""Verified continuation with deterministic face-alignment gradients."""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file, save_file

from ba_dit.config import ROOT, digest, load_config
from ba_dit.data.manifest import file_hash
from ba_dit.logging import log_metrics
from ba_dit.nn.deterministic_identity_loss import identity_loss
from scripts.identity_face_flow import make, experiment, auxiliary, install
from scripts.face_crop_flow import cases, evaluate, memory, prediction, write
from scripts.masked_face_flow import verify, infer, decode

EXTRA=('ba_dit/nn/deterministic_identity_loss.py','scripts/deterministic_identity_flow.py',
       'scripts/run_deterministic_identity_flow.py')


def initialize(run, source, step=500):
    config,old=verify(source)
    checkpoint=source/f'checkpoint-{step:06d}'
    previous=json.loads((checkpoint/'manifest.json').read_text())
    assert previous['identity']==old and previous['sha256']==file_hash(checkpoint/'branch.safetensors')
    assert json.loads((source/'best_checkpoint.json').read_text())['step']==step
    run.mkdir(parents=True,exist_ok=False)
    config['name']=run.name
    (run/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    load_config(run/'resolved_config.yaml')
    identity={**old,'config_sha256':digest(config),
        'continuation':{'source':str(source),'step':step,'identity_sha256':file_hash(source/'identity.json'),
            'checkpoint_sha256':previous['sha256'],'optimizer_sha256':file_hash(checkpoint/'optimizer.pt'),
            'reason':'grid_sample CUDA backward is nondeterministic; use equivalent deterministic bilinear gather',
            'optimizer_reset':False,'historical_validations':'imported unchanged with original checkpoint hashes'},
        'runtime_determinism':{'algorithms':True,'cudnn_deterministic':True,'cudnn_benchmark':False,
            'cublas_workspace_config':':4096:8','resume_check':'exact parameter equality'},
        'source_sha256':{p:file_hash(ROOT/p) for p in (*old['source_sha256'],*EXTRA)}}
    for name in ('initial.safetensors','cache_manifest.json','routing_masks.json','ownership_boxes.json','mask_overlays.png',
                 'initial_parent_image_parity.json','admission.json','first_gradients_1.json','first_gradients_2.json',
                 'training_summary.json','native_off_parity.json'):
        shutil.copyfile(source/name,run/name)
    for name in ('cases','native'):
        shutil.copytree(source/name,run/name,copy_function=os.link)
    for path in sorted(source.glob('validation-*')):
        if int(path.name.split('-')[1])<=step:
            shutil.copytree(path,run/path.name,copy_function=os.link)
    for prefix in ('background_audit_','inference_audit_'):
        for path in source.glob(prefix+'*.json'):
            if int(path.stem.split('_')[-1])<=step:shutil.copyfile(path,run/path.name)
    logs=[json.loads(x) for x in (source/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in logs if r['step']<=step]==list(range(1,step+1))
    (run/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in logs if r['step']<=step))
    write(run/'probes.json',[p for p in json.loads((source/'probes.json').read_text()) if p['step']<=step])
    write(run/'identity.json',identity)
    for count in (50,step):
        src=source/f'checkpoint-{count:06d}';dst=run/src.name;dst.mkdir()
        manifest=json.loads((src/'manifest.json').read_text())
        assert manifest['identity']==old and file_hash(src/'branch.safetensors')==manifest['sha256']
        for name in ('branch.safetensors','optimizer.pt'):os.link(src/name,dst/name)
        write(dst/'manifest.json',{'step':count,'identity':identity,'sha256':manifest['sha256'],
            'imported_from':str(src),'original_manifest_sha256':file_hash(src/'manifest.json')})
    for path in identity['source_sha256']:
        dst=run/'source_snapshot'/path;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,dst)
    write(run/'completed_commands.json',json.loads((source/'completed_commands.json').read_text()))
    write(run/'continuation_evidence.json',{'source':str(source),'original_identity':old,
        'original_status':json.loads((source/'status.json').read_text()),
        'discarded_uncommitted_steps':[r['step'] for r in logs if r['step']>step],
        'old_witnesses_preserved':[str(p) for p in source.glob('expected_update_50*.safetensors')]})
    exp=experiment(config,run)
    exp.log_parameters({'continued_from':str(source),'continuation_step':step,'optimizer_preserved':True,
        'deterministic_face_alignment':True,'historical_steps_imported':[0,500]})
    for folder in run.glob('validation-*'):
        score=json.loads((folder/'quality_summary.json').read_text())['metrics']
        exp.log_metrics({f'validation/{k}':v for k,v in score.items()},step=int(folder.name.split('-')[1]))
    for name in ('identity.json','continuation_evidence.json','resolved_config.yaml','admission.json'):exp.log_asset(str(run/name))
    exp.end()


def train(run, config, identity, resume, until, replay=False):
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(142)
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
    exp=None if replay else experiment(config,run)
    probes=[] if replay else json.loads((run/'probes.json').read_text())
    replay_checks=[]
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
                assert all(torch.equal(v.cpu(),check[n]) for n,v in b.state_dict().items()),max(errors.values())
                record={'step':step,'max_abs':max(errors.values()),'exact':True}
                replay_checks.append(record)
                write(run/'resume_check.json',record)
            elif replay:
                save_file({n:v.cpu().contiguous() for n,v in b.state_dict().items()},expected)
            if replay:
                continue
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
        if replay:
            write(run/'deterministic_replay.json',{'resume':resume,'until':until,'checks':replay_checks,**memory()})
            return
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
    finally:
        if exp:exp.end()


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('init','replay','train','infer','decode','score'))
    p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--step',type=int,default=2000);p.add_argument('--resume',type=int,default=500)
    a=p.parse_args();run=a.run.resolve()
    if a.action=='init':return initialize(run,a.source.resolve())
    config,identity=verify(run)
    if a.action in ('train','replay'):train(run,config,identity,a.resume,a.step,replay=a.action=='replay')
    elif a.action=='infer':infer(run,config,identity,a.step,branch_factory=lambda model:install(model,identity))
    elif a.action=='decode':decode(run,config,identity,a.step,experiment_factory=experiment)
    else:
        subprocess.run([str(ROOT/'envs/metrics/bin/python'),str(ROOT/'scripts/evaluate_metrics.py'),
            '--validation',str(run/f'validation-{a.step:06d}'),'--ownership-boxes',str(run/'ownership_boxes.json'),
            '--log-dir',str(run),'--global-step',str(a.step)],check=True)


if __name__=='__main__':main()
