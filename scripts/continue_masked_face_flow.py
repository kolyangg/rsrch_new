"""Continue verified masked-face BA weights and Adam state in a new run."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from safetensors.torch import load_file, save_file

from ba_dit.config import ROOT, digest, load_config
from ba_dit.data.manifest import file_hash, read_manifest
from ba_dit.nn.face_crop_flow import FaceCropFlow
from scripts.face_crop_flow import cases, experiment, prediction, write
from scripts.masked_face_flow import SOURCES, verify, infer, decode
from scripts.train_masked_face_flow import train

STEPS=(2000,4000,6000,8000,10000)


def schedule(start,target,interval):
    if not 0 < start < target or interval <= 0 or target % interval:
        raise ValueError('Require 0 < parent step < target, with target divisible by validation interval')
    return (start,*range((start//interval+1)*interval,target+1,interval))


def initialize(run,parent,start=2000,target=10000,interval=2000):
    global STEPS
    STEPS=schedule(start,target,interval)
    previous=json.loads((parent/'identity.json').read_text())
    source_config=load_config(parent/'resolved_config.yaml')
    assert previous['config_sha256']==digest(source_config)
    assert all(file_hash(parent/'source_snapshot'/p)==h for p,h in previous['source_sha256'].items())
    original=parent/f'checkpoint-{start:06d}'
    manifest=json.loads((original/'manifest.json').read_text())
    assert manifest['identity']==previous and manifest['sha256']==file_hash(original/'branch.safetensors')
    assert json.loads((parent/'training_summary.json').read_text())['steps']==start
    # Only checkpoint/resume bookkeeping and probe/log frequency changed.
    for p in SOURCES:
        if p!='scripts/train_masked_face_flow.py': assert file_hash(ROOT/p)==previous['source_sha256'][p]
    config=json.loads(json.dumps(source_config))
    config['name']=f'flux4b_masked_face_flow_one_id_24_{target//1000}k'
    config['training'].update(steps=target,checkpoint_every=interval,validation_every=interval)
    run.mkdir(parents=True,exist_ok=False)
    (run/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    load_config(run/'resolved_config.yaml')  # Reject schema errors before creating a Comet run.
    for name in ('initial.safetensors','routing_masks.json','ownership_boxes.json','mask_overlays.png',
                 'native_off_parity.json','cache_manifest.json','probes.json','metrics.jsonl'):
        shutil.copyfile(parent/name,run/name)
    for name in ('cases','native',f'validation-{start:06d}'):
        shutil.copytree(parent/name,run/name,copy_function=os.link)
    for record in json.loads((run/'cache_manifest.json').read_text())['records']:
        assert file_hash(run/record['file'])==record['sha256']
    lineage={'parent_run':str(parent),'parent_comet_key':json.loads((parent/'comet_experiment.json').read_text())['experiment_key'],
             'inherited_step':start,'additional_updates':target-start,'target_step':target,
             'parent_identity_sha256':file_hash(parent/'identity.json'),
             'parent_manifest_sha256':file_hash(original/'manifest.json'),
             'branch_sha256':file_hash(original/'branch.safetensors'),'optimizer_sha256':file_hash(original/'optimizer.pt'),
             'inherited_metrics_through_step':start,'inherited_validation_steps':[start]}
    identity={**previous,'config_sha256':digest(config),'continuation':lineage,
              'source_sha256':{p:file_hash(ROOT/p) for p in (*SOURCES,'scripts/continue_masked_face_flow.py')}}
    if target >= 100000: identity.update(probe_every=1000,live_metric_every=100)
    write(run/'identity.json',identity); write(run/'lineage.json',lineage)
    checkpoint=run/f'checkpoint-{start:06d}'; checkpoint.mkdir()
    for name in ('branch.safetensors','optimizer.pt'): shutil.copyfile(original/name,checkpoint/name)
    write(checkpoint/'manifest.json',{'step':start,'identity':identity,'sha256':lineage['branch_sha256'],
                                    'optimizer_sha256':lineage['optimizer_sha256'],'inherited_from':str(original)})
    for source in identity['source_sha256']:
        path=run/'source_snapshot'/source; path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/source,path)
    size=len([r for r in json.loads((run/'cache_manifest.json').read_text())['records'] if r['split']=='fit'])
    prefix=torch.randint(size,(start,8),generator=torch.Generator().manual_seed(config['training']['seed']))
    extended=torch.randint(size,(target,8),generator=torch.Generator().manual_seed(config['training']['seed']))
    assert torch.equal(prefix,extended[:start])
    write(run/'batch_order_check.json',{'prefix_steps':start,'prefix_identical':True,'next_indices':extended[start].tolist()})
    exp=experiment(config,run)
    exp.log_parameters({'continuation_from_step':start,'additional_optimizer_updates':target-start,'target_optimizer_step':target,
                        'parent_comet_key':lineage['parent_comet_key'],'validation_steps':list(STEPS),
                        'training/lr':config['training']['lr'],'data_and_noise_cache_unchanged':True,
                        'inherited_metrics_through_step':start,'live_metric_stride':identity.get('live_metric_every',25),
                        'cached_probe_every':identity.get('probe_every',250)})
    baseline=json.loads((run/f'validation-{start:06d}/quality_summary.json').read_text())['metrics']
    exp.log_metrics({'validation/'+k:v for k,v in baseline.items()},step=start)
    exp.log_metrics({k:v for k,v in json.loads((run/'probes.json').read_text())[-1].items() if k!='step'},step=start)
    for name in ('lineage.json','identity.json','batch_order_check.json','resolved_config.yaml'): exp.log_asset(str(run/name))
    for row in json.loads((run/f'validation-{start:06d}/validation.json').read_text())['samples']:
        exp.log_image(str(run/f'validation-{start:06d}'/row['image']),name='prompted/'+row['sample_id'],step=start)
    exp.end()


def resume_probe(run,config,identity):
    start=identity['continuation']['inherited_step']
    path=run/f'checkpoint-{start:06d}'
    assert file_hash(path/'optimizer.pt')==identity['continuation']['optimizer_sha256']
    branch=FaceCropFlow(query_residual=True,noise_skip=True,timestep_scaling=False).cuda()
    branch.load_state_dict(load_file(path/'branch.safetensors'))
    optimizer=torch.optim.AdamW(branch.parameters(),lr=config['training']['lr'],weight_decay=0.,fused=True)
    optimizer.load_state_dict(torch.load(path/'optimizer.pt',map_location='cpu',weights_only=True))
    groups=cases(run)
    indices=json.loads((run/'batch_order_check.json').read_text())['next_indices']
    batch={k:torch.cat([groups['fit'][i][k] for i in indices]) for k in groups['fit'][0]}
    loss=(prediction(branch,batch)-batch['flow_target'].float()).square().mean()
    loss.backward(); torch.nn.utils.clip_grad_norm_(branch.parameters(),1.); optimizer.step()
    save_file({n:p.cpu().contiguous() for n,p in branch.state_dict().items()},run/f'expected_update_{start+1}.safetensors')


def score(run,config,identity,step):
    subprocess.run([str(ROOT/'envs/metrics/bin/python'),str(ROOT/'scripts/evaluate_metrics.py'),
        '--validation',str(run/f'validation-{step:06d}'),'--ownership-boxes',str(run/'ownership_boxes.json'),
        '--log-dir',str(run),'--global-step',str(step)],check=True)
    summary={str(s):json.loads((run/f'validation-{s:06d}/quality_summary.json').read_text())['metrics']
             for s in STEPS if (run/f'validation-{s:06d}/quality_summary.json').exists()}
    write(run/'metric_summary.json',summary)
    print('Scored checkpoint',step,json.dumps(summary[str(step)]),flush=True)


def review(run,config,identity):
    start,target=STEPS[0],STEPS[-1]
    summary=json.loads((run/'metric_summary.json').read_text())
    assert set(summary)=={str(s) for s in STEPS}
    logs=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in logs]==list(range(1,target+1))
    assert all(np.isfinite(r['train/loss']) and np.isfinite(r['train/gradient_norm']) for r in logs)
    baseline=load_file(run/f'checkpoint-{start:06d}/branch.safetensors')
    final=load_file(run/f'checkpoint-{target:06d}/branch.safetensors')
    updates={n:float((v-baseline[n]).norm()) for n,v in final.items()}
    assert all(updates.values()) and all(torch.isfinite(v).all() for v in final.values())
    checks=[c for step in STEPS[1:] for c in json.loads((run/f'background_audit_{step}.json').read_text())]
    assert len(checks)==24*(len(STEPS)-1) and max(c['background_max_abs'] for c in checks)==0
    for step in STEPS[1:]:
        audit=json.loads((run/f'inference_audit_{step}.json').read_text())
        assert all(x['exact'] for x in audit['parity'])
    final_audit={'total_steps':target,'additional_steps':target-start,'all_losses_gradients_finite':True,'all_BA_tensors_updated':updates,
                 'new_images':len(checks),'exterior_max_pixel_error':0,'resume':json.loads((run/'resume_check.json').read_text()),
                 'additional_optimizer_seconds':sum(r['train/seconds'] for r in logs if r['step']>start),
                 'peak_train_reserved_gib':max(r['hardware/peak_reserved_gib'] for r in logs),
                 'peak_inference_reserved_gib':max(json.loads((run/f'inference_audit_{s}.json').read_text())['hardware/peak_reserved_gib'] for s in STEPS[1:])}
    best=max(STEPS,key=lambda s:summary[str(s)]['id_sim'])
    final_audit.update(best_id_checkpoint=best,best_id_similarity=summary[str(best)]['id_sim'],
                       final_id_similarity=summary[str(target)]['id_sim'],
                       final_minus_start_id=summary[str(target)]['id_sim']-summary[str(start)]['id_sim'],
                       final_minus_start_clip=summary[str(target)]['text_sim']-summary[str(start)]['text_sim'])
    write(run/'final_audit.json',final_audit)
    rows=read_manifest(config['data']['validation_manifest'])
    artifacts=[]
    for page in range(3):
        sheet=Image.new('RGB',((len(STEPS)+1)*224,8*248+28),'white'); draw=ImageDraw.Draw(sheet)
        labels=('Native',*(str(s) for s in STEPS))
        for col,label in enumerate(labels): draw.text((col*224+4,4),label,fill='black')
        for i,row in enumerate(rows[page*8:page*8+8]):
            for col,folder in enumerate((run/'native',*(run/f'validation-{s:06d}' for s in STEPS))):
                image=Image.open(folder/(row['sample_id']+'.png')); image.thumbnail((224,224))
                sheet.paste(image,(col*224,i*248+28))
            draw.text((4,i*248+254),row['sample_id']+' '+row['prompt'][:160],fill='black')
        destination=run/f'comparison_{page+1}.png'; sheet.save(destination); artifacts.append(destination)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    probes=json.loads((run/'probes.json').read_text())
    for key,label in (('probe/fit/mse','Fit'),('probe/probe/mse','Separate noise'),('probe/probe/reference_read_off_mse','Read off')):
        axes[0].plot([p['step'] for p in probes],[p[key] for p in probes],label=label)
    axes[0].set_ylabel('Face flow MSE'); axes[0].legend()
    axes[1].plot(STEPS,[summary[str(s)]['id_sim'] for s in STEPS],marker='o')
    axes[1].axhline(.3313986753782956,ls=':',color='gray',label='Native'); axes[1].legend(); axes[1].set_ylabel('Identity similarity')
    axes[2].plot(STEPS,[summary[str(s)]['text_sim'] for s in STEPS],marker='o'); axes[2].set_ylabel('CLIP similarity')
    for ax in axes: ax.set_xlabel('Total updates'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(run/'learning_curves.png',dpi=150); plt.close(fig)
    artifacts.append(run/'learning_curves.png')
    lines=[f'# BA prompted generation: continuation from {start} to {target}','',
        f'New Comet run; weights and Adam state continued from the verified {start}-update run. Same 24 prompts/seeds/reference, native backgrounds and masks, full-scene cache, lr .001, batch 8. Only BA trains. Validation loads no target photographs.',
        '', 'The native backbone scores ID 0.33140 and CLIP 28.0729. Exact exterior preservation is enforced by final blending; raw decoder images are saved separately.',
        '', '| Total updates | ID similarity | CLIP | Unowned face |','|---:|---:|---:|---:|']
    for step in STEPS:
        m=summary[str(step)]; lines.append(f"|{step}|{m['id_sim']:.5f}|{m['text_sim']:.4f}|{m['id_sim_unowned']:.1%}|")
    lines += ['',f"Best observed ID score: **{summary[str(best)]['id_sim']:.5f} at {best} updates**. Final-minus-start ID change: {final_audit['final_minus_start_id']:+.5f}; CLIP change: {final_audit['final_minus_start_clip']:+.4f}."]
    lines+=['','Cached flow probes share the same training photographs with separate noise and were used for earlier LR selection; they are not an unbiased test set. Full prompted images are generated through the live frozen backbone.',
            '', '![Metrics](learning_curves.png)',*[f'![Comparison {i+1}](comparison_{i+1}.png)' for i in range(3)],'']
    (run/'report.md').write_text('\n'.join(lines))
    exp=experiment(config,run)
    exp.log_parameters({'status':f'completed_{target}','additional_optimizer_updates':target-start,'live_metric_stride':identity.get('live_metric_every',25)})
    for path in [*artifacts,run/'report.md',run/'metric_summary.json',run/'final_audit.json',run/'metrics.jsonl',run/'probes.json',run/'resume_check.json',*list((run/'source_snapshot').rglob('*.py'))]:
        exp.log_asset(str(path),file_name=str(path.relative_to(run)).replace('/','_'))
    for path in artifacts: exp.log_image(str(path),name=path.stem,step=target)
    exp.end()
    print(json.dumps(final_audit,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','resume_probe','train','infer','decode','score','review'))
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--parent',type=Path,default=ROOT/'runs/flux4b_masked_face_flow_24_stable_20261001')
    parser.add_argument('--target-step',type=int,default=10000)
    parser.add_argument('--validate-every',type=int,default=2000)
    parser.add_argument('--parent-step',type=int,default=2000)
    parser.add_argument('--step',type=int)
    parser.add_argument('--resume',type=int,default=2000)
    args=parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True); torch.set_num_threads(8)
    run=args.run_dir.resolve()
    if args.command=='run': initialize(run,args.parent.resolve(),args.parent_step,args.target_step,args.validate_every)
    config,identity=verify(run)
    global STEPS
    STEPS=schedule(identity['continuation']['inherited_step'],config['training']['steps'],config['training']['validation_every'])
    if args.command=='run':
        worker=[sys.executable,'-m','scripts.continue_masked_face_flow']
        subprocess.run([*worker,'resume_probe','--run-dir',str(run)],check=True)
        for previous,step in zip(STEPS[:-1],STEPS[1:]):
            for command in ('train','infer','decode','score'):
                subprocess.run([*worker,command,'--run-dir',str(run),'--step',str(step),'--resume',str(previous)],check=True)
        subprocess.run([*worker,'review','--run-dir',str(run)],check=True)
    elif args.command=='resume_probe': resume_probe(run,config,identity)
    elif args.command=='train': train(run,config,identity,args.resume,args.step)
    elif args.command=='infer': infer(run,config,identity,args.step)
    elif args.command=='decode': decode(run,config,identity,args.step)
    elif args.command=='score': score(run,config,identity,args.step)
    else: review(run,config,identity)


if __name__=='__main__': main()
