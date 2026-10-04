"""Prompted scenes with frozen native backgrounds and zero-initialized BA faces."""

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
from PIL import Image, ImageDraw
from safetensors.torch import load_file, save_file

from ba_dit.config import ROOT, adapter_identity, digest, load_config
from ba_dit.data.cache import load_pair
from ba_dit.data.manifest import file_hash, read_manifest
from ba_dit.face_diagnostic import target_face_tokens
from ba_dit.nn.face_crop_flow import FaceCropFlow, capture, install
from ba_dit.nn.masked_face_flow import face_alpha, token_alpha, scene_latent, preserve_background
from ba_dit.runtime import backend_module
from ba_dit.validation_masks import load_masks
from scripts.face_crop_flow import write, memory, experiment, prediction
from scripts.train_masked_face_flow import train

BASELINE = ROOT/'runs/flux4b_face_one_id_24_val_20261001'
STEPS = (0,1000,2000)
SIGMAS = (.05,.2,.4,.6,.8,1.)
SOURCES = ('scripts/masked_face_flow.py','scripts/train_masked_face_flow.py','ba_dit/nn/masked_face_flow.py','scripts/face_crop_flow.py',
           'ba_dit/nn/face_crop_flow.py','ba_dit/backends/attention.py','ba_dit/backends/flux_runtime.py')


def initialize(run, cache_from=None):
    run.mkdir(parents=True,exist_ok=False)
    config = load_config(BASELINE/'resolved_config.yaml')
    config['name'] = 'flux4b_masked_face_flow_one_id_24'
    config['training'].update(steps=2000,lr=.001,warmup=100,grad_accum=1,seed=142,checkpoint_every=1000,validation_every=1000)
    (run/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    rows = read_manifest(config['data']['validation_manifest'])
    assert len(rows)==24 and not any('target' in row for row in rows)
    masks = load_masks(config,[r['sample_id'] for r in rows])
    source_report = json.loads((BASELINE/'validation-000000/validation.json').read_text())
    assert source_report['checkpoint'] is None
    samples = {r['sample_id']:r for r in source_report['samples']}
    native = run/'native'
    native.mkdir()
    copied = {}
    for row in rows:
        key = row['sample_id']
        record = masks['samples'][key]
        assert record['face_bbox'] is not None and record['status'] not in ('no_face','ambiguous')
        assert all(samples[key][k]==row[k] for k in ('seed','prompt','identity_id'))
        source = BASELINE/'validation-000000'/f'{key}.png'
        assert file_hash(source)==record['baseline_image_sha256']
        for suffix in ('.png','.safetensors'):
            shutil.copyfile(source.with_suffix(suffix),native/f'{key}{suffix}')
        copied[key] = {**record,'latent_sha256':file_hash(native/f'{key}.safetensors')}
        alpha = face_alpha((768,768),record['face_bbox'])
        overlay = Image.open(source).convert('RGB')
        tint = Image.new('RGB',overlay.size,(255,40,40))
        overlay = Image.composite(tint,overlay,Image.fromarray((alpha.squeeze().numpy()*90).astype(np.uint8)))
        overlay.thumbnail((256,256))
        overlay.save(native/f'{key}_overlay.png')
    write(run/'routing_masks.json',{'usage':'inference routing; explicitly requested 2026-10-01',
          'source':str(BASELINE/'validation-000000'),'source_signature':masks['signature'],
          'feather_pixels':16,'selection':masks['selection'],'samples':copied})
    write(run/'ownership_boxes.json',{k:v['face_bbox'] for k,v in copied.items()})
    sheet = Image.new('RGB',(6*256,4*280),'white')
    draw = ImageDraw.Draw(sheet)
    for i,row in enumerate(rows):
        x,y=(i%6)*256,(i//6)*280
        sheet.paste(Image.open(native/f"{row['sample_id']}_overlay.png"),(x,y))
        draw.text((x+3,y+258),row['sample_id'],fill='black')
    sheet.save(run/'mask_overlays.png')
    torch.manual_seed(142)
    branch = FaceCropFlow(query_residual=True,noise_skip=True,timestep_scaling=False)
    save_file(branch.state_dict(),run/'initial.safetensors')
    identity = {'variant':'masked_full_scene_face_flow_v1','base':adapter_identity(config),
         'config_sha256':digest(config),'source_sha256':{p:file_hash(ROOT/p) for p in SOURCES},
         'train_manifest_sha256':file_hash(config['data']['train_manifest']),
         'validation_manifest_sha256':file_hash(config['data']['validation_manifest']),
         'routing_masks_sha256':file_hash(run/'routing_masks.json'),'initial_sha256':file_hash(run/'initial.safetensors'),
         'query_residual':True,'noise_skip':True,'timestep_scaling':False,
         'trainable_parameters':sum(p.numel() for p in branch.parameters()),
         'loss':'unweighted native flow MSE on 64 sampled face tokens per full-image case',
         'sigmas':SIGMAS,'training_reference':rows[0], 'training_images':19,
         'native_background_cfg':4.,'face_cfg':1.,'validation_targets':False,
         'native_velocity_inside_face':False,'background':'frozen native generated image; noised interpolation; exact final pixel preservation'}
    if cache_from:
        donor=json.loads((cache_from/'identity.json').read_text())
        for field in ('base','train_manifest_sha256','validation_manifest_sha256','training_reference','loss','sigmas'):
            assert json.loads(json.dumps(identity[field]))==donor[field], field
        for source in ('ba_dit/nn/face_crop_flow.py','ba_dit/backends/attention.py','ba_dit/backends/flux_runtime.py'):
            assert identity['source_sha256'][source]==donor['source_sha256'][source]
        assert identity['initial_sha256']==donor['initial_sha256']
        manifest=json.loads((cache_from/'cache_manifest.json').read_text())
        (run/'cases').mkdir()
        for record in manifest['records']:
            source=cache_from/record['file']
            assert file_hash(source)==record['sha256']
            os.link(source,run/record['file'])
        write(run/'cache_manifest.json',manifest)
        shutil.copyfile(cache_from/'native_off_parity.json',run/'native_off_parity.json')
        identity['cache_source']={'run':str(cache_from),'identity_sha256':file_hash(cache_from/'identity.json'),
                                  'manifest_sha256':file_hash(cache_from/'cache_manifest.json')}
    write(run/'identity.json',identity)
    for p in SOURCES:
        destination=run/'source_snapshot'/p
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/p,destination)
    exp=experiment(config,run)
    exp.log_parameters({'task':'reference-and-prompt face generation inside native scenes','face_cfg':1.,
                        'background_cfg':4.,'inference_mask_source':'native BA-off generated images',
                        'validation_samples':24,'full_scene_resolution':768})
    for name in ('identity.json','routing_masks.json','resolved_config.yaml'):
        exp.log_asset(str(run/name))
    exp.log_image(str(run/'mask_overlays.png'),name='native_face_masks',step=0)
    exp.end()


def verify(run):
    config=load_config(run/'resolved_config.yaml')
    identity=json.loads((run/'identity.json').read_text())
    assert digest(config)==identity['config_sha256']
    assert all(file_hash(ROOT/p)==v for p,v in identity['source_sha256'].items())
    for name,field in [('routing_masks.json','routing_masks_sha256'),('initial.safetensors','initial_sha256')]:
        assert file_hash(run/name)==identity[field]
    for split in ('train','validation'):
        assert file_hash(config['data'][split+'_manifest'])==identity[split+'_manifest_sha256']
    return config,identity


def with_fixed_reference(config,row,reference):
    tensors,_=load_pair(config,row,'cuda')
    for key in ('reference_tokens','reference_ids','reference_mask'):
        tensors[key]=reference[key]
    return tensors


@torch.no_grad()
def prepare(run,config,identity):
    backend=backend_module(config)
    model=backend.load_transformer(config)
    reference,_=load_pair(config,identity['training_reference'],'cuda')
    parity_noise=torch.randn((1,128,48,48),dtype=torch.bfloat16,device='cuda')
    parity_sigma=torch.tensor([.5],dtype=torch.bfloat16,device='cuda')
    native_before=backend.predict(model,reference,parity_noise,parity_sigma,config,branch=False)
    branch=install(model,True,True,False)
    branch.load_state_dict(load_file(run/'initial.safetensors'))
    native_after=backend.predict(model,reference,parity_noise,parity_sigma,config,branch=False)
    assert torch.equal(native_before,native_after)
    write(run/'native_off_parity.json',{'exact':True,'max_abs':float((native_before-native_after).abs().max())})
    assert all('reference_branch.' in n for n,p in model.named_parameters() if p.requires_grad)
    reference,_=load_pair(config,identity['training_reference'],'cuda')
    rows=read_manifest(config['data']['train_manifest'],training=True)
    assert len(rows)==19 and all(r['identity_id']==identity['training_reference']['identity_id'] for r in rows)
    destination=run/'cases'
    destination.mkdir(exist_ok=True)
    records=[]
    began=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for split,seeds in (('fit',(142,242)),('probe',(10142,))):
        for i,row in enumerate(rows):
            tensors=with_fixed_reference(config,row,reference)
            target=tensors['target_latent']
            weights=target_face_tokens(row,config).flatten()
            for seed in seeds:
                for j,value in enumerate(SIGMAS):
                    path=destination/f'{split}_{i}_{seed}_{j}.safetensors'
                    if not path.exists():
                        generator=torch.Generator().manual_seed(seed+i*100+j)
                        noise=torch.randn(target.shape,dtype=target.dtype,generator=generator).cuda()
                        sigma=torch.tensor([value],dtype=target.dtype,device='cuda')
                        noisy=(1-sigma)*target+sigma*noise
                        selected=torch.multinomial(weights,64,replacement=True,generator=generator).cuda()
                        features=capture(model,branch,backend,tensors,noisy,sigma,config)
                        cached={k:v.cpu().contiguous() for k,v in features.items()}
                        for key in ('query','noise'): cached[key]=cached[key][:,selected.cpu()].contiguous()
                        cached.update(flow_target=(noise-target).flatten(2).transpose(1,2)[:,selected].cpu().contiguous(),
                                      query_indices=selected.cpu()[None],noisy=noisy.cpu(),sigma=sigma.cpu())
                        if not records:
                            assert not prediction(branch,{k:v.cuda() for k,v in cached.items()}).count_nonzero()
                            repeated=capture(model,branch,backend,tensors,noisy,sigma,config)
                            assert torch.equal(features['query'],repeated['query'])
                        save_file(cached,path)
                    records.append({'file':str(path.relative_to(run)),'sha256':file_hash(path),
                                    'split':split,'row':i,'sigma':value})
            memory()
            print(f'Full-scene feature cache: {split} {i+1}/19',flush=True)
    write(run/'cache_manifest.json',{'records':records,'seconds':time.monotonic()-began,**memory()})


def checkpoint(branch,run,identity,step):
    path=run/f'checkpoint-{step:06d}/branch.safetensors' if step else run/'initial.safetensors'
    if step:
        record=json.loads((path.parent/'manifest.json').read_text())
        assert record['identity']==identity and file_hash(path)==record['sha256']
    branch.load_state_dict(load_file(path),strict=True)
    return file_hash(path)


def combine(pairs):
    assert all(torch.equal(pairs[0]['reference_mask'],p['reference_mask']) for p in pairs)
    return {k:pairs[0][k] if k=='reference_mask' else torch.cat([p[k] for p in pairs]) for k in pairs[0]}


@torch.no_grad()
def infer(run,config,identity,only_step=None,branch_factory=None):
    backend=backend_module(config)
    from extensions_built_in.diffusion_models.flux2.src.sampling import get_schedule
    model=backend.load_transformer(config)
    if branch_factory is not None:
        parity_inputs,_=load_pair(config,identity['training_reference'],'cuda')
        parity_noise=torch.randn((1,128,48,48),dtype=torch.bfloat16,device='cuda')
        parity_sigma=torch.tensor([.5],dtype=torch.bfloat16,device='cuda')
        native_before=backend.predict(model,parity_inputs,parity_noise,parity_sigma,config,branch=False)
    branch=install(model,True,True,False) if branch_factory is None else branch_factory(model)
    if branch_factory is not None:
        native_after=backend.predict(model,parity_inputs,parity_noise,parity_sigma,config,branch=False)
        assert torch.equal(native_before,native_after)
        write(run/'native_off_parity.json',{'exact':True,'max_abs':0.})
    rows=read_manifest(config['data']['validation_manifest'])
    assert not any('target' in row for row in rows)
    masks=json.loads((run/'routing_masks.json').read_text())['samples']
    times=get_schedule(config['validation']['steps'],48*48)
    torch.cuda.reset_peak_memory_stats()
    parity=[]
    for step in STEPS if only_step is None else (only_step,):
        weight_hash=checkpoint(branch,run,identity,step)
        if (run/'cache_manifest.json').exists():
            record=json.loads((run/'cache_manifest.json').read_text())['records'][-1]
            case={k:v.cuda() for k,v in load_file(run/record['file']).items()}
            row=read_manifest(config['data']['train_manifest'],training=True)[record['row']]
            reference,_=load_pair(config,identity['training_reference'],'cuda')
            tensors=with_fixed_reference(config,row,reference)
            fresh=capture(model,branch,backend,tensors,case['noisy'],case['sigma'],config)
            for key in ('query','noise'): fresh[key]=fresh[key][:,case['query_indices'][0]]
            assert torch.equal(branch.predict(**fresh),prediction(branch,case))
            parity.append({'step':step,'exact':True,'case':record['file']})
        folder=run/f'validation-{step:06d}'
        folder.mkdir(exist_ok=True)
        samples=[]
        began=time.monotonic()
        for offset in range(0,len(rows),2):
            batch=rows[offset:offset+2]
            paths=[folder/(r['sample_id']+'.safetensors') for r in batch]
            if not all(p.exists() for p in paths):
                pairs=[load_pair(config,row,'cuda')[0] for row in batch]
                assert not any('target_latent' in p for p in pairs)
                tensors=combine(pairs)
                natives=[]; alphas=[]
                for row in batch:
                    key=row['sample_id']
                    native=run/'native'/f'{key}.safetensors'
                    assert file_hash(native)==masks[key]['latent_sha256']
                    natives.append(load_file(native)['latent'])
                    alphas.append(token_alpha(face_alpha((768,768),masks[key]['face_bbox'])))
                native=torch.cat(natives).cuda()
                alpha=torch.cat(alphas).cuda()
                noise=torch.cat([torch.randn(native[:1].shape,dtype=native.dtype,generator=torch.Generator().manual_seed(r['seed'])) for r in batch]).cuda()
                face=noise.clone()
                if step or not identity.get('initial_flow_is_zero',True):
                    for current,following in zip(times[:-1],times[1:]):
                        sigma=torch.tensor([current],dtype=face.dtype,device='cuda')
                        mixed=scene_latent(face,native,noise,current,alpha)
                        features=capture(model,branch,backend,tensors,mixed,sigma,config)
                        velocity=branch.predict(**features).transpose(1,2).reshape_as(face).to(face.dtype)
                        face=face+(following-current)*velocity
                else:
                    assert not branch.out.weight.count_nonzero() and not branch.noisy_out.weight.count_nonzero()
                    assert torch.equal(face,noise)
                latent=scene_latent(face,native,noise,0.,alpha)
                assert torch.isfinite(latent).all()
                outside=alpha.expand_as(latent)==0
                assert torch.equal(latent[outside],native[outside])
                for i,path in enumerate(paths):
                    save_file({'latent':latent[i:i+1].cpu().contiguous()},path,metadata={'checkpoint_sha256':weight_hash})
                memory()
            for row in batch:
                samples.append({**{k:row[k] for k in ('sample_id','identity_id','prompt','seed')},
                     'image':row['sample_id']+'.png','mask_source':'frozen native generated image','face_cfg':1.,
                     'checkpoint_sha256':weight_hash,'target_photo_loaded':False})
            print(f'Prompted validation {step}: {len(samples)}/24; {time.monotonic()-began:.0f}s',flush=True)
        write(folder/'validation.json',{'samples':samples,'checkpoint':str(run/f'checkpoint-{step:06d}') if step else None,
               'variant':identity['variant'],'checkpoint_sha256':weight_hash,'panel_sha256':identity['validation_manifest_sha256']})
        (folder/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    write(run/f'inference_audit_{only_step}.json',{'parity':parity,**memory()})


@torch.no_grad()
def decode(run,config,identity,only_step=None,experiment_factory=None):
    backend=backend_module(config)
    vae=backend.load_vae(config)
    rows=read_manifest(config['data']['validation_manifest'])
    masks=json.loads((run/'routing_masks.json').read_text())['samples']
    exp=(experiment if experiment_factory is None else experiment_factory)(config,run)
    checks=[]
    for step in STEPS if only_step is None else (only_step,):
        folder=run/f'validation-{step:06d}'
        for row in rows:
            key=row['sample_id']
            original=run/'native'/f'{key}.png'
            assert file_hash(original)==masks[key]['baseline_image_sha256']
            image=backend.decode(vae,load_file(folder/f'{key}.safetensors')['latent'],config)
            image.save(folder/f'{key}_raw.png')
            native=Image.open(original)
            alpha=face_alpha(native.size,masks[key]['face_bbox'])
            result=preserve_background(image,native,alpha)
            result.save(folder/f'{key}.png')
            a=np.asarray(result).astype(float); b=np.asarray(native).astype(float)
            support=alpha.squeeze().numpy()>0
            checks.append({'step':step,'sample_id':key,'background_max_abs':float(np.abs(a-b)[~support].max()),
                           'face_mean_abs':float(np.abs(a-b)[support].mean())})
            exp.log_image(result,name=f"prompted/{key}",step=step)
        print(f'Decoded and logged prompted validation {step}',flush=True)
    write(run/f'background_audit_{only_step}.json',checks)
    exp.end()


def review(run,config,identity):
    python=ROOT/'envs/metrics/bin/python'
    for step in STEPS:
        subprocess.run([str(python),str(ROOT/'scripts/evaluate_metrics.py'),'--validation',str(run/f'validation-{step:06d}'),
                        '--ownership-boxes',str(run/'ownership_boxes.json'),'--log-dir',str(run),'--global-step',str(step)],check=True)
    rows=read_manifest(config['data']['validation_manifest'])
    sheets=[]
    for page in range(3):
        sheet=Image.new('RGB',(4*256,8*280+30),'white')
        draw=ImageDraw.Draw(sheet)
        for col,label in enumerate(('Native backbone','BA 0','BA 1000','BA 2000')):
            draw.text((col*256+5,5),label,fill='black')
        for i,row in enumerate(rows[page*8:page*8+8]):
            key=row['sample_id']
            for col,folder in enumerate((run/'native',*(run/f'validation-{s:06d}' for s in STEPS))):
                image=Image.open(folder/f'{key}.png'); image.thumbnail((256,256))
                sheet.paste(image,(col*256,i*280+30))
            draw.text((4,i*280+288),key+' '+row['prompt'][:130],fill='black')
        path=run/f'comparison_{page+1}.png'; sheet.save(path); sheets.append(path)
    summary={str(s):json.loads((run/f'validation-{s:06d}/quality_summary.json').read_text())['metrics'] for s in STEPS}
    write(run/'metric_summary.json',summary)
    lines=['# Prompted generation: native background and BA face','',
           '24 fixed prompts/seeds, identity 51, 768 px. No target photographs are loaded for validation. Native BA-off images and their face masks are reused by exact image hash.',
           '', 'A two-pass masked-flow adaptation of the CL14 spatial split, not a port of its U-Net attention processor. BA alone predicts face velocity; the known native scene provides background context. Final pixel blending preserves the exterior exactly. Raw decoded images are retained separately.',
           '', 'Only 2,408,448 BA parameters train. Zero output projections leave face latents as noise at step zero. The frozen backbone provides prompt-conditioned query features and ID-reference keys/values. Training uses full-image face tokens, not resized face crops.',
           '', 'Native background CFG=4; BA face CFG=1; 20 Euler steps. No claim of pure identity transport: query/noisy paths and native reference conditioning also contribute.',
           '', '| Step | ID similarity | No detected face | CLIP |','|---:|---:|---:|---:|']
    for s in STEPS:
        m=summary[str(s)]; lines.append(f"|{s}|{m['id_sim']:.4f}|{m['id_sim_no_face']:.1%}|{m['text_sim']:.3f}|")
    lines+=['','![Masks](mask_overlays.png)',*[f'![Comparison {i+1}]({p.name})' for i,p in enumerate(sheets)],'']
    (run/'report.md').write_text('\n'.join(lines))
    exp=experiment(config,run)
    for path in [*sheets,run/'report.md',run/'metric_summary.json',run/'metrics.jsonl',run/'probes.json',run/'resume_check.json',run/'training_summary.json']:
        exp.log_asset(str(path))
    for path in sheets: exp.log_image(str(path),name=path.stem,step=2000)
    exp.end()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','prepare','train','infer','decode','review'))
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--cache-from',type=Path)
    parser.add_argument('--resume',type=int,default=0)
    parser.add_argument('--until',type=int,default=2000)
    parser.add_argument('--step',type=int,choices=STEPS)
    args=parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True); torch.set_num_threads(8)
    run=args.run_dir.resolve()
    if args.command=='run': initialize(run,args.cache_from.resolve() if args.cache_from else None)
    config,identity=verify(run)
    if args.command=='run':
        stages=[('infer',['--step','0']),('decode',['--step','0']),('prepare',[]),('train',['--until','50']),
                ('train',['--resume','50','--until','1000']),('infer',['--step','1000']),('decode',['--step','1000']),
                ('train',['--resume','1000']),('infer',['--step','2000']),('decode',['--step','2000']),('review',[])]
        for command,extra in stages:
            if command=='prepare' and (run/'cache_manifest.json').exists(): continue
            subprocess.run([sys.executable,'-m','scripts.masked_face_flow',command,'--run-dir',str(run),*extra],check=True)
    elif args.command=='prepare': prepare(run,config,identity)
    elif args.command=='train': train(run,config,identity,args.resume,args.until)
    elif args.command=='infer': infer(run,config,identity,args.step)
    elif args.command=='decode': decode(run,config,identity,args.step)
    else: review(run,config,identity)


if __name__=='__main__': main()
