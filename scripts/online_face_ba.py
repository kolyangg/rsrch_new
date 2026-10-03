"""Full-denoiser Q/K/V/O BA with frozen native masks: named one-ID24 or multi-ID96."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ba_dit import adapters
from ba_dit.checkpoint import load_adapters, save_training, trainable_parameters
from ba_dit.config import ROOT, adapter_identity, digest, load_config, revisions
from ba_dit.data.cache import cache_path, cache_spec, load_pair
from ba_dit.data.manifest import assert_disjoint, file_hash, read_manifest
from ba_dit.logging import connect
from ba_dit.nn.masked_face_attention import training_mask
from ba_dit.nn.masked_face_flow import face_alpha, token_alpha, scene_latent, preserve_background
from ba_dit.runtime import backend_module
from ba_dit.training import train_segment


SOURCES = ('scripts/online_face_ba.py','scripts/run_online_face_ba.py',
    'ba_dit/nn/masked_face_attention.py','ba_dit/nn/masked_face_flow.py',
    'ba_dit/backends/flux_runtime.py','ba_dit/backends/attention.py','ba_dit/backends/flux2_native.py',
    'ba_dit/nn/reference_read_delta.py','ba_dit/adapters.py','ba_dit/training.py','ba_dit/checkpoint.py',
    'ba_dit/config.py','ba_dit/runtime.py','ba_dit/logging.py','ba_dit/progress.py',
    'ba_dit/data/cache.py','ba_dit/data/geometry.py','ba_dit/data/manifest.py','ba_dit/metrics.py',
    'patches/flux2_reference_branch_and_offload.patch')
NATIVE_SOURCE = ROOT/'runs/flux4b_deep_identity1024_det_20261001'


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def memory():
    reserved = torch.cuda.max_memory_reserved()
    fraction = reserved/torch.cuda.get_device_properties(0).total_memory
    if fraction >= .9:
        raise RuntimeError(f'CUDA reserved memory exceeds admission limit: {fraction:.3f}')
    return {'peak_reserved_gib':reserved/2**30, 'reserved_fraction':fraction}


def experiment(config, run):
    exp = connect(config, run, name=run.name)
    exp.log_parameters({'experiment/variant':'online_masked_qkvo_v1',
        'trainable_scope':'branch Q/K/V/output LoRA only', 'trainable_parameters':25165824,
        'native_lora_enabled':False, 'full_denoiser_each_microbatch':True,
        'fresh_noise_and_timesteps':True, 'cached_target_hidden_states':False,
        'training/microbatch':1, 'training/world_size':config['training'].get('world_size', 1),
        'training/effective_batch':config['training']['grad_accum']*config['training'].get('world_size', 1),
        'loss':'face-mask-normalized native flow MSE; no ArcFace auxiliary',
        'validation/images':config['validation']['limit'],
        'validation/checkpoints':validation_steps(config),
        'validation/task':'prompt/reference generation; native background composed explicitly',
        'validation/face_cfg':config['validation']['guidance'],
        'validation/masks':'frozen native-generated masks; no target photos'})
    return exp


def validation_steps(config):
    if config['name'].startswith('flux4b_oneid_online_'):
        return [0, 500, 1000, 2000]
    total = config['training']['steps']
    return sorted({0, total, *range(config['training']['validation_every'], total,
                                   config['training']['validation_every'])})


def verify(run):
    config = load_config(run/'resolved_config.yaml')
    identity = json.loads((run/'identity.json').read_text())
    assert identity['config_sha256'] == digest(config)
    assert identity['base'] == adapter_identity(config)
    for path, sha in identity['source_sha256'].items():
        if file_hash(ROOT/path) != sha:
            raise ValueError(f'Immutable run source changed: {path}')
    for split in ('train','validation'):
        assert file_hash(config['data'][split+'_manifest']) == identity[split+'_manifest_sha256']
    assert file_hash(run/'routing_masks.json') == identity['routing_masks_sha256']
    return config, identity


def initialize(run, config_path, admission, native_source=None):
    config = load_config(config_path)
    native_check = json.loads((admission/'native_checks.json').read_text())
    resume_check = json.loads((admission/'resume_parity.json').read_text())
    assert native_check['all_frozen_parameters_exact'] and native_check['reserved_fraction'] < .9
    assert resume_check['exact_parameters'] and resume_check['exact_optimizer_scheduler_rng_cursor']
    assert resume_check.get('world_size', 1) == config['training'].get('world_size', 1)
    if config['data'].get('conditioning') == 'online':
        assert native_check['live_cached_conditioning_exact'] and native_check['conditioning_preserves_training_rng']
    rows = read_manifest(config['data']['validation_manifest'])
    train_rows = read_manifest(config['data']['train_manifest'], training=True)
    assert_disjoint(train_rows, rows)
    assert len(rows)==config['validation']['limit'] and not any('target' in row for row in rows)
    multi_id = native_source is not None
    if multi_id:
        assert len({r['identity_id'] for r in train_rows}) > 1
        assert not any(r.get('split_policy') == 'one_id_diagnostic' for r in train_rows+rows)
        assert resume_check['gradient_accumulation'] == config['training']['grad_accum']
    else:
        assert len(rows)==24 and len(train_rows)==19
    native_source = Path(native_source or NATIVE_SOURCE)
    # Verify cache headers without rereading hundreds of GB of training tensors.
    cached_train = [] if config['data'].get('conditioning') == 'online' else train_rows
    for split_rows, validation in ((cached_train,False),(rows,True)):
        for row in split_rows:
            entries=[(row,'encoder'),(row,'vae')]
            if validation: entries.append(({**row,'prompt':''},'encoder'))
            for item,stage in entries:
                with safe_open(cache_path(config,item,stage),framework='pt') as cache:
                    assert json.loads(cache.metadata()['record'])['key']==digest(cache_spec(config,item,stage))
                    if validation: assert 'target_latent' not in cache.keys()
    masks=json.loads((native_source/'routing_masks.json').read_text())
    signature=masks['source_signature']
    if 'dtype' in config['model']:
        from ba_dit.validation_masks import signature as mask_signature
        assert signature == mask_signature(config)
    assert signature['panel_sha256']==file_hash(config['data']['validation_manifest'])
    assert signature['revisions']==revisions(config)
    assert signature['target_size']==config['data']['target_size'] and signature['reference_size']==config['data']['reference_size']
    assert signature['steps']==config['validation']['steps'] and signature['guidance']==config['validation']['guidance']
    assert masks['feather_pixels']==config['branch']['mask_feather_pixels']
    run.mkdir(parents=True,exist_ok=False)
    (run/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    (run/'native').mkdir()
    for row in rows:
        key=row['sample_id']; record=masks['samples'][key]
        assert record['face_bbox'] and all(record[k]==row[k] for k in ('prompt','seed','identity_id'))
        for suffix,field in (('.png','baseline_image_sha256'),('.safetensors','latent_sha256')):
            source=native_source/'native'/f'{key}{suffix}'
            assert file_hash(source)==record[field]
            shutil.copyfile(source,run/'native'/source.name)
    for name in ('routing_masks.json','ownership_boxes.json','mask_overlays.png'):
        shutil.copyfile(native_source/name,run/name)
    native_quality=native_source/'quality_summary.json'
    if multi_id and not native_quality.is_file(): raise FileNotFoundError(native_quality)
    if native_quality.is_file(): shutil.copyfile(native_quality,run/'native/quality_summary.json')
    for name in ('native_checks.json','resume_parity.json'):
        shutil.copyfile(admission/name,run/name)
    sources=SOURCES + (('scripts/run_multi_id_face_ba.py','scripts/run_flux4b_multi_id.sh',
        'scripts/check_online_face_ba.py','ba_dit/validation_masks.py','ba_dit/data/conditioning.py') if multi_id else ())
    if config['training'].get('world_size', 1) > 1:
        sources += ('ba_dit/distributed_training.py', 'scripts/run_clust_v100.sh')
    for source in sources:
        target=run/'source_snapshot'/source
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/source,target)
    identity={'variant':'online_masked_qkvo_v1','base':adapter_identity(config),
        'config_sha256':digest(config),'source_sha256':{p:file_hash(ROOT/p) for p in sources},
        'train_manifest_sha256':file_hash(config['data']['train_manifest']),
        'validation_manifest_sha256':file_hash(config['data']['validation_manifest']),
        'routing_masks_sha256':file_hash(run/'routing_masks.json'),
        'native_source':str(native_source), 'admission':str(admission),
        'split_policy':'identity_disjoint' if multi_id else 'one_id_diagnostic',
        'trainable_parameters':25165824,'trainable_tensors':64,
        'loss':'mask-normalized flow MSE, fresh native noise and sigma',
        'validation_targets':False,'mask_feather_pixels':16,
        'training_masks':{r['sample_id']:{'target_hash':r['target_hash'],
             'token_coverage':float(training_mask(r,config).mean())} for r in train_rows}}
    write(run/'identity.json',identity)
    torch.manual_seed(config['training']['seed']);random.seed(config['training']['seed'])
    model=backend_module(config).load_transformer(config)
    inventory=adapters.install(model,config,'branch_only')
    assert sum(p.numel() for p in trainable_parameters(model).values())==identity['trainable_parameters']
    write(run/'optimizer_inventory.json',inventory)
    optimizer=torch.optim.AdamW(adapters.groups(model,config['training']['lr']),weight_decay=config['training']['weight_decay'])
    warmup=config['training']['warmup']
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/max(1,warmup)))
    torch.manual_seed(config['training']['seed']);random.seed(config['training']['seed'])
    data_digest=digest([{k:v for k,v in row.items() if k not in {'reference','target'}} for row in train_rows])
    save_training(model,optimizer,scheduler,config,'branch_only',run,0,0,data_digest)
    (run/'latest_checkpoint.txt').write_text(str(run/'checkpoint-000000')+'\n')
    exp=experiment(config,run)
    for name in ('identity.json','resolved_config.yaml','optimizer_inventory.json','native_checks.json','resume_parity.json','routing_masks.json'):
        exp.log_asset(str(run/name))
    exp.log_image(str(run/'mask_overlays.png'),name='native_face_masks',step=0)
    exp.end()
    print(json.dumps({'initialized':str(run),'comet':json.loads((run/'comet_experiment.json').read_text())}),flush=True)


@torch.no_grad()
def infer(run,config,step):
    backend=backend_module(config)
    from extensions_built_in.diffusion_models.flux2.src.sampling import get_schedule
    model=backend.load_transformer(config)
    adapters.install(model,config,'branch_only')
    checkpoint=run/f'checkpoint-{step:06d}'
    load_adapters(model,checkpoint,config,'branch_only')
    sha=file_hash(checkpoint/'adapters.safetensors')
    folder=run/f'validation-{step:06d}';folder.mkdir(exist_ok=True)
    (folder/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    rows=read_manifest(config['data']['validation_manifest'])
    assert not any('target' in r for r in rows)
    masks=json.loads((run/'routing_masks.json').read_text())['samples']
    height,width=config['data']['target_size']
    times=get_schedule(config['validation']['steps'],height*width//256)
    torch.cuda.reset_peak_memory_stats()
    samples=[];started=time.monotonic()
    for row in rows:
        key=row['sample_id'];path=folder/f'{key}.safetensors'
        native_path=run/'native'/path.name
        assert file_hash(native_path)==masks[key]['latent_sha256']
        if path.exists():
            with safe_open(path,framework='pt') as f: assert f.metadata()['checkpoint_sha256']==sha
        else:
            tensors,_=load_pair(config,row,'cuda',negative=True)
            assert 'target_latent' not in tensors
            native=load_file(native_path)['latent'].cuda()
            alpha=token_alpha(face_alpha((width,height),masks[key]['face_bbox'],config['branch']['mask_feather_pixels'])).cuda()
            tensors['target_face_mask']=alpha.flatten(1)
            noise=torch.randn(native.shape,generator=torch.Generator().manual_seed(row['seed']),dtype=native.dtype).cuda()
            face=noise.clone()
            for current,following in zip(times[:-1],times[1:]):
                sigma=torch.tensor([current],device='cuda',dtype=face.dtype)
                mixed=scene_latent(face,native,noise,current,alpha)
                positive=backend.predict(model,tensors,mixed,sigma,config,True)
                negative=backend.predict(model,tensors,mixed,sigma,config,True,negative=True)
                velocity=negative+config['validation']['guidance']*(positive-negative)
                face=face+(following-current)*velocity
            latent=scene_latent(face,native,noise,0.,alpha)
            assert torch.isfinite(latent).all()
            outside=alpha.expand_as(latent)==0
            assert torch.equal(latent[outside],native[outside])
            save_file({'latent':latent.cpu().contiguous()},path,metadata={'checkpoint_sha256':sha})
            memory()
        samples.append({**{k:row[k] for k in ('sample_id','identity_id','prompt','seed')},
            'image':key+'.png','checkpoint_sha256':sha,'target_photo_loaded':False,
            'mask_source':'frozen native generation','face_cfg':config['validation']['guidance']})
        write(folder/'validation.json',{'backend':config['model']['arch'],'mode':'branch_only',
            'variant':'online_masked_qkvo_v1','checkpoint':str(checkpoint),'checkpoint_sha256':sha,
            'panel_sha256':file_hash(config['data']['validation_manifest']),'samples':samples})
        print(f'Validation {step}: {len(samples)}/{len(rows)}; {time.monotonic()-started:.0f}s',flush=True)
    write(run/f'inference_audit_{step}.json',{'checkpoint_sha256':sha,'samples':len(rows),
        'full_denoiser':True,'target_photos_loaded':False,'exact_latent_exterior':True,**memory()})


@torch.no_grad()
def decode(run,config,step):
    backend=backend_module(config);vae=backend.load_vae(config)
    folder=run/f'validation-{step:06d}'
    report=json.loads((folder/'validation.json').read_text())
    masks=json.loads((run/'routing_masks.json').read_text())['samples']
    exp=experiment(config,run);audit=[]
    try:
        for row in report['samples']:
            key=row['sample_id'];original=run/'native'/f'{key}.png'
            assert file_hash(original)==masks[key]['baseline_image_sha256']
            with safe_open(folder/f'{key}.safetensors',framework='pt') as f:
                assert f.metadata()['checkpoint_sha256']==report['checkpoint_sha256']
                latent=f.get_tensor('latent')
            image=backend.decode(vae,latent,config)
            image.save(folder/f'{key}_raw.png')
            native=Image.open(original).convert('RGB')
            alpha=face_alpha(native.size,masks[key]['face_bbox'],config['branch']['mask_feather_pixels'])
            image=preserve_background(image,native,alpha);image.save(folder/f'{key}.png')
            difference=np.abs(np.asarray(image).astype(float)-np.asarray(native).astype(float))
            support=alpha.squeeze().numpy()>0
            audit.append({'sample_id':key,'background_max_abs':float(difference[~support].max()) if (~support).any() else 0.,
                          'face_mean_abs':float(difference[support].mean())})
            exp.log_image(image,name=f"fixed{config['validation']['limit']}/{key}",step=step,metadata={'prompt':row['prompt'],'seed':row['seed']})
        write(run/f'background_audit_{step}.json',audit)
        exp.log_asset(str(folder/'validation.json'),file_name=f'validation_{step:06d}.json')
    finally:exp.end()


def summarize(run,config,step):
    folder=run/f'validation-{step:06d}'
    current=json.loads((folder/'quality_summary.json').read_text())['metrics']
    scores={int(p.parent.name.split('-')[1]):json.loads(p.read_text())['metrics']
            for p in run.glob('validation-*/quality_summary.json')}
    best=max(scores,key=lambda s:scores[s]['id_sim'])
    checkpoint=run/f'checkpoint-{best:06d}'
    write(run/'best_checkpoint.json',{'step':best,'id_sim':scores[best]['id_sim'],
          'directory':str(checkpoint),'sha256':file_hash(checkpoint/'adapters.safetensors')})
    native_quality=run/'native/quality_summary.json'
    native_score=json.loads(native_quality.read_text())['metrics']['id_sim'] if native_quality.exists() else .3313986754
    write(run/'comparison_summary.json',{'metrics':scores,'best_step':best,'latest_step':step,
          'native_panel_id_sim':native_score,'step0_is_untrained_native_initialized_branch':True})
    rows=read_manifest(config['data']['validation_manifest'])
    masks=json.loads((run/'routing_masks.json').read_text())['samples']
    columns=[('Native',run/'native'),('Untrained BA',run/'validation-000000')]
    if step:columns.append((f'BA {step}',folder))
    exp=experiment(config,run)
    try:
        for page in range((len(rows)+7)//8):
            sheet=Image.new('RGB',(224*len(columns),8*234+32),'white');draw=ImageDraw.Draw(sheet)
            for col,(label,_) in enumerate(columns):draw.text((col*224+6,8),label,fill='black')
            for i,row in enumerate(rows[page*8:(page+1)*8]):
                key=row['sample_id'];x0,y0,x1,y1=masks[key]['face_bbox']
                cx,cy=(x0+x1)/2,(y0+y1)/2;side=max(x1-x0,y1-y0)*1.35
                box=tuple(map(int,(cx-side/2,cy-side/2,cx+side/2,cy+side/2)))
                for col,(_,source) in enumerate(columns):
                    im=Image.open(source/f'{key}.png').crop(box).resize((216,210))
                    sheet.paste(im,(224*col,234*i+32));draw.text((224*col+4,234*i+246),key,fill='black')
            path=run/f'paired_faces_{step:06d}_{page+1}.png';sheet.save(path)
            exp.log_image(str(path),name=f'paired_faces/{page+1}',step=step)
        exp.log_metric('validation/best_id_sim',scores[best]['id_sim'],step=step)
        for name in ('best_checkpoint.json','comparison_summary.json'):
            exp.log_asset(str(run/name))
    finally:exp.end()
    print(json.dumps({'validated_step':step,'metrics':current,'best_step':best}),flush=True)


def main(args):
    torch.set_num_threads(8)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    run=args.run.resolve()
    if args.action=='init':
        initialize(run,args.config,args.admission.resolve(),args.native_source);return
    config,_=verify(run)
    if args.action=='train':
        train_segment(config,'branch_only',run,args.step,resume=run/f'checkpoint-{args.resume:06d}')
    elif args.action=='infer':infer(run,config,args.step)
    elif args.action=='decode':decode(run,config,args.step)
    elif args.action=='summarize':summarize(run,config,args.step)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('init','train','infer','decode','summarize'))
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/flux4b_oneid_online_face_qkvo_r128_768.yaml')
    p.add_argument('--admission',type=Path)
    p.add_argument('--native-source',type=Path,help='Explicit frozen native bundle for an identity-disjoint multi-ID run')
    p.add_argument('--step',type=int,default=0)
    p.add_argument('--resume',type=int,default=0)
    main(p.parse_args())
