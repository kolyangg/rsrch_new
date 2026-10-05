"""Fresh FLUX1a on two V100 workers, with serial resource-specific Slurm stages."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NAME = 'FLUX1a_cluster_4b_20261004'
RUN = ROOT/'runs'/NAME
SETUP = RUN.with_name(NAME+'_setup')
CONFIG = ROOT/'configs/clust/FLUX1a_cluster_4b.yaml'
PARENT = Path('/home/nasilaev/rsrch_new_staged/runs/FLUX1_cluster_4b_ddp_20261004')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def context():
    from ba_dit.config import load_config
    from ba_dit.data.manifest import file_hash
    from scripts.clust_flux1a_prepare import install_reader
    config = load_config(CONFIG)
    deployment = json.loads((ROOT/'deployment.json').read_text())
    for path, sha in deployment['sources'].items():
        assert file_hash(ROOT/path) == sha, f'Deployed source changed: {path}'
    record = json.loads((SETUP/'dataset_verification.json').read_text())
    assert file_hash(SETUP/'prepared_manifest.json') == record['prepared_sha256']
    parent_identity = json.loads((PARENT/'identity.json').read_text())
    for split in ('train','validation'):
        assert file_hash(config['data'][split+'_manifest']) == parent_identity[split+'_manifest_sha256']
    rows = json.loads((SETUP/'prepared_manifest.json').read_text())
    read = install_reader(config, rows)
    from ba_dit import training, distributed_training
    from scripts import online_face_ba, check_online_face_ba
    for module in (training, distributed_training, online_face_ba, check_online_face_ba):
        module.read_manifest = read
    import torch
    torch.set_num_threads(int(os.getenv('OMP_NUM_THREADS', '2')))
    torch.use_deterministic_algorithms(True)
    if (RUN/'identity.json').exists():
        online_face_ba.verify(RUN)
    return config, rows


def plan():
    import math
    from PIL import Image
    from ba_dit.data.manifest import file_hash
    from ba_dit.nn.masked_face_attention import training_mask
    from ba_dit.nn.online_identity_loss import supervision_identity
    from ba_dit.training import sample_at
    config, rows = context()
    supervision_identity(config)
    labels = {r['sample_id']:r for r in map(json.loads, (Path(config['data']['identity_supervision'])/'records.jsonl').read_text().splitlines())}
    eligible = [r for r in rows['train'] if labels.get(r['sample_id'],{}).get('accepted')]
    def token_count(row):
        with Image.open(row['reference']) as image: w,h=image.size
        scale = min(1.,math.sqrt(config['data']['reference_size']**2/(w*h)))
        return (int(w*scale)//16)*(int(h*scale)//16)
    selected = [sample_at(rows['train'],i,config['training']['seed']) for i in range(4)]
    selected += [max(eligible,key=lambda r:float(training_mask(r,config).sum())),max(eligible,key=token_count)]
    write(SETUP/'probe_rows.json',selected)
    native = SETUP/'native_bundle';native.mkdir(exist_ok=True)
    if not (native/'native').exists():(native/'native').symlink_to(PARENT/'native',target_is_directory=True)
    for name in ('ownership_boxes.json','mask_overlays.png'):
        shutil.copyfile(PARENT/name,native/name)
    shutil.copyfile(PARENT/'native/quality_summary.json',native/'quality_summary.json')
    masks=json.loads((PARENT/'routing_masks.json').read_text())
    previous=copy.deepcopy(masks['source_signature'])
    masks['source_signature']['encoder_device']='cuda'
    from ba_dit.validation_masks import signature
    assert masks['source_signature']==signature(config)
    masks['encoder_device_migration']={'previous_signature':previous,
        'condition':'cache_probe verifies all96 native encoder/VAE tensors bitwise before initialization',
        'source_sha256':file_hash(PARENT/'routing_masks.json')}
    write(native/'routing_masks.json',masks)


def cache(until, probe=False):
    import gc
    import torch
    from safetensors.torch import load_file
    from ba_dit.config import digest, load_config
    from ba_dit.data.cache import cache_path, cache_spec, write_cache
    from ba_dit.data.manifest import file_hash
    from ba_dit.runtime import backend_module
    from ba_dit.training import sample_at
    from ba_dit.progress import stage_progress
    config, rows = context()
    if not probe:
        from scripts.clust_gpu_metrics import cleanup_cache
        cleanup_cache(RUN, config, rows["validation"], until-2000)
    selected = (json.loads((SETUP/'probe_rows.json').read_text()) + rows['validation'] +
                [{**rows['validation'][0],'prompt':''}]) if probe else [
        sample_at(rows['train'],i,config['training']['seed']) for i in range((until-2000)*2,until*2)]
    estimate=len({r['prompt'] for r in selected})*512*7680*4+len(selected)*2*1024**2
    assert estimate<100*2**30 and shutil.disk_usage(ROOT).free>estimate*1.1
    backend=backend_module(config);created=[];checked=set();parity={'encoder':0,'vae':0}
    old=load_config(PARENT/'resolved_config.yaml')
    for stage in ('encoder','vae'):
        component=backend.load_encoder(config) if stage=='encoder' else backend.load_vae(config)
        encode=backend.encode_text if stage=='encoder' else backend.encode_images
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for i,row in enumerate(selected):
            path=cache_path(config,row,stage);created.append(str(path))
            # Hash consumed files even when a content-addressed tensor already exists.
            if stage=='vae':
                for field in ('reference','target'):
                    if row.get(field) and row[field] not in checked:
                        assert file_hash(row[field])==row[field+'_hash'];checked.add(row[field])
            if not path.exists():
                with torch.no_grad(),torch.random.fork_rng(devices=[0]):tensors,geometry=encode(component,config,row)
                write_cache(path,tensors,{'key':digest(cache_spec(config,row,stage)), 'spec':cache_spec(config,row,stage),**geometry})
            if probe and row in rows['validation']:
                before,after=load_file(cache_path(old,row,stage)),load_file(path)
                assert before.keys()==after.keys() and all(torch.equal(before[k],after[k]) for k in before)
                parity[stage]+=1
            stage_progress('conditioning/'+stage,i+1,len(selected),started)
        assert torch.cuda.max_memory_reserved()/torch.cuda.get_device_properties(0).total_memory<.9
        del component;gc.collect();torch.cuda.empty_cache()
    write((SETUP if probe else RUN)/f'cache_files_{until}.json',created)
    if probe:
        assert parity=={'encoder':96,'vae':96}
        write(SETUP/'conditioning_parity.json',{'fixed96_encoder_and_vae_exact':True,'samples':parity})


def ddp(folder, until, checkpoint=None, probe=False):
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=2',
             '-m','--','scripts.clust_flux1a','train_worker','--child','--until',str(until),'--folder',str(folder)]
    if checkpoint:command+=['--checkpoint',str(checkpoint)]
    if probe:command+=['--probe']
    subprocess.run(command,check=True)


def admit():
    import torch
    import yaml
    from safetensors.torch import load_file
    config,_=context()
    admission=SETUP/'admission';admission.mkdir(exist_ok=False)
    probe=copy.deepcopy(config);probe['training'].update(steps=2,checkpoint_every=1);probe['logging']['enabled']=False
    (admission/'resolved_config.yaml').write_text(yaml.safe_dump(probe,sort_keys=False))
    subprocess.run([sys.executable,'-u','-m','scripts.clust_flux1a','parity','--child'],check=True)
    for name,until,resume in [('continuous',2,None),('restarted',1,None),('restarted',2,'checkpoint-000001')]:
        folder=admission/name;folder.mkdir(exist_ok=True)
        (folder/'resolved_config.yaml').write_text(yaml.safe_dump(probe,sort_keys=False))
        ddp(folder,until,folder/resume if resume else None,True)
    a,b=[admission/name/'checkpoint-000002' for name in ('continuous','restarted')]
    wa,wb=load_file(a/'adapters.safetensors'),load_file(b/'adapters.safetensors')
    assert wa.keys()==wb.keys() and all(torch.equal(wa[k],wb[k]) for k in wa)
    def exact(x,y):
        if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x,y)
        if isinstance(x,dict):return x.keys()==y.keys() and all(exact(x[k],y[k]) for k in x)
        if isinstance(x,(tuple,list)):return len(x)==len(y) and all(exact(i,j) for i,j in zip(x,y))
        return x==y
    sa,sb=[torch.load(p/'training_state.pt',map_location='cpu',weights_only=True) for p in (a,b)]
    assert exact(sa,sb), 'DDP optimizer/scaler/RNG resume mismatch'
    assert sa['distributed']['world_size']==2 and sa['distributed']['scaler']==sa['grad_scaler']
    metrics=[json.loads(s) for s in (admission/'continuous/metrics.jsonl').read_text().splitlines()]
    assert all(m['hardware/reserved_fraction']<.9 for m in metrics)
    write(admission/'resume_parity.json',{'exact_parameters':True,'exact_optimizer_scheduler_rng_cursor':True,
        'exact_gradient_scaler':True,'world_size':2,'gradient_accumulation':1,'optimizer_updates':2,
        'peak_reserved_gib':max(m['hardware/peak_reserved_gib'] for m in metrics)})
    from scripts.online_face_ba import initialize
    initialize(RUN,CONFIG,admission,SETUP/'native_bundle',cached_training_rows=json.loads((SETUP/'probe_rows.json').read_text()))
    for name in ('conditioning_parity.json','dataset_verification.json'):
        shutil.copyfile(SETUP/name,RUN/name)
    shutil.copyfile(ROOT/'deployment.json',RUN/'deployment.json')


def execute(args):
    config,rows=context()
    if args.action=='plan':plan()
    elif args.action=='cache_probe':cache(0,True)
    elif args.action=='cache':cache(args.until)
    elif args.action=='admit':admit()
    elif args.action=='parity':
        from scripts import check_online_face_ba
        # Worst real layouts were selected on CPU in plan; do not scan all
        # training images while the GPU allocation waits for that CPU work.
        selected=json.loads((SETUP/'probe_rows.json').read_text())
        check_online_face_ba.read_manifest=lambda path,training=False,limit=None: selected if training else rows['validation']
        check_online_face_ba.parity(config,SETUP/'admission')
    elif args.action=='train_worker':
        from ba_dit.config import load_config
        from ba_dit.training import train_segment
        config=load_config(args.folder/'resolved_config.yaml')
        train_segment(config,'branch_only',args.folder,args.until,resume=args.checkpoint)
    elif args.action=='train':
        from scripts.run_multi_id_face_ba import latest_checkpoint,archive_uncheckpointed_metrics
        step=latest_checkpoint(RUN,args.until);archive_uncheckpointed_metrics(RUN,step)
        ddp(RUN,args.until,RUN/f'checkpoint-{step:06d}')
    elif args.action in ('infer','decode','summarize'):
        from scripts import online_face_ba
        from scripts import clust_stream_decode
        if args.action=='infer':clust_stream_decode.infer(online_face_ba,RUN,config,args.until)
        elif args.action=='decode' and clust_stream_decode.decoded(RUN,config,args.until):pass
        else:getattr(online_face_ba,args.action)(RUN,config,args.until)

    else:
        module='scripts.evaluate_metrics' if args.action=='score' else 'scripts.evaluate_face_quality'
        env='metrics' if args.action=='score' else 'face-quality'
        cmd=[str(ROOT/'envs/clust-v100'/env/'bin/python'),'-m',module,'--validation',str(RUN/f'validation-{args.until:06d}'),
             '--log-dir',str(RUN),'--global-step',str(args.until),'--no-comet']
        if args.action=='score':cmd+=['--ownership-boxes',str(RUN/'ownership_boxes.json')]
        else:cmd+=['--threads',os.getenv('SLURM_CPUS_PER_TASK','4')]
        subprocess.run(cmd,check=True)


def duration(seconds):
    if seconds is None:return 'estimating'
    hours,rem=divmod(max(0,int(seconds)),3600);minutes,seconds=divmod(rem,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def report(action,step,started):
    stamp=datetime.now().astimezone().strftime('%H:%M:%S %Z')
    prefix=f'{stamp} FLUX1a | {action} | checkpoint {step}'
    if action=='train' and (RUN/'metrics.jsonl').exists():
        from collections import deque
        m=json.loads(deque((RUN/'metrics.jsonl').open(),maxlen=1)[0])
        print(f"{stamp} TRAIN {m['step']}/20000 | loss {m['train/loss']:.5f} | {m['train/seconds']:.2f}s/step | training ETA {duration(m['train/eta_seconds'])} | next validation {step}",flush=True)
    elif (SETUP/'stage_progress.json').exists():
        p=json.loads((SETUP/'stage_progress.json').read_text())
        print(f"{prefix} | {p['stage']} {p['completed']}/{p['total']} | stage ETA {duration(p['eta_seconds'])}",flush=True)
    else:print(f'{prefix} | starting/loading | elapsed {duration(time.monotonic()-started)} | ETA estimating',flush=True)


def run_stage(args):
    import fcntl
    SETUP.mkdir(parents=True,exist_ok=True)
    lock_name=f'validation_{args.until}.lock' if args.action in ('score','face_quality','summarize') else 'gpu_stage.lock'
    lock=(SETUP/lock_name).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    marker=(RUN if RUN.exists() else SETUP)/f'{args.action}_{args.until}.done.json'
    if marker.exists():
        enqueue_next(args.action,args.until)
        return
    (SETUP/'stage_progress.json').unlink(missing_ok=True)
    logs=SETUP/'diagnostics';logs.mkdir(exist_ok=True)
    log=logs/f'{args.action}_{args.until}_{os.environ.get("SLURM_JOB_ID","local")}.log'
    write(SETUP/'status.json',{'stage':args.action,'step':args.until,'status':'running','job_id':os.getenv('SLURM_JOB_ID')})
    started=time.monotonic()
    env={**os.environ,'BA_STAGE_PROGRESS':str(SETUP/'stage_progress.json'),
         'BA_STAGE_ACTION':args.action,'BA_STAGE_STEP':str(args.until)}
    print(f'FLUX1a | stage {args.action} | checkpoint {args.until} | diagnostics {log}',flush=True)
    with log.open('w') as stream:
        child=subprocess.Popen([sys.executable,'-u','-m','scripts.clust_flux1a',args.action,'--child','--until',str(args.until)],
                               stdout=stream,stderr=subprocess.STDOUT,env=env)
        while child.poll() is None:
            try:report(args.action,args.until,started)
            except (OSError,json.JSONDecodeError):
                print(f'{args.action} | waiting for a complete progress record',flush=True)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
    if child.returncode:
        write(SETUP/'status.json',{'stage':args.action,'status':'failed','returncode':child.returncode,'diagnostics':str(log)})
        print(f'FAILED {args.action}: exit {child.returncode}; see {log}',flush=True)
        raise SystemExit(child.returncode)
    if args.action=='infer':
        subprocess.run([sys.executable,'-u','-m','scripts.clust_gpu_metrics','--run',str(RUN),'--step',str(args.until)],check=True,env=env)
    if args.action=='admit':
        print('ADMISSION PASSED | native parity, identity gradients, <90% memory, exact two-GPU save/resume',flush=True)
    if args.action=='summarize':
        folder=RUN/f'validation-{args.until:06d}'
        metrics={}
        for name,prefix in [('quality_summary.json',''),('face_quality_summary.json','face/')]:
            metrics.update({prefix+k:v for k,v in json.loads((folder/name).read_text())['metrics'].items()})
        print(f'VALIDATION {args.until} COMPLETE | '+ ' | '.join(f'{k}={v:.5f}' for k,v in metrics.items() if isinstance(v,(int,float))),flush=True)
    marker=(RUN if RUN.exists() else SETUP)/marker.name
    write(marker,{'finished_utc':datetime.now(timezone.utc).isoformat(),'job_id':os.getenv('SLURM_JOB_ID'),'seconds':time.monotonic()-started})
    print(f'COMPLETE {args.action} | elapsed {duration(time.monotonic()-started)}',flush=True)
    enqueue_next(args.action,args.until)


def stages():
    result=[('plan',0,0,2,'00:30:00'),('cache_probe',0,1,2,'00:30:00'),('admit',0,2,4,'00:45:00')]
    for step in range(0,20001,2000):
        if step:result += [('cache',step,1,2,'01:15:00'),('train',step,2,4,'06:00:00')]
        result += [('infer',step,1,2,'03:00:00'),('decode',step,0,2,'00:30:00'),
                   ('score',step,0,4,'02:00:00'),('face_quality',step,0,4,'02:00:00'),('summarize',step,0,1,'00:15:00')]
    return result


def enqueue_next(action=None, step=None):
    """One bounded successor per successful stage; never retry a failed submission."""
    import fcntl
    path=ROOT/'scratch'/f'pipeline_{NAME}.json'
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=json.loads(path.read_text());previous=record['jobs'][-1]
        if action is not None and (previous['action'],previous['step'])!=(action,step):return
        if not record['remaining_stages']:return
        action,step,gpus,cpus,limit=record['remaining_stages'][0]
        dependency=previous['id']
        if action=='infer' and record.get('stream_decode_admission_job'):
            dependency+=':'+record['stream_decode_admission_job']
        cmd=['sbatch','--parsable',f'--job-name=FLUX1a-{action}-{step}',f'--cpus-per-task={cpus}',f'--time={limit}',
             '--dependency=afterok:'+dependency,f'--output={ROOT}/logs/%x-%j.out',f'--error={ROOT}/logs/%x-%j.err']
        if gpus:cmd+=[f'--gpus=v100:{gpus}']
        cmd += [str(ROOT/'jobs/flux1a_clust_stage.sbatch'),action,'--until',str(step)]
        result=subprocess.run(cmd,text=True,capture_output=True)
        if result.returncode:
            write(SETUP/'submission_failure.json',{'command':cmd,'stderr':result.stderr,'returncode':result.returncode})
            raise RuntimeError('Successor submission rejected; see setup/submission_failure.json')
        job=result.stdout.strip().split(';')[0];assert job.isdigit()
        record['jobs'].append({'id':job,'action':action,'step':step,'gpus':gpus,'cpus':cpus,'command':cmd})
        record['remaining_stages'].pop(0);write(path,record)
        print(f'SCHEDULED {action} {step} | job {job} | after successful {previous["id"]}',flush=True)


def submit():
    path=ROOT/'scratch'/f'pipeline_{NAME}.json'
    if path.exists():raise FileExistsError('Pipeline already submitted; inspect saved IDs')
    prep=json.loads((ROOT/'scratch/FLUX1a_prepare_job.json').read_text())
    record={'run':str(RUN),'setup':str(SETUP),'monitor_job_id':prep['id'],
            'jobs':[{**prep,'step':0}],'remaining_stages':stages()}
    write(path,record);enqueue_next()
    print(json.dumps({'pipeline':str(path),'planned_stages':len(stages())+1,'monitor_job_id':prep['id']}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['submit','plan','cache_probe','cache','admit','parity','train_worker','train','infer','decode','score','face_quality','summarize'])
    p.add_argument('--until',type=int,default=0);p.add_argument('--child',action='store_true')
    p.add_argument('--folder',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--probe',action='store_true')
    args=p.parse_args()
    if args.action=='submit':submit()
    elif args.child:execute(args)
    else:run_stage(args)


if __name__=='__main__':main()
