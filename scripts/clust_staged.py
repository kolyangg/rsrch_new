"""FLUX1 cluster continuation with resources allocated only to the active stage.

Run this executor beside the frozen parent sources. It changes execution and
conditioning storage, not the backbone, loss, optimizer, data order or panel.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def execution_config(original, name):
    result = copy.deepcopy(original)
    if (original['model'].get('compute_precision') != 'amp_fp16_fp32_branch' or
            original['data'].get('conditioning') != 'online' or
            original['data'].get('encoder_device') != 'cuda:1' or
            original['training'].get('world_size', 1) != 1 or
            original['training'].get('grad_accum') != 1 or
            original['training'].get('microbatch_size', 1) != 1 or
            original['training'].get('steps') != 20000 or
            original['training'].get('validation_every') != 2000 or
            original['validation'].get('limit') != 96):
        raise ValueError('Expected the measured single-worker two-V100 parent')
    result['name'] = name
    result['data'].update(conditioning='cached', encoder_device='cuda')
    result['training']['world_size'] = 2
    return result


def prepare(args):
    import torch
    import yaml
    from ba_dit.config import adapter_identity, config_digest, digest, load_config
    from ba_dit.data.manifest import read_manifest, assert_disjoint
    from ba_dit.checkpoint import training_code_digest
    from ba_dit.training import sample_at
    parent, run = args.parent.resolve(), args.run.resolve()
    old = load_config(parent/'resolved_config.yaml')
    identity = json.loads((parent/'identity.json').read_text())
    parent_root = parent.parent.parent
    assert all(sha(parent_root/p) == h for p,h in identity['source_sha256'].items())
    checkpoint = parent/f'checkpoint-{args.step:06d}'
    manifest = json.loads((checkpoint/'manifest.json').read_text())
    previous_root=os.environ.get('BA_ROOT')
    os.environ['BA_ROOT']=str(parent_root)
    try:
        assert manifest['step'] == args.step and manifest['config_sha256'] == config_digest(old)
    finally:
        if previous_root is None:os.environ.pop('BA_ROOT',None)
        else:os.environ['BA_ROOT']=previous_root
    assert manifest['identity'] == adapter_identity(old)
    # Parent source hashes above establish the original code; this execution
    # explicitly adds mixed-precision DDP scaler support in the isolated runtime.
    if run.exists():
        raise FileExistsError('Choose a fresh continuation directory')
    config = execution_config(old, run.name)
    config['data']['cache_dir'] = str(run/'conditioning_cache')
    # Expensive full-image checks happen in this CPU-only allocation once.
    train = read_manifest(old['data']['train_manifest'], training=True)
    val = read_manifest(old['data']['validation_manifest'])
    assert_disjoint(train, val)
    data_digest = digest([{k:v for k,v in r.items() if k not in {'reference','target'}} for r in train])
    assert data_digest == manifest['data_sha256']
    state = torch.load(checkpoint/'training_state.pt', map_location='cpu', weights_only=True)
    assert state['cursor'] == args.step * old['training']['grad_accum']
    assert len(state['cuda_rng']) == 2 and state['grad_scaler']['scale'] > 0
    count = min(2000, old['training']['steps']-args.step)*config['training']['grad_accum']*2
    needed = [sample_at(train, state['cursor']+i, old['training']['seed']) for i in range(count)]
    estimate = len({r['prompt'] for r in needed})*512*7680*4 + len(needed)*2*1024**2
    if estimate > 100*2**30 or shutil.disk_usage(ROOT).free < estimate*1.1:
        raise RuntimeError('Insufficient approved cache budget/free space')
    run.mkdir(parents=True)
    (run/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    write(run/'prepared_manifest.json', {'train':train,'validation':val})
    original_hashes = {p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()}
    # Create a new, explicitly migrated checkpoint; never edit the parent files.
    dest = run/checkpoint.name; dest.mkdir()
    shutil.copyfile(checkpoint/'adapters.safetensors', dest/'adapters.safetensors')
    rank0 = {'torch_rng':state['torch_rng'],'python_rng':state['python_rng'],'cuda_rng':state['cuda_rng'][0]}
    # Rank0 resumes its stream; rank1 gets an explicit independent new stream.
    import random
    generator=torch.Generator().manual_seed(config['training']['seed']+1)
    rank1={'torch_rng':generator.get_state(),'python_rng':random.Random(config['training']['seed']+1).getstate(),
           'cuda_rng':None,'cuda_seed':config['training']['seed']+1}
    state['distributed']={'world_size':2,'ranks':[rank0,rank1],'scaler':state['grad_scaler']}
    state['cuda_rng']=[]
    torch.save(state,dest/'training_state.pt')
    new_manifest = {**manifest,'config_sha256':config_digest(config),
                    'training_code_sha256':training_code_digest(config)}
    write(dest/'manifest.json',new_manifest)
    (dest/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    from ba_dit.config import portable_config
    (dest/'resume_config.yaml').write_text(yaml.safe_dump(portable_config(config),sort_keys=False))
    (run/'checkpoint-000000').symlink_to(parent/'checkpoint-000000',target_is_directory=True)
    for folder in ('native','validation-000000'):
        (run/folder).symlink_to(parent/folder,target_is_directory=True)
    for name in ('routing_masks.json','ownership_boxes.json','mask_overlays.png','comet_experiment.json'):
        shutil.copyfile(parent/name,run/name)
    for path in parent.glob('paired_faces_000000_*.png'):
        shutil.copyfile(path,run/path.name)
    for action in ('infer','decode','score','face_quality','summarize'):
        write(run/f'{action}_0.done.json',{'inherited_from':str(parent)})
    rows = [json.loads(line) for line in (parent/'metrics.jsonl').read_text().splitlines()]
    (run/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step']<=args.step))
    sources={p:sha(ROOT/p) for p in (*identity['source_sha256'],'ba_dit/distributed_training.py','scripts/clust_staged.py')}
    write(run/'identity.json',{**identity,'config_sha256':digest(config),'source_sha256':sources,'parent_run':str(parent)})
    write(run/'execution_transition.json',{'parent':str(parent),'start_step':args.step,
        'old_config':old,'config_changes':['name','data.conditioning','data.encoder_device','data.cache_dir','training.world_size'],
        'parent_checkpoint_sha256':original_hashes,'prepared_manifest_sha256':sha(run/'prepared_manifest.json'),
        'data_digest':data_digest,'cache_estimate_gib':estimate/2**30,
        'rng_change':f"Preserve rank0 RNG; initialize rank1 at seed{config['training']['seed']+1}; preserve optimizer/scaler/global cursor",
        'batch_change':{'previous':1,'new':2},'start_cursor':state['cursor'],
        'migration_requires':'native conditioning parity, two-rank finite updates and exact fresh-process DDP replay' })
    print(json.dumps({'prepared':str(run),'start_step':args.step,'cache_estimate_gib':estimate/2**30}),flush=True)


def context(run):
    import torch
    from scripts.online_face_ba import verify
    from ba_dit.data.manifest import file_hash
    from ba_dit import training, distributed_training
    config, identity = verify(run)
    transition = json.loads((run/'execution_transition.json').read_text())
    prepared = run/'prepared_manifest.json'
    assert sha(prepared) == transition['prepared_manifest_sha256']
    rows = json.loads(prepared.read_text())
    for split in ('train','validation'):
        assert file_hash(config['data'][split+'_manifest']) == identity[split+'_manifest_sha256']
    def prepared_read(path, training=False, limit=None):
        split = 'train' if Path(path).resolve()==Path(config['data']['train_manifest']).resolve() else 'validation'
        if Path(path).resolve()!=Path(config['data'][split+'_manifest']).resolve():
            raise ValueError('Only the prepared immutable manifests may be read')
        return rows[split][:limit] if limit else rows[split]
    # Explicit executor input: these normalized rows were fully image-hashed on
    # CPU and matched the parent checkpoint's data digest. No GPU-time full scan.
    training.read_manifest = prepared_read
    distributed_training.read_manifest = prepared_read
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK','2')))
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    return config, transition, rows


def cache(run, until, probe=False):
    import gc
    import torch
    from safetensors.torch import load_file
    from ba_dit.config import digest
    from ba_dit.data.cache import cache_path, cache_spec, write_cache
    from ba_dit.runtime import backend_module
    from ba_dit.training import sample_at
    config, transition, rows = context(run)
    if not probe:
        from scripts.clust_gpu_metrics import cleanup_cache
        cleanup_cache(run, config, rows["validation"], until-2000)
    start = transition['start_step'] if probe else max(transition['start_step'],until-2000)
    count = 4 if probe else (until-start)*2
    cursor = transition['start_cursor']+(start-transition['start_step'])*2
    selected=[sample_at(rows['train'],i,config['training']['seed']) for i in range(cursor,cursor+count)]
    if probe:
        selected += rows['validation'] + [{**rows['validation'][0],'prompt':''}]
    backend=backend_module(config)
    checked=set();parity={'encoder':0,'vae':0};created=[]
    for stage in ('encoder','vae'):
        component=backend.load_encoder(config) if stage=='encoder' else backend.load_vae(config)
        encode=backend.encode_text if stage=='encoder' else backend.encode_images
        torch.cuda.reset_peak_memory_stats()
        for index,row in enumerate(selected):
            path=cache_path(config,row,stage)
            created.append(str(path))
            if not path.exists():
                if stage=='vae':
                    for field in ('reference','target'):
                        if row.get(field) and row[field] not in checked:
                            assert sha(row[field])==row[field+'_hash']
                            checked.add(row[field])
                with torch.no_grad(),torch.random.fork_rng(devices=[0]):
                    tensors,geometry=encode(component,config,row)
                write_cache(path,tensors,{'key':digest(cache_spec(config,row,stage)),
                    'spec':cache_spec(config,row,stage),**geometry})
            if probe and row in rows['validation']:
                old_path=cache_path(transition['old_config'],row,stage)
                old,new=load_file(old_path),load_file(path)
                assert old.keys()==new.keys() and all(torch.equal(old[k],new[k]) for k in old)
                parity[stage]+=1
            if index%25==0:print(json.dumps({'stage':stage,'completed':index+1,'total':len(selected)}),flush=True)
        peak=torch.cuda.max_memory_reserved()/torch.cuda.get_device_properties(0).total_memory
        assert peak<config['training']['max_reserved_fraction']
        del component
        gc.collect();torch.cuda.empty_cache()
    write(run/f'cache_files_{until}.json',created)
    if probe:
        assert parity=={'encoder':96,'vae':96}
        write(run/'conditioning_parity.json',{'fixed96_encoder_and_vae_exact':True,'samples':parity})


def ddp_probe(run, folder, until, checkpoint):
    from ba_dit.training import train_segment
    config,_,_=context(run)
    train_segment(config,'branch_only',folder,until,resume=checkpoint)


def compare_probe(run):
    import torch
    from safetensors.torch import load_file
    transition=json.loads((run/'execution_transition.json').read_text())
    step=transition['start_step']+2
    a,b=[run/name/f'checkpoint-{step:06d}' for name in ('probe_continuous','probe_resumed')]
    wa,wb=load_file(a/'adapters.safetensors'),load_file(b/'adapters.safetensors')
    assert wa.keys()==wb.keys() and all(torch.equal(wa[k],wb[k]) for k in wa)
    parent=load_file(run/f"checkpoint-{transition['start_step']:06d}"/'adapters.safetensors')
    assert all(not torch.equal(wa[k],parent[k]) for k in wa)
    sa,sb=[torch.load(p/'training_state.pt',map_location='cpu',weights_only=True) for p in (a,b)]
    def exact(x,y):
        if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x,y)
        if isinstance(x,dict):return x.keys()==y.keys() and all(exact(x[k],y[k]) for k in x)
        if isinstance(x,(tuple,list)):return len(x)==len(y) and all(exact(i,j) for i,j in zip(x,y))
        return x==y
    assert exact(sa,sb),'Full DDP optimizer/scaler/RNG replay differs'
    rows=[json.loads(line) for line in (run/'probe_continuous/metrics.jsonl').read_text().splitlines()]
    assert all(r['hardware/reserved_fraction']<.9 for r in rows)
    write(run/'execution_admission.json',{'exact_parameters_optimizer_scheduler_rng_cursor':True,
        'all_64_adapters_updated':True,'updates':2,'step':step,'effective_batch':2,
        'peak_reserved_gib':max(r['hardware/peak_reserved_gib'] for r in rows),
        'mean_seconds':sum(r['train/seconds'] for r in rows)/len(rows)})


def launch_ddp(run, action, *extra):
    subprocess.run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=2',
                    '-m','--','scripts.clust_staged',action,'--run',str(run),*map(str,extra)],check=True)


def worker(args):
    run=args.run.resolve()
    import fcntl
    lock_name=f'validation_{args.until}.lock' if args.action in ('score','face_quality','summarize') else 'gpu_stage.lock'
    lock=(run/lock_name).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    key=f'{args.action}_{args.until}'
    if (run/f'{key}.done.json').exists():return
    write(run/'status.json',{'status':'running','stage':key,'job_id':os.getenv('SLURM_JOB_ID')})
    os.environ.update(BA_STAGE_PROGRESS=str(run/'stage_progress.json'), BA_STAGE_ACTION=args.action, BA_STAGE_STEP=str(args.until), BA_ORT_THREADS='2')
    if args.action=='cache':cache(run,args.until)
    elif args.action=='admit':
        import yaml
        config,transition,_=context(run); start=transition['start_step']
        for name in ('probe_continuous','probe_resumed'):
            folder=run/name;folder.mkdir(exist_ok=True)
            (folder/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        launch_ddp(run,'probe_worker','--folder','probe_continuous','--until',start+2,'--checkpoint',run/f'checkpoint-{start:06d}')
        launch_ddp(run,'probe_worker','--folder','probe_resumed','--until',start+1,'--checkpoint',run/f'checkpoint-{start:06d}')
        launch_ddp(run,'probe_worker','--folder','probe_resumed','--until',start+2,'--checkpoint',run/'probe_resumed'/f'checkpoint-{start+1:06d}')
        compare_probe(run)
    elif args.action=='train':
        from ba_dit.training import train_segment
        from scripts.run_multi_id_face_ba import latest_checkpoint,archive_uncheckpointed_metrics
        config,_,_=context(run)
        assert json.loads((run/'execution_admission.json').read_text())['exact_parameters_optimizer_scheduler_rng_cursor']
        step=latest_checkpoint(run,args.until);archive_uncheckpointed_metrics(run,step)
        launch_ddp(run,'train_worker','--until',args.until,'--checkpoint',run/f'checkpoint-{step:06d}')
    elif args.action in ('infer','decode','summarize'):
        from scripts import online_face_ba
        config,_,_=context(run)
        from scripts import clust_stream_decode
        if args.action=='infer':
            subprocess.run([sys.executable,'-u','-c',
                'from pathlib import Path; from scripts import clust_staged, clust_stream_decode, online_face_ba; '
                'import sys; run=Path(sys.argv[1]); config,_,_=clust_staged.context(run); '
                'clust_stream_decode.infer(online_face_ba,run,config,int(sys.argv[2]))',str(run),str(args.until)],check=True)
            from scripts.clust_gpu_metrics import score
            score(ROOT,run,args.until)
        elif args.action=='decode' and clust_stream_decode.decoded(run,config,args.until):pass
        else:getattr(online_face_ba,args.action)(run,config,args.until)

    else:
        name='metrics' if args.action=='score' else 'face-quality'
        module='scripts.evaluate_metrics' if args.action=='score' else 'scripts.evaluate_face_quality'
        command=[str(ROOT/'envs/clust-v100'/name/'bin/python'),'-m',module,'--validation',str(run/f'validation-{args.until:06d}'),
                 '--log-dir',str(run),'--global-step',str(args.until),'--no-comet']
        if args.action=='score':command+=['--ownership-boxes',str(run/'ownership_boxes.json')]
        else:command+=['--threads',os.getenv('SLURM_CPUS_PER_TASK','4')]
        subprocess.run(command,check=True)
    write(run/f'{key}.done.json',{'finished_utc':datetime.now(timezone.utc).isoformat(),'job_id':os.getenv('SLURM_JOB_ID')})
    write(run/'status.json',{'status':'completed' if args.action=='summarize' and args.until==20000 else 'waiting',
                            'stage':key,'job_id':os.getenv('SLURM_JOB_ID')})


def submit(args):
    import fcntl
    run=args.run.resolve()
    path=ROOT/'scratch'/f'pipeline_{run.name}.json'
    lock=path.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():raise FileExistsError('Pipeline already submitted; inspect its exact job IDs')
    if not args.parent or not args.after:raise ValueError('Supply parent run and its checkpoint-stop job')
    stages=[('prepare',args.step,0,1,'02:00:00'),('probe_cache',0,1,2,'00:30:00'),('admit',0,2,4,'00:30:00')]
    for step in range((args.step//2000+1)*2000,20001,2000):
        stages.extend((name,step,gpus,cpus,limit) for name,gpus,cpus,limit in [
            ('cache',1,2,'01:15:00'),('train',2,4,'03:00:00'),('infer',1,2,'01:30:00'),
            ('decode',0,2,'00:30:00'),('score',0,4,'02:00:00'),('face_quality',0,4,'02:00:00'),
            ('summarize',0,1,'00:30:00')])
    record={'run':str(run),'parent':str(args.parent.resolve()),'jobs':[],'created_utc':datetime.now(timezone.utc).isoformat()}
    write(path,record)
    previous=args.after
    for index,(action,step,gpus,cpus,limit) in enumerate(stages):
        command=['sbatch','--parsable',f'--job-name=FLUX1-{action}-{step}',f'--cpus-per-task={cpus}',
                 f'--time={limit}',f'--dependency={"afterany" if index==0 else "afterok"}:{previous}',
                 f'--output={ROOT}/logs/%x-%j.out',f'--error={ROOT}/logs/%x-%j.err']
        if gpus:command.append(f'--gpus=v100:{gpus}')
        command += [str(ROOT/'jobs/flux1_clust_stage.sbatch'),action,'--run',str(run),'--until',str(step)]
        if action=='prepare':command+=['--parent',str(args.parent.resolve()),'--step',str(args.step)]
        reply=subprocess.run(command,text=True,capture_output=True,check=True)
        job=reply.stdout.strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('Uncertain submission; inspect queue before retrying')
        record['jobs'].append({'id':job,'action':action,'step':step,'gpus':gpus,'cpus':cpus,'command':command})
        write(path,record);previous=job
    print(json.dumps({'pipeline':str(path),'jobs':len(record['jobs']),'final_job_id':previous}),flush=True)


def status(args):
    pipeline=json.loads((ROOT/'scratch'/f'pipeline_{args.run.name}.json').read_text())
    jobs=pipeline['jobs']
    reply=subprocess.run(['sacct','-n','-X','-P','-j',','.join(j['id'] for j in jobs),'--format=JobID,State'],
                         text=True,capture_output=True,check=True,timeout=30)
    states={line.split('|')[0]:line.split('|')[1].split()[0] for line in reply.stdout.splitlines() if '|' in line}
    failed=next((j for j in jobs if states.get(j['id']) in {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'}),None)
    running_jobs=[j for j in jobs if states.get(j['id'])=='RUNNING']
    running=next((j for j in running_jobs if j.get('gpus',0)),next(iter(running_jobs),None))
    current=failed or running or next((j for j in jobs if states.get(j['id'])!='COMPLETED'),jobs[-1])
    state=states.get(current['id'],'PENDING')
    result={'job_id':pipeline.get('monitor_job_id',jobs[-1]['id']),'worker_job_id':current['id'],'slurm_state':state,
            'active':not failed and (bool(pipeline.get('remaining_stages')) or any(states.get(j['id'])!='COMPLETED' for j in jobs)),
            'stage':f"{current['action']}_{current['step']}",'observed_unix_seconds':time.time()}
    result['parallel_stages']=[{'job_id':j['id'],'stage':f"{j['action']}_{j['step']}"} for j in running_jobs if j!=current]
    artifact=args.run if (args.run/'comet_experiment.json').exists() else Path(pipeline.get('setup',args.run))
    result['artifact_run_name']=artifact.name
    progress=artifact/'stage_progress.json'
    if pipeline.get('setup'):progress=Path(pipeline['setup'])/'stage_progress.json'
    if progress.exists():
        detail=json.loads(progress.read_text())
        if detail.get('action')==current['action'] and detail.get('checkpoint')==current['step']:
            result['progress']=detail
    if artifact.exists():write(artifact/'comet-live-status.json',result)
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['submit','status','prepare','cache','admit','continue','probe_cache','probe_worker','train_worker','probe_compare',
                                   'train','infer','decode','score','face_quality','summarize'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--parent',type=Path)
    p.add_argument('--step',type=int,default=1000);p.add_argument('--until',type=int,default=0)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--folder')
    p.add_argument('--after')
    a=p.parse_args()
    if a.action=='submit':submit(a)
    elif a.action=='status':status(a)
    elif a.action=='prepare':prepare(a)
    elif a.action=='probe_cache':cache(a.run.resolve(),0,probe=True)
    elif a.action in ('probe_worker','train_worker'):
        ddp_probe(a.run.resolve(),a.run.resolve()/a.folder if a.folder else a.run.resolve(),a.until,a.checkpoint)
    elif a.action=='probe_compare':compare_probe(a.run.resolve())
    elif a.action=='continue':
        # Prepared inputs let admission and the first segment share one two-GPU
        # allocation. Every later segment still requires the admission receipt.
        worker(argparse.Namespace(**{**vars(a),'action':'admit','until':0}))
        worker(argparse.Namespace(**{**vars(a),'action':'train'}))
    else:worker(a)


if __name__=='__main__':main()
