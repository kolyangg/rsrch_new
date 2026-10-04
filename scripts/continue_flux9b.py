"""Continue the completed fixed12 9B pilot with intact Adam/RNG/data cursor."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml
from ba_dit.config import ROOT, adapter_identity, config_digest, digest, load_config
from ba_dit.data.manifest import file_hash


def initialize(parent, run, target):
    from scripts.online_face_ba import SOURCES, write
    from ba_dit.checkpoint import training_code_digest
    from ba_dit.continuation import verify_extension
    identity=json.loads((parent/'identity.json').read_text())
    for path,sha in identity['source_sha256'].items():
        assert file_hash(parent.parents[1]/path)==sha, path
    assert json.loads((parent/'status.json').read_text())['status']=='completed'
    checkpoint=parent/'checkpoint-002000'
    manifest=json.loads((checkpoint/'manifest.json').read_text())
    config=load_config(checkpoint/'resume_config.yaml')
    assert manifest['step']==2000 and manifest['config_sha256']==config_digest(config)
    assert manifest['training_code_sha256']==training_code_digest(config)
    assert manifest['identity']==adapter_identity(config)
    for split in ('train','validation'):
        assert file_hash(config['data'][split+'_manifest'])==identity[split+'_manifest_sha256']
    config['training'].update(steps=target,validation_every=2000)
    verify_extension(checkpoint,config,manifest['config_sha256'])
    run.mkdir(parents=True,exist_ok=False)
    (run/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    for step in (0,1000,2000):
        # Parent checkpoints stay byte-identical; they are read-only inputs here.
        (run/f'checkpoint-{step:06d}').symlink_to(parent/f'checkpoint-{step:06d}',target_is_directory=True)
        shutil.copytree(parent/f'validation-{step:06d}',run/f'validation-{step:06d}')
    shutil.copytree(parent/'native',run/'native')
    for name in ('routing_masks.json','ownership_boxes.json','mask_overlays.png','metrics.jsonl'):
        shutil.copyfile(parent/name,run/name)
    sources=(*SOURCES,'ba_dit/continuation.py','scripts/continue_flux9b.py')
    for path in sources:
        dest=run/'source_snapshot'/path;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/path,dest)
    new_identity={**identity,'config_sha256':digest(config),'base':adapter_identity(config),
                  'source_sha256':{p:file_hash(ROOT/p) for p in sources},'parent_run':str(parent)}
    write(run/'identity.json',new_identity)
    write(run/'continuation.json',{'parent':str(parent),'parent_comet':json.loads((parent/'comet_experiment.json').read_text()),
        'start_step':2000,'target_step':target,'validation_steps':list(range(4000,target+1,2000)),
        'changed_config_fields':['training.steps','training.validation_every'],
        'parent_checkpoint_sha256':{p.name:file_hash(p) for p in checkpoint.iterdir() if p.is_file()},
        'optimizer_rng_cursor_preserved':True,'imported_panels':[0,1000,2000]})
    (run/'latest_checkpoint.txt').write_text(str(checkpoint)+'\n')


def main(args):
    from scripts.online_face_ba import experiment, verify, write
    from scripts.run_multi_id_face_ba import latest_checkpoint, archive_uncheckpointed_metrics
    run=args.run.resolve()
    if args.worker:
        from ba_dit.training import train_segment
        config,_=verify(run)
        resume=latest_checkpoint(run,args.until)
        archive_uncheckpointed_metrics(run,resume)
        exp=experiment(config,run)
        exp.log_asset(str(run/'continuation.json'))
        exp.log_parameters({'continuation/start_step':2000,'continuation/optimizer_resumed':True})
        exp.end()
        train_segment(config,'branch_only',run,args.until,resume=run/f'checkpoint-{resume:06d}')
        return
    lock=(ROOT/'runs/face_flow_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);os.set_inheritable(lock.fileno(),True)
    if not args.resume:initialize(args.parent.resolve(),run,args.target)
    config,_=verify(run)
    (run/'controller.pid').write_text(str(os.getpid())+'\n')
    def call(key,command):
        receipt=run/f'{key}.done.json'
        if receipt.exists():return
        verify(run)
        with (run/f'{key}.log').open('a') as log:
            child=subprocess.Popen(list(map(str,command)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
            write(run/'status.json',{'status':'running','stage':key,'controller_pid':os.getpid(),'child_pid':child.pid})
            code=child.wait()
        if code:
            write(run/'status.json',{'status':'failed','stage':key,'exit_code':code})
            raise RuntimeError(f'{key} failed; inspect its log')
        write(receipt,{'command':list(map(str,command))})
    for step in range(4000,config['training']['steps']+1,2000):
        call(f'train_{step}',[sys.executable,'-m','scripts.continue_flux9b','--run',run,'--worker','--until',step])
        for action in ('infer','decode'):
            call(f'{action}_{step}',[sys.executable,'-m','scripts.online_face_ba',action,'--run',run,'--step',step])
        call(f'score_{step}',[ROOT/'envs/metrics/bin/python','-m','scripts.evaluate_metrics','--validation',run/f'validation-{step:06d}',
            '--log-dir',run,'--global-step',step,'--ownership-boxes',run/'ownership_boxes.json'])
        call(f'summarize_{step}',[sys.executable,'-m','scripts.online_face_ba','summarize','--run',run,'--step',step])
    write(run/'status.json',{'status':'completed','training_steps':config['training']['steps'],'last_completed_validation':config['training']['steps']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--target',type=int,default=8000)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--worker',action='store_true')
    p.add_argument('--until',type=int)
    a=p.parse_args()
    if a.target<=2000 or a.target%2000:p.error('target must be a multiple of2000 above2000')
    main(a)
