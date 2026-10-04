"""Start the approved 9B pilot immediately; evaluate preserved step zero at 1000."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml
from ba_dit.config import ROOT, adapter_identity, digest, load_config
from ba_dit.data.manifest import file_hash


def main(args):
    from scripts.online_face_ba import SOURCES, write, verify
    from scripts.run_multi_id_face_ba import latest_checkpoint, archive_uncheckpointed_metrics, freeze_native
    run=args.run.resolve()
    if args.worker:
        from ba_dit.training import train_segment
        config=load_config(run/'resolved_config.yaml')
        from scripts.online_face_ba import experiment
        exp=experiment(config,run)
        exp.log_parameters({'validation/step_zero_evaluated_after_update':1000,'startup/immediate_training':True})
        exp.end()
        checkpoints=list(run.glob('checkpoint-*/training_state.pt'))
        resume=latest_checkpoint(run,args.until) if checkpoints else None
        if resume is not None: archive_uncheckpointed_metrics(run,resume)
        probe = None
        if args.profile_validation:
            from scripts.profile_validation_budget import profile
            probe = profile
        train_segment(config,'branch_only',run,args.until,
                      resume=run/f'checkpoint-{resume:06d}' if resume is not None else None,
                      save_initial=resume is None,timing_probe=probe)
        return
    (ROOT/'runs').mkdir(exist_ok=True)
    lock=(ROOT/'runs/face_flow_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.set_inheritable(lock.fileno(),True)
    config=load_config(run/'resolved_config.yaml' if args.resume else args.config)
    assert config['model']['arch']=='flux2_klein_9b' and config['branch']['kind']=='masked_face_qkvo'
    assert config['training']['steps']==2000 and config['validation']['limit']==12
    assert config['data']['conditioning']=='online' and config['logging']['enabled']
    sources=(*SOURCES,'scripts/run_flux9b_training_first.py','scripts/run_multi_id_face_ba.py','ba_dit/validation_masks.py','ba_dit/data/conditioning.py')
    if args.profile_validation:sources+=('scripts/profile_validation_budget.py',)
    guard={'config_sha256':digest(config),'base':adapter_identity(config),
           'source_sha256':{p:file_hash(ROOT/p) for p in sources},
           'train_manifest_sha256':file_hash(config['data']['train_manifest']),
           'validation_manifest_sha256':file_hash(config['data']['validation_manifest'])}
    if args.resume:
        assert json.loads((run/'training_identity.json').read_text())==guard
    else:
        run.mkdir(parents=True,exist_ok=False)
        (run/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        write(run/'training_identity.json',guard)
        for p in sources:
            target=run/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/p,target)
        write(run/'execution_plan.json',{'steps':2000,'validation_steps':[0,1000,2000],
            'step_zero_validation_deferred_until':1000,'initial_checkpoint_preserved':True,
            'reason':'User explicitly requested immediate training; pretrained native/off, gradient and frozen checks already passed; fresh-process replay incomplete.'})
    (run/'controller.pid').write_text(str(os.getpid())+'\n')
    def call(key,cmd):
        receipt=run/f'{key}.done.json'
        if receipt.exists():return
        for p,sha in guard['source_sha256'].items():assert file_hash(ROOT/p)==sha
        write(run/'status.json',{'stage':key,'status':'running','controller_pid':os.getpid()})
        with (run/f'{key}.log').open('a') as log:
            child=subprocess.Popen(list(map(str,cmd)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
            write(run/'status.json',{'stage':key,'status':'running','controller_pid':os.getpid(),'child_pid':child.pid})
            code=child.wait()
        if code:
            write(run/'status.json',{'stage':key,'status':'failed','exit_code':code})
            raise RuntimeError(f'{key} failed; inspect {run}/{key}.log')
        write(receipt,{'command':list(map(str,cmd))})
    python=sys.executable
    def train(until):call(f'train_{until}',[python,'-m','scripts.run_flux9b_training_first','--run',run,'--worker','--until',until]+(['--profile-validation'] if args.profile_validation else []))
    train(1000)
    cli=[python,'-m','ba_dit.cli']
    native=run/'native_baseline'
    call('cache_validation',cli+['precompute','--config',run/'resolved_config.yaml','--split','validation'])
    command=cli+['infer','--config',run/'resolved_config.yaml','--mode','native','--output-dir',native,'--no-comet','--no-quality-metrics','--skip-output-masks']
    if native.exists():command+=['--resume-validation']
    call('native12',command)
    metrics=ROOT/'envs/metrics/bin/python'
    call('masks',[metrics,ROOT/'scripts/build_validation_output_masks.py','--validation',native])
    call('native_score',[metrics,'-m','scripts.evaluate_metrics','--validation',native,'--no-comet'])
    from ba_dit.data.manifest import read_manifest
    if not (run/'identity.json').exists():
        freeze_native(config,native,run,read_manifest(config['data']['validation_manifest']))
        shutil.copyfile(run/'quality_summary.json',run/'native/quality_summary.json')
        write(run/'identity.json',{**guard,'routing_masks_sha256':file_hash(run/'routing_masks.json'),
            'variant':'online_masked_qkvo_v1','split_policy':'identity_disjoint',
            'initial_validation_deferred':True})
    def validate(step):
        verify(run)
        for action in ('infer','decode'):
            call(f'{action}_{step}',[python,'-m','scripts.online_face_ba',action,'--run',run,'--step',step])
        call(f'score_{step}',[metrics,'-m','scripts.evaluate_metrics','--validation',run/f'validation-{step:06d}',
            '--log-dir',run,'--global-step',step,'--ownership-boxes',run/'ownership_boxes.json'])
        call(f'summarize_{step}',[python,'-m','scripts.online_face_ba','summarize','--run',run,'--step',step])
    validate(0);validate(1000)
    train(2000);validate(2000)
    write(run/'status.json',{'status':'completed','training_steps':2000,'last_completed_validation':2000})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--worker',action='store_true')
    p.add_argument('--until',type=int)
    p.add_argument('--profile-validation',action='store_true')
    main(p.parse_args())
