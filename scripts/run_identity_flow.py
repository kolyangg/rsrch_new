"""One GPU, serial identity-aware BA training and generated-image validation."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ba_dit.config import ROOT
from ba_dit.data.manifest import file_hash
from scripts.face_crop_flow import write
from scripts.run_reference_refiner import convergence


def main(run):
    from scripts.identity_face_flow import experiment
    from scripts.masked_face_flow import verify
    config,identity=verify(run)
    assert identity['head']['kind']=='deep_identity'
    lock=(ROOT/'runs/face_flow_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan={'image_validation_steps':[0,500,2000,*range(5000,config['training']['steps']+1,5000)],
          'minimum_id_gain':.003,'patience':2,'safety_limit':config['training']['steps'],
          'note':'Full fixed 24-image generation; image ID gain decides continuation, not the identity training loss.',
          'controller_sha256':file_hash(Path(__file__))}
    plan_path=run/'execution_plan.json'
    if plan_path.exists():assert json.loads(plan_path.read_text())==plan
    else:write(plan_path,plan)
    completed_path=run/'completed_commands.json'
    completed=json.loads(completed_path.read_text()) if completed_path.exists() else []
    status={'controller_pid':os.getpid(),'status':'running','run':str(run)}

    def update(**values):
        status.update(values,updated_at=datetime.now(timezone.utc).isoformat())
        write(run/'status.json',status)

    def command(action,step,resume=0):
        key=f'{action}_{step}'
        if key in completed:return
        update(stage=action,target_step=step)
        with (run/f'{key}.log').open('a') as stream:
            child=subprocess.Popen([sys.executable,'-m','scripts.identity_face_flow',action,'--run',str(run),
                  '--step',str(step),'--resume',str(resume)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            update(child_pid=child.pid);code=child.wait()
        if code:raise RuntimeError(f'{key} exited {code}; inspect its log before restarting')
        completed.append(key);write(completed_path,completed)
        if action=='train':update(training_steps=step)
        update(child_pid=None)

    scores={}
    try:
        command('train',50)
        previous=50
        exp=experiment(config,run);exp.log_asset(str(plan_path));exp.end()
        for step in plan['image_validation_steps']:
            if step:
                command('train',step,previous);previous=step
            for action in ('infer','decode','score'):command(action,step)
            folder=run/f'validation-{step:06d}'
            summary=json.loads((folder/'quality_summary.json').read_text())
            validation=json.loads((folder/'validation.json').read_text())
            checkpoint=run/f'checkpoint-{step:06d}/branch.safetensors' if step else run/'initial.safetensors'
            assert len(validation['samples'])==24 and validation['checkpoint_sha256']==file_hash(checkpoint)
            scores[step]=summary['metrics']['id_sim'];decision=convergence(scores)
            write(run/'convergence.json',decision)
            best=decision['best_step']
            best_path=run/f'checkpoint-{best:06d}/branch.safetensors' if best else run/'initial.safetensors'
            write(run/'best_checkpoint.json',{'step':best,'id_sim':decision['best_id_sim'],
                  'file':str(best_path),'sha256':file_hash(best_path)})
            update(last_completed_validation=step,**decision)
            exp=experiment(config,run)
            exp.log_metrics({'convergence/best_id_sim':decision['best_id_sim'],
                'convergence/small_gain_checks':decision['consecutive_small_gains']},step=step)
            exp.log_asset(str(run/'convergence.json'));exp.log_asset(str(run/'best_checkpoint.json'));exp.end()
            print(json.dumps({'validated_step':step,**decision}),flush=True)
            if decision['plateau']:
                update(status='completed',stage='ID plateau',child_pid=None);break
        else:update(status='limit_reached',stage='safety limit; convergence not established',child_pid=None)
    except BaseException as error:
        update(status='failed',error=str(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
