"""Serial local16GB controller: fixed24 at 0/500/1000/2000, then stop."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from ba_dit.config import ROOT
from scripts.online_face_ba import verify, write, validation_steps


def main(run):
    config,_=verify(run)
    lock=(ROOT/'runs/face_flow_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (run/'controller.pid').write_text(str(os.getpid())+'\n')
    steps=validation_steps(config)
    assert config['training']['steps']==steps[-1]
    completed_path=run/'completed_commands.json'
    completed=json.loads(completed_path.read_text()) if completed_path.exists() else []
    status={'controller_pid':os.getpid(),'run':str(run),'status':'running'}
    write(run/'execution_plan.json',{'validation_steps':steps,'optimizer_updates':2000,
        'microbatch':1,'gradient_accumulation':config['training']['grad_accum'],
        'stop_policy':'Stop after 2000 and final scoring; inspect results before extending.'})
    def update(**fields):
        status.update(fields,updated_at=datetime.now(timezone.utc).isoformat())
        write(run/'status.json',status)
    def call(action,step,resume=0):
        key=f'{action}_{step}'
        if key in completed:return
        verify(run)
        update(stage=action,target_step=step)
        if action=='score':
            command=[str(ROOT/'envs/metrics/bin/python'),'-m','scripts.evaluate_metrics',
                '--validation',str(run/f'validation-{step:06d}'),'--ownership-boxes',str(run/'ownership_boxes.json'),
                '--log-dir',str(run),'--global-step',str(step)]
        else:
            command=[sys.executable,'-m','scripts.online_face_ba',action,'--run',str(run),
                     '--step',str(step),'--resume',str(resume)]
        with (run/f'{key}.log').open('a') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            update(child_pid=child.pid);code=child.wait()
        if code:raise RuntimeError(f'{key} failed with exit {code}; inspect {key}.log')
        completed.append(key);write(completed_path,completed);update(child_pid=None)
    try:
        previous=0
        for step in steps:
            if step:
                call('train',step,previous);previous=step
                update(training_steps=step)
            for action in ('infer','decode','score','summarize'):call(action,step)
            update(last_completed_validation=step)
        update(status='completed',stage='2000 updates and validation/scoring complete')
    except BaseException as error:
        update(status='failed',error=str(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
