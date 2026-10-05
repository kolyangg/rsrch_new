"""One authorized FLUX1a launch, after the historical fixed96 replay completes."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ba_dit.config import ROOT, load_config
from ba_dit.logging import connect
from scripts.online_face_ba import write


def main():
    if ROOT != Path('/workspace/rsrch_FLUX1abc'):
        raise RuntimeError('This deployment controller is scoped to the isolated GB10 checkout')
    run = ROOT/'runs/FLUX1a_vast9B_20261004'
    setup = run.with_name(run.name+'_setup')
    setup.mkdir(parents=True,exist_ok=True)
    config_path = ROOT/'configs/FLUX1a_vast_9b.yaml'
    config = load_config(config_path)
    experiment = connect(config,setup,name=run.name)
    try:
        experiment.log_other('launch/authorized','User requested FLUX1a immediately after the existing 2k validation')
        experiment.log_other('launch/checkout',str(ROOT))
        experiment.log_other('launch/stage','preparing_identity_supervision')
    finally:
        experiment.end()
    write(setup/'launch_request.json',{'requested_at_utc':datetime.now(timezone.utc).isoformat(),
        'instance_id':53994096,'experiment':'FLUX1a','run':str(run),'updates':4000,
        'validation_steps':[0,2000,4000],'new_training_authorized':True})
    envs=Path(os.getenv('BA_ENVS_DIR',ROOT/'envs'))
    command=[str(envs/'metrics/bin/python'),'-m','scripts.prepare_flux1_identity',
             '--config',str(config_path),'--workers','4','--threads','2','--scheduled-only']
    write(setup/'status.json',{'status':'running','stage':'prepare_identity','controller_pid':os.getpid()})
    with (setup/'prepare_identity.log').open('a') as stream:
        subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,
                       env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'})
    previous=Path('/workspace/rsrch_9b_8k/runs/FLUX1_vast9B_2000_fixed96_20261004')
    while not (previous/'ready.json').exists():
        write(setup/'status.json',{'status':'waiting','stage':'previous_validation_publication','controller_pid':os.getpid()})
        time.sleep(30)
    assert json.loads((previous/'comet_verified.json').read_text())['verified']
    command=[sys.executable,'-m','scripts.run_flux1_experiment','--experiment','FLUX1a',
             '--action','run','--run',str(run),'--native-bundle',str(ROOT/'runs/native_fixed96_bundle'), '--resume']
    subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        setup=ROOT/'runs/FLUX1a_vast9B_20261004_setup'
        if setup.exists():write(setup/'status.json',{'status':'failed','error':str(error)})
        raise
