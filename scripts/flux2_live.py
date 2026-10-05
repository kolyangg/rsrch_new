"""Live Comet session and batch-image publishing around an immutable local run.

AICODE-NOTE: Execution-only overlay. Never edits the run identity, configuration,
model, optimizer, sampler, checkpoint schedule, or historical source snapshot.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

from ba_dit.config import ROOT
from ba_dit.data.manifest import file_hash


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def verified_policy(run):
    policy=read(run/'live_execution_policy.json')
    for name,sha in policy['source_sha256'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError(f'Live execution source changed: {name}')
    if read(run/'comet_experiment.json')['experiment_key']!=policy['experiment_key']:
        raise ValueError('Comet experiment identity changed')
    return policy


def latest_metric(run):
    for line in reversed((run/'metrics.jsonl').read_text().splitlines()):
        try:return json.loads(line)
        except json.JSONDecodeError:continue
    return {'step':0}


def comet(run):
    for line in (ROOT/'.env').read_text().splitlines():
        if line.startswith('COMET_API_KEY='):
            os.environ.setdefault('COMET_API_KEY',line.split('=',1)[1].strip().strip('"\''))
    from comet_ml.api import API
    return API(cache=False).get_experiment_by_key(verified_policy(run)['experiment_key'])


def publish_images(api, run, limit, receipts):
    """Upload only complete, hashed batch outputs; retry unacknowledged uploads."""
    for path in sorted(run.glob('stream_decode_*.json')):
        step=int(path.stem.rsplit('_',1)[1]);decoded=read(path)
        report=read(run/f'validation-{step:06d}/validation.json')
        rows={row['sample_id']:row for row in report['samples']}
        for key,saved in decoded['samples'].items():
            token=f'{step}/{key}'
            if token in receipts:
                assert receipts[token]['image_sha256']==saved['image_sha256']
                continue
            assert saved['checkpoint_sha256']==report['checkpoint_sha256']
            image=run/f'validation-{step:06d}'/rows[key]['image']
            assert file_hash(image)==saved['image_sha256']
            result=api.log_image(str(image),image_name=f'fixed{limit}/{key}',step=step,
                                metadata={'prompt':rows[key]['prompt'],'seed':rows[key]['seed']})
            if result is None:raise RuntimeError('Comet did not acknowledge image upload')
            receipts[token]={'image_sha256':saved['image_sha256'],'uploaded_at':datetime.now(timezone.utc).isoformat()}
            write(run/'live_uploaded_images.json',receipts)
            print(f'Comet {step}: published {key} immediately after batch decode',flush=True)


def publisher(run):
    from ba_dit.config import load_config
    policy=verified_policy(run);config=load_config(run/'resolved_config.yaml');api=comet(run)
    path=run/'live_uploaded_images.json';receipts=read(path) if path.exists() else {}
    while True:
        state=read(run/'status.json');last=latest_metric(run)
        active=any(subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode==0
                   for unit in policy['controller_units'])
        status=state['status'] if active or state['status']=='completed' else 'failed_or_stopped'
        progress={'status':status,'stage':state.get('stage'),'optimizer_step':last['step'],
                  'validation_checkpoint':state.get('target_step'),'checked_at':datetime.now(timezone.utc).isoformat()}
        try:
            if status=='running':
                api.set_state('running');api.update_status()
            api.log_other('runtime/live_progress',json.dumps(progress))
            api.log_metric('runtime/optimizer_step',last['step'],step=last['step'])
            publish_images(api,run,config['validation']['limit'],receipts)
            progress['published_batch_images']=len(receipts)
            write(run/'live_progress.json',progress)
            if status!='running':
                api.set_state('finished' if status=='completed' else 'crashed')
                return
        except Exception as error:
            print(f'Comet publisher retry: {type(error).__name__}: {error}',flush=True)
        time.sleep(5)


def wrap_command(command):
    command=list(command)
    if len(command)>2 and command[1:3] in [['-m','scripts.online_face_ba'],['-m','scripts.evaluate_metrics']]:
        return command[:2]+['scripts.flux2_live','worker',command[2],*command[3:]]
    return command


def worker(module, arguments):
    # Flush stage metrics normally; only the publisher owns terminal run status.
    from comet_ml._online import Experiment
    Experiment._mark_as_ended=lambda self:None
    if module!='scripts.online_face_ba':
        sys.argv=[module,*arguments];runpy.run_module(module,run_name='__main__');return
    from scripts import online_face_ba as m
    from scripts.clust_stream_decode import StreamingDecoder
    original_infer=m.infer;original_decode=m.decode
    def infer(run,config,step):
        verified_policy(run)
        decoder=StreamingDecoder(m,run,config,step);original_write=m.write
        def completed(path,value):
            original_write(path,value)
            if Path(path)==decoder.folder/'validation.json':decoder.update(value)
        m.write=completed
        try:original_infer(run,config,step)
        finally:m.write=original_write
    def decode(run,config,step):
        path=run/f'stream_decode_{step}.json'
        if not path.exists():return original_decode(run,config,step)
        receipt=read(path);report=read(run/f'validation-{step:06d}/validation.json')
        assert len(receipt['samples'])==len(report['samples'])==config['validation']['limit']
        for row in report['samples']:
            saved=receipt['samples'][row['sample_id']]
            assert saved['checkpoint_sha256']==report['checkpoint_sha256']
            assert file_hash(run/f'validation-{step:06d}'/row['image'])==saved['image_sha256']
        exp=m.experiment(config,run)
        try:exp.log_asset(str(run/f'validation-{step:06d}/validation.json'),file_name=f'validation_{step:06d}.json')
        finally:exp.end()
        print(f'Validation {step}: all {len(report["samples"])} images decoded per batch',flush=True)
    m.infer,m.decode=infer,decode
    p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--run',type=Path,required=True)
    p.add_argument('--step',type=int,default=0);p.add_argument('--resume',type=int,default=0)
    m.main(p.parse_args(arguments))


def controller(run):
    verified_policy(run)
    original=subprocess.Popen
    class LivePopen(original):
        def __init__(self,command,*args,**kwargs):super().__init__(wrap_command(command),*args,**kwargs)
    subprocess.Popen=LivePopen
    from scripts.run_online_face_ba import main
    main(run)


def handoff(run, old_unit):
    """Wait for a completed training segment; only interrupt restartable validation."""
    verified_policy(run)
    while True:
        state=read(run/'status.json');completed=read(run/'completed_commands.json')
        if state['status']!='running':raise RuntimeError('Old runner stopped before safe handoff')
        if state.get('stage')=='infer' and f'train_{state["target_step"]}' in completed:
            break
        if subprocess.run(['systemctl','--user','is-active','--quiet',old_unit]).returncode:
            raise RuntimeError('Original controller is inactive; no safe handoff was made')
        time.sleep(1)
    step=state['target_step'];checkpoint=run/f'checkpoint-{step:06d}'
    assert read(checkpoint/'manifest.json')['step']==step and (checkpoint/'training_state.pt').is_file()
    subprocess.run(['systemctl','--user','stop',old_unit],check=True)
    # Preserve any partial latent interrupted during an atomicity-free old save.
    from safetensors.torch import load_file
    from safetensors import safe_open
    expected=file_hash(checkpoint/'adapters.safetensors')
    folder=run/f'validation-{step:06d}'
    for path in folder.glob('*.safetensors'):
        try:
            with safe_open(path,framework='pt') as f:assert f.metadata()['checkpoint_sha256']==expected
            load_file(path)
        except Exception:
            quarantine=run/'interrupted_validation_files';quarantine.mkdir(exist_ok=True)
            path.rename(quarantine/f'{step}_{path.name}')
    write(run/'live_handoff.json',{'checkpoint_step':step,'checkpoint_sha256':expected,
          'old_unit':old_unit,'training_updates_discarded':0,'at':datetime.now(timezone.utc).isoformat()})
    print(f'Continuing at checkpoint {step}; no optimizer updates discarded',flush=True)
    controller(run)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3:])
    else:
        p=argparse.ArgumentParser();p.add_argument('action',choices=('publisher','controller','handoff'))
        p.add_argument('--run',type=Path,required=True);p.add_argument('--old-unit')
        args=p.parse_args();run=args.run.resolve()
        if args.action=='handoff':handoff(run,args.old_unit)
        else:globals()[args.action](run)
