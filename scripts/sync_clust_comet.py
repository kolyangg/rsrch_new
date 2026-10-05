"""Relay one cluster run's completed outputs and live metrics through this workstation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def sync(run_name, destination, remote_root='/home/nasilaev/rsrch_new'):
    includes = ['metrics.jsonl', 'status.json', 'comet-live-status.json', 'comet_experiment.json',
                'resolved_config.yaml', 'resume_latest.json', 'uncheckpointed_metrics_*.jsonl',
                'execution_transition.json', 'execution_admission.json', 'conditioning_parity.json',
                'native_checks.json', 'resume_parity.json', 'gpu_scoring_admission.json', 'deployment.json', 'stage_progress.json',
                '*.done.json', 'paired_faces_*.png', 'mask_overlays.png']
    command = ['redshield-vpn', 'exec', 'russia', '--', 'rsync', '-rtL', '--partial',
               '--delay-updates', '--timeout=90', *[f'--include=/{p}' for p in includes],
               '--include=/validation-*/', '--exclude=*_raw.png', '--include=/validation-*/*.png',
               '--include=/validation-*/*.json', '--include=/validation-*/*.csv', '--exclude=*',
               f'clust:{remote_root}/runs/{run_name}/', str(destination)+'/']
    subprocess.run(command, check=True, timeout=180, capture_output=True)


def pipeline_status(run_name, remote_root):
    # A brief login-node scheduler query needs no idle CPU/GPU allocation.
    command=(f'cd {shlex.quote(remote_root)} && '
             'source /home/nasilaev/rsrch_new/scripts/activate_clust_env.sh && '
             f'BA_ROOT={shlex.quote(remote_root)} PYTHONPATH={shlex.quote(remote_root)} '
             f'python -m scripts.clust_staged status --run runs/{run_name}')
    result=subprocess.run(['redshield-vpn','exec','russia','--','ssh','clust','bash -lc '+shlex.quote(command)],
                          text=True,capture_output=True,check=True,timeout=90)
    return json.loads(next(line for line in reversed(result.stdout.splitlines()) if line.startswith('{')))


def training_rows(folder):
    path = folder/'metrics.jsonl'
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n')]
    if rows and [r['step'] for r in rows] != list(range(1, rows[-1]['step']+1)):
        raise ValueError('Expected contiguous optimizer updates starting at one')
    if any(not math.isfinite(v) for r in rows for k,v in r.items() if k != 'step'):
        raise ValueError('Refusing nonfinite training metrics')
    return rows


def completed_outputs(folder):
    from PIL import Image
    images, assets, summaries = [], [], []
    if (folder/'mask_overlays.png').exists():
        images.append((folder/'mask_overlays.png', 'native_face_masks', 0, {}))
    for path in sorted(folder.glob('validation-*')):
        step = int(path.name.split('-')[1])
        try:
            report = json.loads((path/'validation.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        complete = (folder/f'decode_{step}.done.json').exists()
        if len(report['samples']) > 96 or (complete and len(report['samples']) != 96):
            raise ValueError('Cluster run requires its original fixed96 panel')
        for row in report['samples']:
            image_path = path/row['image']
            if not image_path.is_file():
                if complete:raise FileNotFoundError(image_path)
                continue
            # rsync may observe a PNG while the decoder is still writing it.
            # Only publish complete files; the next transfer retries partial ones.
            try:
                with Image.open(image_path) as decoded:decoded.verify()
            except (OSError, SyntaxError):
                if complete:raise
                continue
            images.append((image_path, 'fixed96/'+row['sample_id'], step,
                           {'prompt':row['prompt'], 'seed':row['seed']}))
        if not complete:continue
        if (folder/f'summarize_{step}.done.json').exists():
            for image_path in sorted(folder.glob(f'paired_faces_{step:06d}_*.png')):
                images.append((image_path, 'paired_faces/'+image_path.stem.rsplit('_',1)[1], step, {}))
        for name, prefix in [('quality_summary.json','validation/'), ('face_quality_summary.json','face_quality/')]:
            if (path/name).exists():
                values = json.loads((path/name).read_text())['metrics']
                summaries.append((step, {prefix+k:v for k,v in values.items() if v is not None}))
        assets.extend((p, f'{path.name}/{p.name}', step) for p in path.iterdir() if p.suffix in {'.json','.csv'})
    return images, assets, summaries


class Publisher:
    def __init__(self, folder, key, metric_prefix='', metric_every=2, clean_console=False):
        from comet_ml import API, ExistingExperiment
        self.folder, self.key = folder, key
        self.metric_prefix, self.metric_every = metric_prefix, metric_every
        self.clean_console = clean_console
        self.summary_steps = {}
        self.summary_queued = {}
        self.transition_logged = False
        self.loss_metric = metric_prefix+'train/loss'
        self.api = API(cache=False).get_experiment_by_key(key)
        if self.api is None:
            raise RuntimeError('The existing Comet experiment was not found')
        self.experiment = ExistingExperiment(previous_experiment=key, auto_param_logging=False,
            auto_metric_logging=False, log_env_details=False, log_code=False, log_git_metadata=False,
            log_git_patch=False, log_graph=False, auto_output_logging=None)
        self.experiment.log_other('cluster/artifact_transport', 'workstation rsync relay')
        self.experiment.log_other('cluster/active_training_metric', self.loss_metric)
        self.experiment.log_other('cluster/metric_every', metric_every)
        self.steps = {int(r['step']) for r in self.api.get_metrics(self.loss_metric) if r['step'] is not None}
        self.receipt = folder/'publisher_verified.json'
        self.delivery_path = folder/'asset_delivery.json'
        self.delivery = json.loads(self.delivery_path.read_text()) if self.delivery_path.exists() else {}
        self.sent_steps = set(self.steps)
        self.metric_queued = {}

    def deliver(self, images, assets, budget=24):
        """Bounded, durable outbox: delayed acknowledgements never restart uploads."""
        items = [('image',p,n,s,m) for p,n,s,m in sorted(images,key=lambda r:r[2],reverse=True)]
        items += [('asset',p,n,s,{}) for p,n,s in assets]
        confirmed = 0
        names = {}
        def lookup(name,kind):
            if (name,kind) not in names:
                names[name,kind]=self.api._api._client.get_experiment_assets_list_by_name(
                    self.key,name,asset_type='image' if kind=='image' else None,timeout=15) or []
            return names[name,kind]
        for kind,path,name,step,metadata in items:
            key=json.dumps([kind,name,step])
            record=self.delivery.get(key,{})
            content_hash=hashlib.sha256(path.read_bytes()).hexdigest() if kind=='asset' else None
            if content_hash and record.get('sha256') not in (None,content_hash):
                record={'remote_name':f'{name}.sha256-{content_hash[:12]}'}
                self.delivery[key]=record
                write(self.delivery_path,self.delivery)
            if record.get('asset_id'):
                confirmed += kind=='image'
                continue
            if budget<=0:continue
            budget-=1
            # Comet stores the supplied logical image name without adding .png.
            # Listing the entire experiment grew huge due to the old suffix bug.
            remote_name=record.get('remote_name',name)
            # Run metadata has no requested optimizer step. ExistingExperiment
            # may attach its current step (often 0), which is still this asset.
            matches=[a for a in lookup(remote_name,kind) if
                     (kind=='asset' and step is None) or a.get('step')==step]
            if not matches and kind=='image' and 'remote_name' not in record:
                # Keep historical names, but give new uploads a bounded lookup
                # identity even if the old logical name has many duplicates.
                remote_name=f'{name}/step-{step:06d}'
                matches=[a for a in lookup(remote_name,kind) if a.get('step')==step]
            if matches:
                record.update(asset_id=matches[-1]['assetId'],remote_name=remote_name,confirmed_unix_seconds=time.time())
                if content_hash:record['sha256']=content_hash
                self.delivery[key]=record;write(self.delivery_path,self.delivery)
                confirmed += kind=='image'
                continue
            if time.time()-record.get('queued_unix_seconds',0)<600:continue
            # Save before enqueueing, including ambiguous network failures. On
            # restart, check the server before considering another submission.
            self.delivery[key]={'queued_unix_seconds':time.time(),
                                'remote_name':remote_name,
                                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            write(self.delivery_path,self.delivery)
            if kind=='image':self.experiment.log_image(str(path),name=remote_name,step=step,metadata=metadata)
            else:self.experiment.log_asset(str(path),file_name=remote_name,step=step)
        pending=sum(not self.delivery.get(json.dumps([k,n,s]),{}).get('asset_id') for k,_,n,s,_ in items)
        return confirmed,pending

    def publish(self):
        rows = training_rows(self.folder)
        transition = self.folder/'execution_transition.json'
        if transition.exists() and not self.transition_logged:
            record = json.loads(transition.read_text())
            self.experiment.log_parameters({'continuation/run': self.folder.name,
                'continuation/start_step': record['start_step'],
                'continuation/previous_effective_batch': record['batch_change']['previous'],
                'continuation/effective_batch': record['batch_change']['new']})
            self.transition_logged = True
        # 20k updates exceed Comet's 15k-values-per-metric limit. Full history
        # stays in metrics.jsonl; the default chart interval gives 10,001 points.
        chart_rows = [r for r in rows if r['step'] == 1 or r['step'] % self.metric_every == 0]
        new = [r for r in chart_rows if r['step'] not in self.steps and
               time.time()-self.metric_queued.get(r['step'],0)>300]
        for row in new:
            self.experiment.log_metrics({self.metric_prefix+k:v for k,v in row.items() if k != 'step'}, step=row['step'])
            self.sent_steps.add(row['step'])
            self.metric_queued[row['step']]=time.time()
        images, assets, summaries = completed_outputs(self.folder)
        assets.extend((p, p.name, None) for p in self.folder.glob('uncheckpointed_metrics_*.jsonl'))
        for name in ('execution_transition.json', 'execution_admission.json', 'conditioning_parity.json',
                     'native_checks.json','resume_parity.json','gpu_scoring_admission.json','deployment.json','resolved_config.yaml'):
            path = self.folder/name
            if path.exists():
                assets.append((path, f'{self.folder.name}/{name}', None))
        if (self.folder/'resume_latest.json').exists():
            record = json.loads((self.folder/'resume_latest.json').read_text())
            assets.append((self.folder/'resume_latest.json', f"resume-{record['job_id']}.json", record['checkpoint_step']))
        verified_images,pending_assets = self.deliver(images,assets)
        for step, values in summaries:
            for name in values:
                if name not in self.summary_steps:
                    self.summary_steps[name] = {int(v['step']) for v in self.api.get_metrics(name) if v.get('step') is not None}
            missing = {k:v for k,v in values.items() if step not in self.summary_steps[k] and
                       time.time()-self.summary_queued.get((k,step),0)>300}
            if missing:
                self.experiment.log_metrics(missing, step=step)
                for name in missing:self.summary_queued[name,step]=time.time()
                if self.clean_console:
                    line=f'VALIDATION {step} metrics | '+ ' | '.join(f'{k}={v:.5f}' for k,v in missing.items())
                    print(line,flush=True)
                    self.api.log_output(time.strftime('%Y-%m-%d %H:%M:%S %Z')+' | '+line+'\n',timestamp=time.time())
        self.experiment.flush(timeout=2)
        if chart_rows:
            # SDK batches may arrive out of step order; summary.stepCurrent is
            # the most recently received point, not the highest optimizer step.
            actual = {int(v['step']):float(v['metricValue']) for v in self.api.get_metrics(self.loss_metric)
                      if v['step'] is not None}
            self.steps = {r['step'] for r in chart_rows if r['step'] in actual and
                          math.isclose(actual[r['step']],r['train/loss'],rel_tol=1e-6,abs_tol=1e-8)}
        for name in {n for n,step in self.summary_queued if step not in self.summary_steps[n]}:
            self.summary_steps[name].update(int(v['step']) for v in self.api.get_metrics(name) if v.get('step') is not None)
        pending_summaries=sum(step not in self.summary_steps[name] for step,values in summaries for name in values)
        result = {'experiment_key':self.key, 'verified_unix_seconds':time.time(),
                  'training_step':rows[-1]['step'] if rows else 0, 'verified_images':verified_images,
                  'available_images':len(images),'pending_assets':pending_assets,
                  'pending_metric_steps':len([r for r in chart_rows if r['step'] not in self.steps]),
                  'pending_summary_metrics':pending_summaries,
                  'loss_metric':self.loss_metric, 'metric_every':self.metric_every,
                  'verified_metric_step':max(self.steps,default=0),
                  'new_training_updates':len(new), 'completed_panels':sorted(int(p.name.split('_')[1].split('.')[0])
                      for p in self.folder.glob('decode_*.done.json'))}
        write(self.receipt,result)
        if self.clean_console:
            if new:
                from scripts.clust_flux1a import duration
                last=new[-1]
                print(f"TRAIN {last['step']}/20000 | loss {last['train/loss']:.5f} | training ETA {duration(last.get('train/eta_seconds'))} | Comet confirmed through {result['verified_metric_step']}",flush=True)
            print(f'IMAGES | {verified_images}/{len(images)} confirmed | {pending_assets} assets pending',flush=True)
        else:print(json.dumps(result),flush=True)
        return result


class PublicationWorker:
    """Coalesce sync requests while uploads run independently of Slurm polling."""
    def __init__(self,args,folder):
        self.args,self.folder=args,folder
        self.publisher=None
        self.requested=threading.Event()
        self.stop=threading.Event()
        self.observed=None
        self.result=None
        self.thread=threading.Thread(target=self.run,daemon=True)

    def submit(self,observed):
        self.observed=dict(observed)
        self.requested.set()

    def publish_once(self,observed):
        args=self.args
        sync(observed.get('artifact_run_name',args.run_name),self.folder,args.remote_root)
        record=json.loads((self.folder/'comet_experiment.json').read_text())
        if record['experiment_key']!=args.experiment_key:
            raise ValueError('Remote run belongs to another Comet experiment')
        self.publisher=self.publisher or Publisher(self.folder,args.experiment_key,
            args.metric_prefix,args.metric_every,args.clean_console)
        result=self.publisher.publish()
        result['observation']=observed
        self.result=result
        write(self.folder/'publication_status.json',{'status':'ok',**result})
        return result

    def run(self):
        while not self.stop.is_set():
            if not self.requested.wait(1):continue
            self.requested.clear()
            observed=self.observed
            try:self.publish_once(observed)
            except Exception as error:
                detail=getattr(error,'stderr',None) or str(error)
                if isinstance(detail,bytes):detail=detail.decode(errors='replace')
                write(self.folder/'publication_status.json',{'status':'retry','error':detail[-2000:],
                    'unix_seconds':time.time()})
                print('UPLOAD RETRY | '+detail[-2000:].strip(),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--job-id',required=True)
    parser.add_argument('--experiment-key',required=True)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--metric-prefix',default='',help='Unique prefix for a resumed attempt, retaining interrupted curves')
    parser.add_argument('--metric-every',type=int,default=2)
    parser.add_argument('--remote-root',default='/home/nasilaev/rsrch_new')
    parser.add_argument('--pipeline',action='store_true',help='Poll the staged job chain without a Slurm logging allocation')
    parser.add_argument('--clean-console',action='store_true',help='Publish concise status instead of relay diagnostic JSON')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_name) or not args.job_id.isdigit():
        parser.error('Expected a simple run name and numeric Slurm job ID')
    if args.metric_every < 1 or (args.metric_prefix and not re.fullmatch(r'[A-Za-z0-9_-]+/',args.metric_prefix)):
        parser.error('Expected a positive metric interval and a simple optional prefix ending in /')
    if not re.fullmatch(r'/home/nasilaev/[A-Za-z0-9_-]+',args.remote_root):
        parser.error('Expected a project directory under /home/nasilaev')
    for line in (ROOT/'.env').read_text().splitlines():
        if line.startswith('COMET_API_KEY='):
            os.environ['COMET_API_KEY']=line.split('=',1)[1].strip().strip('"\'')
    os.environ['COMET_DISPLAY_SUMMARY_LEVEL']='0'
    if args.clean_console:os.environ['COMET_LOGGING_CONSOLE']='ERROR'
    folder=ROOT/'runs/clust_comet_mirror'/args.run_name
    folder.mkdir(parents=True,exist_ok=True)
    # The service supervises this one publisher; no changes to the training process.
    import fcntl
    lock=(folder/'publisher.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from comet_ml import API
    from comet_ml.config import get_config
    # Retry in our durable outbox; SDK retry multiplication previously hid a
    # 20-second asset request behind minutes of retries.
    get_config().override['comet.http_session.retry_total']=0
    monitor=API(cache=False).get_experiment_by_key(args.experiment_key)
    worker=PublicationWorker(args,folder)
    if not args.once:worker.thread.start()
    terminal_state=None
    console_sent = None
    console_time = 0.
    from scripts.upload_clust_comet import keep_alive, scheduler_state
    alive, stop = threading.Event(), threading.Event()
    heartbeat = threading.Thread(target=keep_alive,
        args=(args.experiment_key, alive, stop, folder/'heartbeat.json'), daemon=True)
    heartbeat.start()
    try:
        while True:
            try:
                if args.pipeline:observed=pipeline_status(args.run_name,args.remote_root)
                else:
                    state,active=scheduler_state(args.job_id)
                    observed={'job_id':args.job_id,'worker_job_id':args.job_id,'stage':'training',
                        'slurm_state':state,'active':active,'observed_unix_seconds':time.time()}
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                alive.clear()
                detail=getattr(error,'stderr',None) or str(error)
                if isinstance(detail,bytes):detail=detail.decode(errors='replace')
                monitor.log_other('cluster/monitoring_status','unreachable; last worker observation is stale')
                monitor.log_other('cluster/monitoring_error',detail[-2000:])
                print('MONITOR UNREACHABLE | '+detail[-2000:].strip(),flush=True)
                if args.once:raise
                time.sleep(30)
                continue
            if observed['job_id']!=args.job_id:
                raise ValueError('Slurm job identity changed; refusing another run')
            write(folder/'scheduler_observation.json',observed)
            alive.set() if observed['active'] else alive.clear()
            monitor.log_other('cluster/monitoring_status','connected')
            monitor.log_other('cluster/monitoring_error','')
            monitor.log_other('cluster/slurm_state',observed['slurm_state'])
            monitor.log_other('cluster/stage',observed['stage'])
            monitor.log_other('cluster/job_id',observed['worker_job_id'])
            monitor.log_other('cluster/status',json.dumps(observed))
            if observed['active']:monitor.update_status()
            from scripts.clust_flux1a import duration
            progress=observed.get('progress')
            detail=(f" | {progress['stage']} {progress['completed']}/{progress['total']} | stage ETA {duration(progress['eta_seconds'])}"
                    if progress else '')
            line=f"STATUS | {observed['slurm_state']} | job {observed['worker_job_id']} | {observed['stage']}{detail}"
            if worker.result and "training_step" in worker.result:
                r=worker.result
                line+=f" | optimizer {r['training_step']}/20000 | images {r['verified_images']}/{r['available_images']} | uploads pending {r['pending_assets']}"
            print(line,flush=True)
            # Explicit wall-clock timestamp: SDK/API monotonic defaults sort
            # resumed-session console lines before the historical output.
            now=time.time()
            if line!=console_sent or now-console_time>=60:
                try:
                    monitor.log_output(time.strftime('%Y-%m-%d %H:%M:%S %Z')+' | '+line+'\n',timestamp=now)
                    console_sent,console_time=line,now
                except Exception as error:
                    print(f'CONSOLE RETRY | {type(error).__name__}',flush=True)
            if args.once:
                worker.publish_once(observed)
                return
            worker.submit(observed)
            result=worker.result
            # Allow queued artifacts to drain after the final cluster stage.
            if (not observed['active'] and result and not result['observation']['active'] and
                    not result['pending_assets'] and not result['pending_metric_steps'] and not result['pending_summary_metrics']):
                terminal_state='finished' if observed['slurm_state']=='COMPLETED' else 'crashed'
                return
            time.sleep(20)
    finally:
        alive.clear();stop.set();worker.stop.set()
        heartbeat.join(timeout=10)
        if not args.once:worker.thread.join(timeout=5)
        if worker.publisher is not None and not worker.thread.is_alive():
            worker.publisher.experiment.end()
        if terminal_state is not None:monitor.set_state(terminal_state)


if __name__=='__main__':
    main()
