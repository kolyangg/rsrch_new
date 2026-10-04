"""Relay one cluster run's completed outputs and live metrics through this workstation."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def sync(run_name, destination):
    includes = ['metrics.jsonl', 'status.json', 'comet-live-status.json', 'comet_experiment.json',
                'resolved_config.yaml', 'resume_latest.json', 'uncheckpointed_metrics_*.jsonl',
                '*.done.json', 'paired_faces_*.png', 'mask_overlays.png']
    command = ['redshield-vpn', 'exec', 'russia', '--', 'rsync', '-rt', '--partial',
               '--delay-updates', '--timeout=90', *[f'--include=/{p}' for p in includes],
               '--include=/validation-*/', '--exclude=*_raw.png', '--include=/validation-*/*.png',
               '--include=/validation-*/*.json', '--include=/validation-*/*.csv', '--exclude=*',
               f'clust:/home/nasilaev/rsrch_new/runs/{run_name}/', str(destination)+'/']
    subprocess.run(command, check=True, timeout=900)


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
    images, assets, summaries = [], [], []
    if (folder/'mask_overlays.png').exists():
        images.append((folder/'mask_overlays.png', 'native_face_masks', 0, {}))
    for path in sorted(folder.glob('validation-*')):
        step = int(path.name.split('-')[1])
        if not (folder/f'decode_{step}.done.json').exists():
            continue
        report = json.loads((path/'validation.json').read_text())
        if len(report['samples']) != 96:
            raise ValueError('Cluster run requires its original fixed96 panel')
        for row in report['samples']:
            image_path = path/row['image']
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            images.append((image_path, 'fixed96/'+row['sample_id'], step,
                           {'prompt':row['prompt'], 'seed':row['seed']}))
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
    def __init__(self, folder, key, metric_prefix='', metric_every=2):
        from comet_ml import API, ExistingExperiment
        self.folder, self.key = folder, key
        self.metric_prefix, self.metric_every = metric_prefix, metric_every
        self.loss_metric = metric_prefix+'train/loss'
        self.api = API(cache=False).get_experiment_by_key(key)
        if self.api is None:
            raise RuntimeError('The existing Comet experiment was not found')
        self.experiment = ExistingExperiment(previous_experiment=key, auto_param_logging=False,
            auto_metric_logging=False, log_env_details=False, log_code=False, log_git_metadata=False,
            log_git_patch=False, log_graph=False)
        self.experiment.log_other('cluster/artifact_transport', 'workstation rsync relay')
        self.experiment.log_other('cluster/active_training_metric', self.loss_metric)
        self.experiment.log_other('cluster/metric_every', metric_every)
        self.steps = {int(r['step']) for r in self.api.get_metrics(self.loss_metric) if r['step'] is not None}
        self.receipt = folder/'publisher_verified.json'

    def publish(self):
        rows = training_rows(self.folder)
        # 20k updates exceed Comet's 15k-values-per-metric limit. Full history
        # stays in metrics.jsonl; the default chart interval gives 10,001 points.
        chart_rows = [r for r in rows if r['step'] == 1 or r['step'] % self.metric_every == 0]
        existing_assets = self.api.get_asset_list()
        image_keys = {(a['fileName'], a.get('step')) for a in existing_assets if a.get('type') == 'image'}
        asset_names = {a['fileName'] for a in existing_assets}
        current = {(v['name'],v.get('stepCurrent')) for v in self.api.get_metrics_summary()}
        new = [r for r in chart_rows if r['step'] not in self.steps]
        for row in new:
            self.experiment.log_metrics({self.metric_prefix+k:v for k,v in row.items() if k != 'step'}, step=row['step'])
        images, assets, summaries = completed_outputs(self.folder)
        assets.extend((p, p.name, None) for p in self.folder.glob('uncheckpointed_metrics_*.jsonl'))
        if (self.folder/'resume_latest.json').exists():
            record = json.loads((self.folder/'resume_latest.json').read_text())
            assets.append((self.folder/'resume_latest.json', f"resume-{record['job_id']}.json", record['checkpoint_step']))
        wanted_images = {(name+'.png',step) for _,name,step,_ in images}
        for path,name,step,metadata in images:
            if (name+'.png',step) not in image_keys:
                self.experiment.log_image(str(path), name=name, step=step, metadata=metadata)
        for path,name,step in assets:
            if name not in asset_names:
                self.experiment.log_asset(str(path), file_name=name, step=step)
        for step, values in summaries:
            missing = {k:v for k,v in values.items() if (k,step) not in current}
            if missing:
                self.experiment.log_metrics(missing, step=step)
        if not self.experiment.flush(timeout=180):
            raise RuntimeError('Comet flush did not finish; files remain local for retry')
        actual_images = {(a['fileName'],a.get('step')) for a in self.api.get_asset_list() if a.get('type') == 'image'}
        if not wanted_images <= actual_images:
            raise RuntimeError(f'Comet has not confirmed {len(wanted_images-actual_images)} images yet')
        if chart_rows:
            # SDK batches may arrive out of step order; summary.stepCurrent is
            # the most recently received point, not the highest optimizer step.
            for attempt in range(6):
                actual = {int(v['step']):float(v['metricValue']) for v in self.api.get_metrics(self.loss_metric)
                          if v['step'] is not None}
                if all(r['step'] in actual and math.isclose(actual[r['step']], r['train/loss'],
                           rel_tol=1e-6, abs_tol=1e-8) for r in chart_rows):
                    break
                if attempt == 5:
                    raise RuntimeError('Comet training history has not caught up')
                time.sleep(5)
        self.steps.update(r['step'] for r in new)
        result = {'experiment_key':self.key, 'verified_unix_seconds':time.time(),
                  'training_step':rows[-1]['step'] if rows else 0, 'verified_images':len(wanted_images),
                  'loss_metric':self.loss_metric, 'metric_every':self.metric_every,
                  'verified_metric_step':chart_rows[-1]['step'] if chart_rows else 0,
                  'new_training_updates':len(new), 'completed_panels':sorted({s for _,_,s,_ in images})}
        write(self.receipt,result)
        print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--job-id',required=True)
    parser.add_argument('--experiment-key',required=True)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--metric-prefix',default='',help='Unique prefix for a resumed attempt, retaining interrupted curves')
    parser.add_argument('--metric-every',type=int,default=2)
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_name) or not args.job_id.isdigit():
        parser.error('Expected a simple run name and numeric Slurm job ID')
    if args.metric_every < 1 or (args.metric_prefix and not re.fullmatch(r'[A-Za-z0-9_-]+/',args.metric_prefix)):
        parser.error('Expected a positive metric interval and a simple optional prefix ending in /')
    for line in (ROOT/'.env').read_text().splitlines():
        if line.startswith('COMET_API_KEY='):
            os.environ['COMET_API_KEY']=line.split('=',1)[1].strip().strip('"\'')
    os.environ['COMET_DISPLAY_SUMMARY_LEVEL']='0'
    folder=ROOT/'runs/clust_comet_mirror'/args.run_name
    folder.mkdir(parents=True,exist_ok=True)
    # The service supervises this one publisher; no changes to the training process.
    import fcntl
    lock=(folder/'publisher.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    publisher=None
    terminal_state=None
    try:
        while True:
            sync(args.run_name,folder)
            record=json.loads((folder/'comet_experiment.json').read_text())
            if record['experiment_key'] != args.experiment_key:
                raise ValueError('Remote run belongs to another Comet experiment')
            observed_path=folder/'comet-live-status.json'
            observed=json.loads(observed_path.read_text()) if observed_path.exists() else None
            if observed and observed['job_id'] != args.job_id:
                raise ValueError('Slurm job identity changed; stop rather than publishing a different run')
            publisher=publisher or Publisher(folder,args.experiment_key,args.metric_prefix,args.metric_every)
            publisher.publish()
            if args.once or (observed and not observed['active']):
                if observed and not observed['active']:
                    terminal_state='finished' if observed['slurm_state']=='COMPLETED' else 'crashed'
                return
            time.sleep(60)
    finally:
        if publisher is not None:
            publisher.experiment.end()
            if terminal_state is not None:
                publisher.api.set_state(terminal_state)


if __name__=='__main__':
    main()
