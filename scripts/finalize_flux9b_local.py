"""Download, consolidate Comet, and stop only the explicitly authorized Vast53994096."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shlex
import subprocess
import time

from scripts.vast_gpu import config as credentials, DEFAULT_ENV, env_value, owned_instance, ssh_target, vast

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'runs/flux9b_8k_completion'
REMOTE = '/workspace/rsrch_9b_8k'
TRAIN = 'flux9b_large_qkvo_r128_to8k'
FULL = 'flux9b_8000_fixed96_20261004'
CANONICAL = 'cfbd6f879e3d4158959bbbef7fb8e894'
SOURCE = 'c2c484aa06f1488b92ce1d9795857b90'
CHECKPOINT_FILES = {'adapters.safetensors', 'training_state.pt', 'manifest.json',
                    'resolved_config.yaml', 'resume_config.yaml'}


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


class Transport:
    def __init__(self):
        self.env, key, _ = credentials(argparse.Namespace(env_file=DEFAULT_ENV, private_key=None, public_key=None))
        user, host, port = ssh_target(self.env, 53994096)
        self.host = f'{user}@{host}'
        self.ssh = ['ssh', '-i', str(key), '-p', str(port), '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
                    '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=20', '-o', 'IPQoS=none',
                    '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3']

    def remote(self, command):
        return subprocess.run(self.ssh+[self.host, command], check=True, capture_output=True, text=True,
                              timeout=120).stdout

    def python(self, source):
        return self.remote(shlex.join(['python3', '-c', source]))

    def sync(self, relative, destination, excludes=()):
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync', '-a', '--partial', '--timeout=180', *[f'--exclude={x}' for x in excludes],
                        '-e', shlex.join(self.ssh), f'{self.host}:{REMOTE}/{relative}/', str(destination)+'/'],
                       check=True, timeout=3600)

    def probe(self):
        return json.loads(self.python(f'''import json,hashlib
from pathlib import Path
p=Path({REMOTE!r})/'runs'/{TRAIN!r};f=p.parent/{FULL!r}
cp=p/'checkpoint-008000'
print(json.dumps({{'training':json.loads((p/'status.json').read_text()),
 'full96':json.loads((f/'status.json').read_text()),'ready':(f/'ready.json').exists(),
 'checkpoint_sha256':{{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in cp.iterdir() if x.is_file()}} if cp.exists() else None}}))'''))


def download_checkpoint(t, expected):
    assert set(expected) == CHECKPOINT_FILES, 'Full checkpoint files are required'
    destination = ROOT/'runs'/TRAIN/'checkpoint-008000'
    t.sync(f'runs/{TRAIN}/checkpoint-008000', destination)
    actual = {name:sha(destination/name) for name in expected}
    assert actual == expected, 'Local checkpoint SHA256 differs from remote'
    assert json.loads((destination/'manifest.json').read_text())['step'] == 8000
    receipt = {'verified':True, 'step':8000, 'sha256':actual, 'local_directory':str(destination)}
    write(STATE/'checkpoint_verified.json', receipt)
    text = json.dumps(receipt, indent=2)+'\n'
    t.python(f'from pathlib import Path; p=Path({(REMOTE+"/runs/"+FULL+"/local_checkpoint_verified.json")!r}); '
             f'q=p.with_suffix(".tmp");q.write_text({text!r});q.replace(p)')


def metrics_and_images():
    train, full = ROOT/'runs'/TRAIN, ROOT/'runs'/FULL
    rows = [json.loads(line) for line in (train/'metrics.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['step'] > 2000]
    assert [r['step'] for r in rows] == list(range(2001, 8001)), 'Continuation must contain every update exactly once'
    values = {(name, row['step']):value for row in rows for name,value in row.items() if name != 'step'}
    images = []
    comparison = json.loads((train/'comparison_summary.json').read_text())['metrics']
    for step in (4000, 6000, 8000):
        folder = train/f'validation-{step:06d}'
        report = json.loads((folder/'validation.json').read_text())
        assert len(report['samples']) == 12
        summary = json.loads((folder/'quality_summary.json').read_text())['metrics']
        values.update({('validation/'+key, step):value for key,value in summary.items()})
        values[('validation/best_id_sim',step)] = max(v['id_sim'] for s,v in comparison.items() if int(s) <= step)
        for row in report['samples']:
            images.append((folder/row['image'], 'fixed12/'+row['sample_id'], step))
        for panel in train.glob(f'paired_faces_{step:06d}_*.png'):
            images.append((panel, 'paired_faces/'+panel.stem.rsplit('_',1)[1], step))
    folder = full/'validation-008000'
    report = json.loads((folder/'validation.json').read_text())
    assert len(report['samples']) == 96
    assert report['checkpoint_sha256'] == sha(train/'checkpoint-008000/adapters.safetensors')
    summary = json.loads((folder/'quality_summary.json').read_text())['metrics']
    values.update({('validation96/'+key,8000):value for key,value in summary.items()})
    native = json.loads((full/'native/quality_summary.json').read_text())['metrics']
    values.update({('validation96/native_'+key,8000):value for key,value in native.items()})
    for row in report['samples']:
        images.append((folder/row['image'], 'fixed96_8k/'+row['sample_id'], 8000))
    assert all(math.isfinite(v) for v in values.values()), 'Nonfinite metric in final results'
    return values, images


def consolidate_comet():
    from comet_ml import ExistingExperiment
    from comet_ml.api import API
    os.environ['COMET_API_KEY'] = env_value(DEFAULT_ENV, 'COMET_API_KEY')
    api = API(cache=False)
    destination = api.get_experiment('nikolay-2104', 'rsrch-new', CANONICAL)
    expected, images = metrics_and_images()
    def metric_map():
        return {(r['metricName'],int(r['step'])):float(r['metricValue'])
                for r in destination.get_metrics() if r['step'] is not None}
    existing = metric_map()
    assets = {(re.sub(r'(?: \(\d+\))?\.png$', '', a['fileName']),a.get('step')) for a in destination.get_asset_list() if a.get('type') == 'image'}
    experiment = ExistingExperiment(previous_experiment=CANONICAL, auto_param_logging=False,
        auto_metric_logging=False, log_env_details=False, log_code=False, log_git_metadata=False, log_git_patch=False)
    try:
        experiment.set_name('flux9b_large_qkvo_r128_b1_0_to_8000')
        experiment.log_parameters({'training/steps':8000, 'continuation/source_comet':SOURCE,
            'continuation/start_step':2000, 'continuation/optimizer_resumed':True,
            'training/validation_every':2000, 'continuation/validation_steps':[4000,6000,8000],
            'validation96/step':8000, 'validation96/images':96,
            'validation96/metric_prefix':'validation96/', 'validation/images':12})
        grouped = {}
        for key,value in expected.items():
            if key in existing:
                assert math.isclose(existing[key],value,rel_tol=1e-7,abs_tol=1e-9), f'Conflicting Comet metric: {key}'
            else:
                grouped.setdefault(key[1],{})[key[0]] = value
        for step, values in sorted(grouped.items()):
            experiment.log_metrics(values, step=step)
        for path, name, step in images:
            if (name,step) not in assets:
                experiment.log_image(str(path), name=name, step=step)
        for run in (TRAIN,FULL):
            for path in (ROOT/'runs'/run).rglob('*.json'):
                if 'source_snapshot' not in path.parts and 'checkpoint-' not in str(path):
                    experiment.log_asset(str(path), file_name=f'{run}/{path.relative_to(ROOT/"runs"/run)}')
        experiment.log_asset(str(ROOT/'runs'/TRAIN/'metrics.jsonl'), file_name='continuation_metrics.jsonl')
        experiment.log_asset(str(STATE/'checkpoint_verified.json'))
    finally:
        experiment.end()
    # Read back server state; no successful upload assumption based on end() alone.
    for attempt in range(8):
        actual = metric_map()
        assets = {(re.sub(r'(?: \(\d+\))?\.png$', '', a['fileName']),a.get('step')) for a in destination.get_asset_list() if a.get('type') == 'image'}
        valid = all(k in actual and math.isclose(actual[k],v,rel_tol=1e-7,abs_tol=1e-9) for k,v in expected.items())
        if valid and all((name,step) in assets for _,name,step in images):
            receipt = {'verified':True, 'canonical_comet':CANONICAL, 'source_comet':SOURCE,
                       'metric_points_verified':len(expected), 'image_assets_verified':len(images)}
            write(STATE/'comet_verified.json', receipt)
            return
        time.sleep(30)
    raise RuntimeError('Comet read-back incomplete; machine must remain running')


def stop_gate(policy, checkpoint, results, comet):
    return (policy.get('instance_id') == 53994096 and policy.get('stop_authorized') is True
            and checkpoint.get('verified') is True and checkpoint.get('step') == 8000
            and set(checkpoint.get('sha256',{})) == CHECKPOINT_FILES
            and results.get('verified') is True and results.get('samples') == 96
            and results.get('checkpoint_sha256') == checkpoint.get('sha256')
            and comet.get('verified') is True and comet.get('canonical_comet') == CANONICAL)


def main(check_only=False):
    STATE.mkdir(exist_ok=True)
    policy = json.loads((STATE/'policy.json').read_text())
    assert policy['instance_id'] == 53994096
    lock = (STATE/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    transport = Transport()
    if check_only:
        print(json.dumps(transport.probe(),indent=2))
        assert not stop_gate(policy, {}, {}, {})
        return
    while True:
        try:
            if (STATE/'stopped.json').exists():
                return
            if (STATE/'stop_requested.json').exists():
                instance = owned_instance(transport.env,53994096)
                if instance.get('actual_status') in ('exited','stopped'):
                    write(STATE/'stopped.json',{'instance_id':53994096,'actual_status':instance['actual_status'],
                          'time_utc':datetime.now(timezone.utc).isoformat()})
                    return
                time.sleep(30)
                continue
            state = transport.probe()
            stage = ('finishing' if state['ready'] else 'waiting_for_full96'
                     if state['training'].get('status') == 'completed' else 'waiting_for_training')
            write(STATE/'status.json', {'stage':stage, **state})
            if state['training'].get('status') == 'completed':
                assert state['training']['training_steps'] == state['training']['last_completed_validation'] == 8000
                if not (STATE/'checkpoint_verified.json').exists():
                    download_checkpoint(transport,state['checkpoint_sha256'])
                else:
                    # Re-send a missing remote acknowledgement safely after a transport failure.
                    receipt = json.loads((STATE/'checkpoint_verified.json').read_text())
                    assert receipt['sha256'] == state['checkpoint_sha256']
                    path = REMOTE+'/runs/'+FULL+'/local_checkpoint_verified.json'
                    transport.python(f'from pathlib import Path; Path({path!r}).write_text({json.dumps(receipt)!r})')
            if state['ready']:
                if not (STATE/'results_verified.json').exists():
                    for run in (TRAIN,FULL):
                        transport.sync('runs/'+run, ROOT/'runs'/run, excludes=('checkpoint-*','*.log'))
                    ready = json.loads((ROOT/'runs'/FULL/'ready.json').read_text())
                    for path,expected in ready['files'].items():
                        if not path.endswith('.log'):
                            assert sha(ROOT/'runs'/FULL/path) == expected, path
                    assert ready['samples'] == 96 and ready['step'] == 8000
                    audit = json.loads((ROOT/'runs'/FULL/'background_audit_8000.json').read_text())
                    assert len(audit) == 96 and all(row['background_max_abs'] == 0 for row in audit)
                    assert all(math.isfinite(row['face_mean_abs']) for row in audit)
                    write(STATE/'results_verified.json', {'verified':True,'samples':96,
                          'checkpoint_sha256':ready['checkpoint_sha256']})
                if not (STATE/'comet_verified.json').exists():
                    consolidate_comet()
                receipts = [json.loads((STATE/name).read_text()) for name in
                            ('checkpoint_verified.json','results_verified.json','comet_verified.json')]
                assert stop_gate(policy,*receipts), 'Automatic stop gate failed'
                for name,expected in receipts[0]['sha256'].items():
                    assert sha(ROOT/'runs'/TRAIN/'checkpoint-008000'/name) == expected
                not_before = datetime.fromisoformat(policy['stop_not_before_utc']).timestamp()
                if time.time() >= not_before:
                    # A final shared-lock and GPU check prevents interrupting another job.
                    command = 'flock -n '+shlex.quote(REMOTE+'/runs/face_flow_gpu.lock')+' nvidia-smi --query-compute-apps=pid --format=csv,noheader'
                    assert not transport.remote(command).strip(), 'GPU still has active processes'
                    owned_instance(transport.env,53994096)
                    result = vast(transport.env,'stop','instance','53994096',raw=False)
                    assert result.startswith('stopping instance'), result
                    write(STATE/'stop_requested.json',{'time_utc':datetime.now(timezone.utc).isoformat(),'instance_id':53994096})
                    for _ in range(30):
                        instance = owned_instance(transport.env,53994096)
                        if instance.get('actual_status') in ('exited','stopped'):
                            write(STATE/'stopped.json',{'instance_id':53994096,'actual_status':instance['actual_status'],
                                  'time_utc':datetime.now(timezone.utc).isoformat()})
                            print('Verified checkpoint/results/Comet; Vast53994096 is stopped.',flush=True)
                            return
                        time.sleep(10)
                    raise RuntimeError('Stop requested but not yet confirmed')
        except Exception as error:
            write(STATE/'last_error.json',{'type':type(error).__name__,'message':str(error),
                  'time_utc':datetime.now(timezone.utc).isoformat()})
            print(f'Finalization retry: {type(error).__name__}: {error}',flush=True)
        time.sleep(60)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    main(parser.parse_args().check_only)
