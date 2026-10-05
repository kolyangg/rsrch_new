"""Collect the one-off 2k full96 evaluation and mirror it into the original training run."""
import json
import math
import os
from pathlib import Path
import re
import time

from scripts.finalize_flux9b_local import Transport, ROOT, REMOTE, sha, write
from scripts.vast_gpu import DEFAULT_ENV, env_value

NAME = 'FLUX1_vast9B_2000_fixed96_20261004'
RUN = ROOT / 'runs' / NAME
STATE = ROOT / 'runs/FLUX1_vast9B_2k_collection'
CANONICAL = 'cfbd6f879e3d4158959bbbef7fb8e894'
FULL96 = '2b3eda1b1f364ecfaa584ffe273102a1'


def mirror():
    from comet_ml import ExistingExperiment
    from comet_ml.api import API
    os.environ['COMET_API_KEY'] = env_value(DEFAULT_ENV, 'COMET_API_KEY')
    remote = API(cache=False).get_experiment('nikolay-2104', 'rsrch-new', CANONICAL)
    folder = RUN / 'validation-002000'
    current = json.loads((folder / 'quality_summary.json').read_text())['metrics']
    native = json.loads((RUN / 'native/quality_summary.json').read_text())['metrics']
    expected = {**{'validation96/' + k: v for k, v in current.items()},
                **{'validation96/native_' + k: v for k, v in native.items()}}
    samples = json.loads((folder / 'validation.json').read_text())['samples']

    def values(name):
        return {int(float(r['step'])): float(r['metricValue']) for r in remote.get_metrics(name)
                if r.get('step') is not None}

    def images():
        return {re.sub(r'(?: \(\d+\))?(?:\.png)?$', '', a['fileName']) for a in remote.get_asset_list()
                if a.get('type') == 'image' and a.get('step') is not None
                and int(float(a['step'])) == 2000}

    before = {name: values(name) for name in expected}
    pilot_before = values('validation/id_sim').get(2000)
    names = {'fixed96_2k/' + r['sample_id'] for r in samples}
    present = images()
    exp = ExistingExperiment(previous_experiment=CANONICAL, auto_param_logging=False,
        auto_metric_logging=False, log_env_details=False, log_code=False,
        log_git_metadata=False, log_git_patch=False)
    try:
        missing = {}
        for name, value in expected.items():
            if 2000 in before[name]:
                assert math.isclose(before[name][2000], value, rel_tol=1e-7, abs_tol=1e-9), name
            else:
                missing[name] = value
        if missing:
            exp.log_metrics(missing, step=2000)
        for row in samples:
            name = 'fixed96_2k/' + row['sample_id']
            if name not in present:
                exp.log_image(str(folder / row['image']), name=name, step=2000)
        for relative in ('validation_request.json', 'comet_verified.json',
                         'validation-002000/quality_summary.json', 'validation-002000/quality_per_image.csv'):
            exp.log_asset(str(RUN / relative), file_name='fixed96_2k/' + relative, step=2000)
    finally:
        exp.end()
    for _ in range(8):
        after = {name: values(name) for name in expected}
        assert values('validation/id_sim').get(2000) == pilot_before, 'Pilot12 score changed'
        for name, prior in before.items():
            if 8000 in prior:
                assert after[name].get(8000) == prior[8000], f'8k metric changed: {name}'
        if all(2000 in after[name] and math.isclose(after[name][2000], value, rel_tol=1e-7, abs_tol=1e-9)
               for name, value in expected.items()) and names <= images():
            write(STATE / 'comet_verified.json', {'verified': True, 'step': 2000, 'samples': 96,
                  'dedicated_full96_comet': FULL96, 'canonical_comet': CANONICAL,
                  'canonical_metric_prefix': 'validation96/', 'pilot12_preserved': True,
                  'previous_8k_metrics_preserved': True, 'metrics': expected})
            return
        time.sleep(30)
    raise RuntimeError('Canonical Comet read-back incomplete')


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'comet_verified.json').exists():
        return
    t = Transport()
    deadline = time.monotonic() + 4 * 3600
    while time.monotonic() < deadline:
        try:
            status = json.loads(t.python('from pathlib import Path; print(Path(' +
                repr(REMOTE + '/runs/' + NAME + '/status.json') + ').read_text())'))
        except Exception as error:
            write(STATE / 'status.json', {'status': 'connection_retry', 'error': type(error).__name__})
            time.sleep(60)
            continue
        write(STATE / 'status.json', {'status': 'waiting_for_validation', 'remote': status})
        if status['status'] == 'failed':
            raise RuntimeError('Remote validation failed: ' + status.get('error', 'unknown'))
        if status['status'] == 'completed':
            break
        time.sleep(60)
    else:
        raise TimeoutError('Validation did not finish within four hours')
    t.sync('runs/' + NAME, RUN, excludes=('checkpoint-*', '*.log'))
    ready = json.loads((RUN / 'ready.json').read_text())
    assert ready['step'] == 2000 and ready['samples'] == 96 and ready['comet_experiment_key'] == FULL96
    for path, expected in ready['files'].items():
        assert sha(RUN / path) == expected, path
    assert json.loads((RUN / 'comet_verified.json').read_text())['verified']
    write(STATE / 'download_verified.json', {'verified': True, 'step': 2000, 'samples': 96,
          'files_verified': len(ready['files']), 'checkpoint_sha256': ready['checkpoint_sha256']})
    write(STATE / 'status.json', {'status': 'mirroring_comet', 'step': 2000, 'samples': 96})
    mirror()
    write(STATE / 'status.json', {'status': 'completed', 'step': 2000, 'samples': 96})
    print('2k full96 downloaded, hashes verified, and logged to both existing Comet runs.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        STATE.mkdir(exist_ok=True)
        write(STATE / 'status.json', {'status': 'failed', 'error': str(error)})
        raise
