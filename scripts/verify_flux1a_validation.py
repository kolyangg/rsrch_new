"""Read back FLUX1a fixed96 metrics/images from Comet; repair missing uploads only."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/FLUX1a_vast9B_20261004'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(step, repair=False):
    if ROOT != Path('/workspace/rsrch_FLUX1abc'):
        raise RuntimeError('Verifier is scoped to the frozen GB10 FLUX1a deployment')
    if not (RUN/f'summarize_{step}.done.json').exists():
        return {'verified': False, 'status': 'waiting_for_validation', 'step': step}
    import yaml
    config = yaml.safe_load((RUN/'resolved_config.yaml').read_text())
    folder = RUN/f'validation-{step:06d}'
    report = json.loads((folder/'validation.json').read_text())
    summary = json.loads((folder/'quality_summary.json').read_text())
    panel_path = Path(config['data']['validation_manifest'])
    panel = [json.loads(line) for line in panel_path.read_text().splitlines() if line.strip()]
    samples = report['samples']
    keys = [r['sample_id'] for r in panel]
    assert len(keys) == len(set(keys)) == 96
    assert [r['sample_id'] for r in samples] == keys
    assert report['panel_sha256'] == sha(panel_path)
    checkpoint_sha = sha(RUN/f'checkpoint-{step:06d}'/'adapters.safetensors')
    assert report['checkpoint_sha256'] == checkpoint_sha
    for expected, actual in zip(panel, samples):
        assert all(actual[k] == expected[k] for k in ('prompt', 'seed', 'identity_id'))
        assert actual['target_photo_loaded'] is False
        assert (folder/actual['image']).is_file()
        assert (folder/f"{actual['sample_id']}.safetensors").is_file()
    with (folder/'quality_per_image.csv').open() as stream:
        scored = list(csv.DictReader(stream))
    assert [r['sample_id'] for r in scored] == keys
    for key, value in summary['metrics'].items():
        mean = sum(float(r[key]) for r in scored)/96
        assert math.isfinite(value) and math.isclose(mean, value, rel_tol=1e-7, abs_tol=1e-9)
    audit = json.loads((RUN/f'inference_audit_{step}.json').read_text())
    assert audit['samples'] == 96 and audit['full_denoiser'] and audit['exact_latent_exterior']
    assert audit['checkpoint_sha256'] == checkpoint_sha and not audit['target_photos_loaded']
    backgrounds = json.loads((RUN/f'background_audit_{step}.json').read_text())
    assert [r['sample_id'] for r in backgrounds] == keys
    assert all(r['background_max_abs'] == 0 for r in backgrounds)
    for line in (ROOT/'.env').read_text().splitlines():
        if line.startswith('COMET_API_KEY='):
            os.environ.setdefault('COMET_API_KEY', line.split('=', 1)[1].strip().strip('\"\''))
    from comet_ml.api import API
    key = json.loads((RUN/'comet_experiment.json').read_text())['experiment_key']
    assert key == '2018ec7a730243bc98d922178e58aa5c'
    experiment = API(cache=False).get_experiment_by_key(key)
    expected_metrics = {'validation/'+k: v for k, v in summary['metrics'].items()}

    def missing_uploads():
        missing_metrics = {}
        for name, value in expected_metrics.items():
            logged = [float(r['metricValue']) for r in experiment.get_metrics(name)
                      if r.get('step') is not None and int(float(r['step'])) == step]
            assert all(math.isclose(v, value, rel_tol=1e-7, abs_tol=1e-9) for v in logged), name
            if not logged:
                missing_metrics[name] = value
        present = {re.sub(r'(?: \(\d+\))?(?:\.png)?$', '', a['fileName'])
                   for a in experiment.get_asset_list()
                   if a.get('type') == 'image' and a.get('step') is not None
                   and int(float(a['step'])) == step}
        missing_images = [r for r in samples if 'fixed96/'+r['sample_id'] not in present]
        return missing_metrics, missing_images

    missing_metrics, missing_images = missing_uploads()
    if repair:
        for name, value in missing_metrics.items():
            experiment.log_metric(name, value, step=step)
        for row in missing_images:
            experiment.log_image(str(folder/row['image']), image_name='fixed96/'+row['sample_id'],
                                 step=step, metadata={'prompt': row['prompt'], 'seed': row['seed']})
        if missing_metrics or missing_images:
            missing_metrics, missing_images = missing_uploads()
    assert not missing_metrics and not missing_images, 'Comet read-back incomplete; retry after uploads settle'
    receipt = {'verified': True, 'step': step, 'samples': 96, 'comet_experiment_key': key,
               'verified_at_utc': datetime.now(timezone.utc).isoformat(),
               'checkpoint_sha256': checkpoint_sha, 'panel_sha256': report['panel_sha256'],
               'metrics': expected_metrics,
               'image_sha256': {r['sample_id']: sha(folder/r['image']) for r in samples}}
    path = RUN/f'comet_verified_{step:06d}.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt, indent=2)+'\n')
    temporary.replace(path)
    return {k: v for k, v in receipt.items() if k != 'image_sha256'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, choices=(0, 2000, 4000, 6000), required=True)
    parser.add_argument('--repair-missing', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.step, args.repair_missing), indent=2))
