"""CPU identity preparation using the cluster's already verified dataset snapshot."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ba_dit.config import ROOT, load_config
from ba_dit.data.manifest import file_hash, assert_disjoint


def prepared_rows(config, source):
    record = json.loads((source/'identity.json').read_text())
    transition = json.loads((source/'execution_transition.json').read_text())
    path = source/'prepared_manifest.json'
    assert file_hash(path) == transition['prepared_manifest_sha256']
    for split in ('train', 'validation'):
        assert file_hash(config['data'][split+'_manifest']) == record[split+'_manifest_sha256']
    rows = json.loads(path.read_text())
    assert_disjoint(rows['train'], rows['validation'])
    return rows, file_hash(path)


def install_reader(config, rows):
    def read(path, training=False, limit=None):
        split = next((s for s in rows if Path(path).resolve() == Path(config['data'][s+'_manifest']).resolve()), None)
        if split is None:
            raise ValueError('Unverified dataset manifest')
        return rows[split][:limit] if limit else rows[split]
    return read


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--child', action='store_true')
    a = p.parse_args()
    config_path = ROOT/'configs/clust/FLUX1a_cluster_4b.yaml'
    config = load_config(config_path)
    setup = ROOT/'runs/FLUX1a_cluster_4b_20261004_setup'
    setup.mkdir(parents=True, exist_ok=True)
    if not a.child:
        from ba_dit.progress import stage_progress
        os.environ.update(BA_STAGE_PROGRESS=str(setup/'stage_progress.json'),BA_STAGE_ACTION='prepare',BA_STAGE_STEP='0')
        total=config['training']['steps']*config['training']['world_size']
        started=time.monotonic()
        log=setup/f'prepare_{os.getenv("SLURM_JOB_ID","local")}.diagnostic.log'
        with log.open('a') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','scripts.clust_flux1a_prepare','--source',str(a.source),
                '--workers',str(a.workers),'--child'],stdout=stream,stderr=subprocess.STDOUT)
            while child.poll() is None:
                items=Path(config['data']['identity_supervision'])/'items'
                count=sum(1 for _ in items.glob('*.json'))
                stage_progress('prepare/identity_labels',count,total,started)
                eta=(time.monotonic()-started)*(total-count)/count if count else None
                print(f'PREPARE identity labels {count}/{total} | ETA {eta/60:.1f} min' if eta is not None else
                      f'PREPARE identity labels {count}/{total} | loading | ETA estimating',flush=True)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
        if child.returncode:raise RuntimeError(f'Preparation failed; see {log}')
        print('PREPARE complete | identity labels verified',flush=True)
        return
    rows, sha = prepared_rows(config, a.source)
    (setup/'prepared_manifest.json').write_text(json.dumps(rows)+'\n')
    (setup/'dataset_verification.json').write_text(json.dumps({'source':str(a.source),
        'source_prepared_sha256':sha, 'prepared_sha256':file_hash(setup/'prepared_manifest.json'),
        'policy':'Reuse CPU-verified immutable rows; rehash each consumed image during GPU caching'}, indent=2)+'\n')
    from scripts import prepare_flux1_identity
    prepare_flux1_identity.read_manifest = install_reader(config, rows)
    prepare_flux1_identity.main(argparse.Namespace(config=config_path, output=None, limit=None,
        scheduled_only=True, workers=a.workers, threads=1))


if __name__ == '__main__':
    main()
