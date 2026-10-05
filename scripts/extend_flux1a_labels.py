"""Extend derived identity labels while retaining every existing target/embedding."""
import copy
import hashlib
import json
from pathlib import Path
import shutil


def directory(config):
    original = Path(config['data']['identity_supervision'])
    return original.with_name(original.name+'_to6000')


def seed(config):
    original = Path(config['data']['identity_supervision'])
    destination = directory(config)
    (destination/'items').mkdir(parents=True, exist_ok=True)
    for source in (original/'items').glob('*.json'):
        target = destination/'items'/source.name
        if not target.exists():
            shutil.copyfile(source, target)
    parity = destination/'arcface_parity.npz'
    if not parity.exists():
        shutil.copyfile(original/'arcface_parity.npz', parity)
    return destination


def verify(run, original_config):
    import numpy as np
    from ba_dit.nn.online_identity_loss import supervision_identity
    original = Path(original_config['data']['identity_supervision'])
    destination = directory(original_config)
    config = copy.deepcopy(original_config)
    config['training']['steps'] = 6000
    config['data']['identity_supervision'] = str(destination)
    old_identity = supervision_identity(original_config)
    new_identity = supervision_identity(config)
    old = {r['sample_id']: r for r in map(json.loads, (original/'records.jsonl').read_text().splitlines())}
    new = {r['sample_id']: r for r in map(json.loads, (destination/'records.jsonl').read_text().splitlines())}
    old_embeddings = np.load(original/'target_embeddings.npy', mmap_mode='r')
    new_embeddings = np.load(destination/'target_embeddings.npy', mmap_mode='r')
    for key, row in old.items():
        candidate = new[key]
        assert {k:v for k,v in row.items() if k != 'embedding_index'} == {
            k:v for k,v in candidate.items() if k != 'embedding_index'}, key
        if row['accepted']:
            assert np.array_equal(old_embeddings[row['embedding_index']],
                                  new_embeddings[candidate['embedding_index']]), key
    receipt = {'verified': True, 'original_directory': str(original), 'extended_directory': str(destination),
               'original_identity': old_identity, 'extended_identity': new_identity,
               'original_rows_exact': len(old), 'extended_rows': len(new),
               'trajectory_updates': 6000, 'training_order_and_targets_unchanged': True}
    temporary = run/'identity_extension_6000.tmp'
    temporary.write_text(json.dumps(receipt, indent=2)+'\n')
    temporary.replace(run/'identity_extension_6000.json')
    return receipt


def apply(config, run):
    receipt = json.loads((run/'identity_extension_6000.json').read_text())
    destination = directory(config)
    assert receipt['verified'] and receipt['original_directory'] == config['data']['identity_supervision']
    assert receipt['extended_directory'] == str(destination)
    assert hashlib.sha256((destination/'manifest.json').read_bytes()).hexdigest() == receipt['extended_identity']['manifest_sha256']
    config['data']['identity_supervision'] = str(destination)
    return config


def verify_resume(checkpoint, config, original_digest, strict_verify):
    """The sole extra change is the verified superset of immutable derived labels."""
    from ba_dit.config import load_config
    saved = load_config(checkpoint/'resume_config.yaml')
    expected = directory(saved)
    assert Path(config['data']['identity_supervision']) == expected
    canonical = copy.deepcopy(config)
    canonical['data']['identity_supervision'] = saved['data']['identity_supervision']
    strict_verify(checkpoint, canonical, original_digest)
