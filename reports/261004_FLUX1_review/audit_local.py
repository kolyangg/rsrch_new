"""Reproduce the small CPU-only reference-prototype and checkpoint audits.

envs/metrics/bin/python reports/261004_FLUX1_review/audit_local.py references
envs/flux-toolkit/bin/python reports/261004_FLUX1_review/audit_local.py checkpoints
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'evidence'
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2)+'\n')
    print(name, 'saved')


def references():
    import torch
    from PIL import Image
    from ba_dit.data.manifest import read_manifest
    from ba_dit.metrics import LegacyFaces, cosine
    rows = read_manifest(ROOT / 'data/validation/manual_val_96.jsonl')
    refs = {r['identity_id']: r for r in rows}
    current = torch.load(ROOT / 'data/validation/id_embeds_manual_val_subject_v2.pth', map_location='cpu', weights_only=True)
    legacy = torch.load(ROOT / 'data/validation/id_embeds_manual_val.pth', map_location='cpu', weights_only=True)
    detector = LegacyFaces()
    results = {}
    for identity, row in refs.items():
        faces = detector(Image.open(row['reference']))
        _, embedding = max(faces, key=lambda x: (x[0][2]-x[0][0])*(x[0][3]-x[0][1]))
        matches = sorted([(k, cosine(embedding, v)) for k, v in current.items()], key=lambda x: -x[1])
        results[identity] = {'reference_face_count': len(faces), 'reference_box': row['reference_box'],
            'subject_v2_cosine': cosine(embedding, current[identity]), 'legacy_cosine': cosine(embedding, legacy[identity]),
            'top_prototype': matches[0][0], 'top_cosine': matches[0][1]}
    save('reference_metric_audit.json', results)


def checkpoints():
    import torch
    from safetensors.torch import load_file
    old = ROOT / 'runs/flux9b_large_qkvo_r128_fast5h_b1/checkpoint-002000'
    new = ROOT / 'runs/flux9b_large_qkvo_r128_to8k/checkpoint-008000'
    a, b = load_file(old / 'adapters.safetensors'), load_file(new / 'adapters.safetensors')
    norms = {}
    for component in 'qkvo':
        norms[component] = {}
        for label, params in [('2000', a), ('8000', b)]:
            squares = []
            for key, value in params.items():
                if f'.{component}_delta.a' not in key:
                    continue
                left, right = value.double(), params[key[:-1]+'b'].double()
                # ||BA||_F^2 without materializing a width-by-width update.
                squares.append(float(((right.T @ right) * (left @ left.T)).sum()))
            norms[component][label] = sum(squares)**.5
    # This is the user's own trusted, previously verified local training state.
    state = torch.load(new / 'training_state.pt', map_location='cpu', weights_only=False)
    adam = state['optimizer']['state']
    steps = sorted({int(v['step']) for v in adam.values()})
    finite = all(bool(torch.isfinite(t).all()) for v in adam.values() for t in v.values() if isinstance(t, torch.Tensor))
    receipt = json.loads((ROOT / 'runs/flux9b_8k_completion/checkpoint_verified.json').read_text())
    hashes_match = all(hashlib.sha256((new/name).read_bytes()).hexdigest() == expected for name,expected in receipt['sha256'].items())
    result = {'delta_frobenius': norms, 'finite_64_tensors': all(bool(torch.isfinite(v).all()) for v in b.values()),
        'all_64_changed': all(not torch.equal(a[k], b[k]) for k in a), 'parameter_count': sum(v.numel() for v in b.values()),
        'optimizer_states': len(adam), 'optimizer_steps': steps, 'finite_optimizer_tensors': finite,
        'cursor': state['cursor'], 'checkpoint_files_match_receipt': hashes_match,
        'rng_fields_present': all(k in state for k in ('torch_rng', 'cuda_rng', 'python_rng'))}
    assert result['finite_64_tensors'] and result['all_64_changed'] and hashes_match
    assert len(adam) == 64 and steps == [8000] and finite and state['cursor'] == 8000
    save('checkpoint_audit.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['references', 'checkpoints'])
    args = parser.parse_args()
    globals()[args.mode]()
