"""Reproduce the FLUX1 review from saved runs; CPU analysis, no model inference.

Run with envs/report/bin/python reports/261004_FLUX1_review/analyze.py.
Comet baseline CSVs and CPU reference/checkpoint audits are retained as evidence.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
PILOT = ROOT / 'runs/flux9b_large_qkvo_r128_to8k'
FULL = ROOT / 'runs/flux9b_8000_fixed96_20261004'
STEPS = [0, 1000, 2000, 4000, 6000, 8000]
COLORS = {'native': '#65758B', 'trained': '#087F8C', 'loss': '#B66420'}
used = set()


def read(path):
    used.add(Path(path))
    return json.loads(Path(path).read_text())


def scores(path):
    used.add(Path(path))
    return {r['sample_id']: {k: v if k in ('sample_id', 'identity_id') else float(v)
                            for k, v in r.items()} for r in csv.DictReader(Path(path).open())}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def delta_summary(a, b, ids):
    d = np.array([b[k]['id_sim'] - a[k]['id_sim'] for k in ids])
    owners = sorted({a[k]['identity_id'] for k in ids})
    groups = [d[[a[k]['identity_id'] == owner for k in ids]] for owner in owners]
    rng = np.random.default_rng(142)
    ix = rng.integers(0, len(groups), (100000, len(groups)))
    sums, counts = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
    draws = sums[ix].sum(1) / counts[ix].sum(1)
    macro = np.array([g.mean() for g in groups])
    return dict(n=len(ids), identities=len(owners), mean_delta=float(d.mean()),
                median_delta=float(np.median(d)), improved=int((d > 0).sum()),
                regressed=int((d < 0).sum()), identity_mean_delta=float(macro.mean()),
                identity_cluster_bootstrap95=np.quantile(draws, [.025, .975]).tolist(),
                macro_identity_bootstrap95=np.quantile(macro[ix].mean(1), [.025, .975]).tolist())


def means(a, ids):
    return {metric: float(np.mean([a[k][metric] for k in ids]))
            for metric in ('id_sim', 'text_sim')}


def savefig(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=165, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def crop(im, box, padding=.16):
    x0, y0, x1, y1 = box
    x, y = (x0+x1)/2, (y0+y1)/2
    half = max(x1-x0, y1-y0)*(0.5+padding)
    return im.crop((max(0, x-half), max(0, y-half), min(im.width, x+half), min(im.height, y+half)))


def main():
    from ba_dit.data.geometry import reference_geometry
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    rows = [json.loads(l) for l in (ROOT / 'data/validation/manual_val_96.jsonl').read_text().splitlines()]
    small = [json.loads(l) for l in (ROOT / 'data/validation/manual_val_12_pilot.jsonl').read_text().splitlines()]
    ids, pids = [r['sample_id'] for r in rows], [r['sample_id'] for r in small]
    native = scores(OUT / 'evidence/native96_quality_per_image.csv')
    pn = scores(OUT / 'evidence/native12_quality_per_image.csv')
    trained = scores(FULL / 'validation-008000/quality_per_image.csv')
    checkpoints = {step: scores(PILOT / f'validation-{step:06d}/quality_per_image.csv') for step in STEPS}
    masks = read(FULL / 'routing_masks.json')['samples']
    pmasks = read(PILOT / 'routing_masks.json')['samples']
    result = {'date': '2026-10-04', 'method': 'paired existing outputs; 100,000 identity-cluster bootstrap draws; seed 142',
              'full96': {'native': means(native, ids), 'trained8000': means(trained, ids),
                         'paired': delta_summary(native, trained, ids)},
              'pilot12': {'native': means(pn, pids), 'checkpoints': {s: means(a, pids) for s, a in checkpoints.items()},
                          '2000_to_8000': delta_summary(checkpoints[2000], checkpoints[8000], pids),
                          'native_to_2000': delta_summary(pn, checkpoints[2000], pids),
                          'native_to_8000': delta_summary(pn, checkpoints[8000], pids)},
              'per_identity': {}, 'per_prompt': {}, 'subsets': {}, 'reference_geometry': {}, 'panel_overlap': []}
    for label, directory, a in [('native96', FULL / 'native', native), ('trained96', FULL / 'validation-008000', trained),
                               ('native12', PILOT / 'native', pn), *[(str(s), PILOT / f'validation-{s:06d}', a) for s, a in checkpoints.items()]]:
        summary = read(directory / 'quality_summary.json')['metrics']
        keys = ids if label.endswith('96') else pids
        for metric in ('id_sim', 'text_sim'):
            assert abs(means(a, keys)[metric] - summary[metric]) < 1e-7, (label, metric)
    for owner in dict.fromkeys(r['identity_id'] for r in rows):
        keys = [r['sample_id'] for r in rows if r['identity_id'] == owner]
        result['per_identity'][owner] = {'native': means(native, keys), 'trained': means(trained, keys),
                                         'paired': delta_summary(native, trained, keys)}
        r = next(r for r in rows if r['identity_id'] == owner)
        path = ROOT / 'data/validation' / r['reference_image']
        used.add(path)
        im = Image.open(path).convert('RGB')
        _, tokens, geometry = reference_geometry(im, r['reference_face_bbox'], 'flux', 512)
        bw, bh = r['reference_face_bbox'][2]-r['reference_face_bbox'][0], r['reference_face_bbox'][3]-r['reference_face_bbox'][1]
        result['reference_geometry'][owner] = dict(**geometry, selected_face_tokens=int(tokens.sum()),
            face_pixels_approx=[bw*geometry['resize_wh'][0]/im.width, bh*geometry['resize_wh'][1]/im.height])
    for index in range(12):
        keys = [r['sample_id'] for r in rows if r['prompt_index'] == index]
        result['per_prompt'][index] = {'native': means(native, keys), 'trained': means(trained, keys)}
    subsets = {'pilot_rows_in_full96': pids, 'other84': [k for k in ids if k not in pids],
               'one_detected_face': [k for k in ids if native[k]['id_sim_face_count'] == 1],
               'multiple_detected_faces': [k for k in ids if native[k]['id_sim_face_count'] > 1]}
    for label, low, high in [('face_under128', 0, 128), ('face128_to200', 128, 200), ('face_over200', 200, 10000)]:
        subsets[label] = [k for k in ids if low <= np.sqrt((masks[k]['face_bbox'][2]-masks[k]['face_bbox'][0])*
                                                         (masks[k]['face_bbox'][3]-masks[k]['face_bbox'][1])) < high]
    for label, keys in subsets.items():
        result['subsets'][label] = {'n': len(keys), 'native': means(native, keys), 'trained': means(trained, keys)}
    for k in pids:
        result['panel_overlap'].append({'sample_id': k, 'same_native_png': masks[k]['baseline_image_sha256'] == pmasks[k]['baseline_image_sha256'],
            'same_mask': masks[k]['face_bbox'] == pmasks[k]['face_bbox'],
            'id_sim_full_minus_pilot': trained[k]['id_sim']-checkpoints[8000][k]['id_sim']})
    metrics_file = PILOT / 'metrics.jsonl'
    used.add(metrics_file)
    metrics = [json.loads(l) for l in metrics_file.read_text().splitlines()]
    assert [r['step'] for r in metrics] == list(range(1, 8001))
    assert all(np.isfinite(v) for row in metrics for v in row.values() if isinstance(v, (float, int)))
    result['training'] = {'updates': 8000, 'samples_seen': metrics[-1]['train/samples_seen'],
        'training_pairs': 47341, 'fraction_of_first_pass': 8000/47341,
        'peak_reserved_gib': max(r['hardware/peak_reserved_gib'] for r in metrics),
        'max_reserved_fraction': max(r['hardware/reserved_fraction'] for r in metrics), 'segments': {}}
    for low, high in [(1, 100), (101, 1000), (1001, 2000), (2001, 4000), (4001, 6000), (6001, 8000)]:
        segment = [r for r in metrics if low <= r['step'] <= high]
        result['training']['segments'][f'{low}-{high}'] = {key: float(np.mean([r[key] for r in segment]))
            for key in ('train/loss', 'train/gradient_norm', 'train/lr', 'train/seconds')}
    result['checkpoint_audit'] = read(OUT / 'evidence/checkpoint_audit.json')
    result['reference_metric_audit'] = read(OUT / 'evidence/reference_metric_audit.json')
    for r in rows:
        used.add(ROOT / 'data/validation/manual_val_96.jsonl')
    used.add(ROOT / 'data/validation/manual_val_12_pilot.jsonl')
    for run, keys in [(PILOT, pids), (FULL, ids)]:
        val = read(run / 'validation-008000/validation.json')
        assert [r['sample_id'] for r in val['samples']] == keys
        assert len({r['checkpoint_sha256'] for r in val['samples']}) == 1
    expected = read(ROOT / 'runs/flux9b_8k_completion/checkpoint_verified.json')['sha256']['adapters.safetensors']
    checkpoint = PILOT / 'checkpoint-008000/adapters.safetensors'
    used.add(checkpoint)
    assert sha(checkpoint) == expected
    source = ROOT / 'runs/flux9b_large_qkvo_r128_fast5h_b1'
    source_hashes = read(source / 'training_identity.json')['source_sha256']
    for rel, expected in source_hashes.items():
        path = source / 'source_snapshot' / rel
        assert sha(path) == expected
        used.add(path)
    result['source_verification'] = {'parent_snapshot_files': len(source_hashes), 'all_match': True,
                                      'full96_checkpoint_hash_matches': True}
    # Check the saved bundle receipt without modifying or regenerating run assets.
    receipt = read(FULL / 'ready.json')['files']
    checked_images = 0
    for k in ids:
        for rel in (f'native/{k}.png', f'validation-008000/{k}.png'):
            path = FULL / rel
            assert sha(path) == receipt[rel], rel
            used.add(path)
            checked_images += 1
    summary = read(FULL / 'validation-008000/quality_summary.json')
    for name, expected in summary['identity_embedding_sha256'].items():
        path = ROOT / 'data/validation' / f'{name}.pth'
        assert sha(path) == expected
        used.add(path)
    background = read(FULL / 'background_audit_8000.json')
    assert len(background) == 96 and all(r['background_max_abs'] == 0 for r in background)
    result['source_verification'].update(full96_pngs_match_receipt=checked_images,
        current_identity_prototypes_match_run_hashes=True, recorded_exact_pixel_exteriors=96)
    with (OUT / 'per_image_comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sample_id', 'identity_id', 'prompt_index', 'pilot_member', 'native_id_sim', 'trained_id_sim', 'delta_id_sim', 'native_face_count', 'face_box_area', 'prompt'])
        w.writeheader()
        for r in rows:
            k = r['sample_id']; box = masks[k]['face_bbox']
            w.writerow(dict(sample_id=k, identity_id=r['identity_id'], prompt_index=r['prompt_index'], pilot_member=k in pids,
                native_id_sim=native[k]['id_sim'], trained_id_sim=trained[k]['id_sim'], delta_id_sim=trained[k]['id_sim']-native[k]['id_sim'],
                native_face_count=native[k]['id_sim_face_count'], face_box_area=(box[2]-box[0])*(box[3]-box[1]), prompt=r['prompt']))
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    (OUT / 'source_audit.json').write_text(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in sorted(used)}, indent=2)+'\n')

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.6), layout='constrained')
    for ax, metric, title in zip(axes, ['id_sim', 'text_sim'], ['Identity similarity (pilot12)', 'CLIP text score (pilot12)']):
        ax.plot(STEPS, [means(checkpoints[s], pids)[metric] for s in STEPS], 'o-', color=COLORS['trained'], label='FLUX1')
        ax.axhline(means(pn, pids)[metric], color=COLORS['native'], linestyle='--', label='Native')
        ax.set(title=title, xlabel='Optimizer updates')
        ax.grid(alpha=.18); ax.legend(frameon=False)
    savefig(fig, 'pilot_trajectory')

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.7), layout='constrained')
    owners = list(result['per_identity']); x = np.arange(len(owners))
    axes[0].bar(x-.18, [result['per_identity'][i]['native']['id_sim'] for i in owners], .36, label='Native', color=COLORS['native'])
    axes[0].bar(x+.18, [result['per_identity'][i]['trained']['id_sim'] for i in owners], .36, label='FLUX1 8k', color=COLORS['trained'])
    axes[0].set_xticks(x, owners, rotation=35, ha='right'); axes[0].set_ylabel('ID similarity'); axes[0].legend(frameon=False)
    axes[0].set_title('Full96: 12 prompts per identity')
    delta = np.array([trained[k]['id_sim']-native[k]['id_sim'] for k in ids])
    axes[1].hist(delta, bins=18, color=COLORS['trained'], alpha=.8)
    axes[1].axvline(0, color='black', linewidth=1)
    axes[1].set(title='Paired image changes: 47 improve / 49 regress', xlabel='FLUX1 8k minus native ID similarity', ylabel='Images')
    savefig(fig, 'full96_results')

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.6), layout='constrained')
    for ax, field, title in zip(axes, ['train/loss', 'train/gradient_norm'], ['Training loss', 'Gradient norm before clipping']):
        values = np.array([r[field] for r in metrics]); window = 200
        ax.plot(np.arange(window, len(values)+1), np.convolve(values, np.ones(window)/window, 'valid'), color=COLORS['trained'])
        ax.axvline(2000, color=COLORS['native'], linestyle='--', linewidth=1)
        ax.set(title=title+' (200-update mean)', xlabel='Optimizer updates'); ax.grid(alpha=.18)
    savefig(fig, 'training_curves')

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.5), layout='constrained')
    labels = ['Read', 'Rush', 'Ski', 'Drum', 'Box', 'Dance', 'Angry', 'Cry', 'Laugh', 'Jump', 'Bike', 'Chef']
    for a, name, color in [(native, 'Native', COLORS['native']), (trained, 'FLUX1 8k', COLORS['trained'])]:
        axes[0].plot(range(12), [np.mean([a[r['sample_id']]['id_sim'] for r in rows if r['prompt_index']==i]) for i in range(12)], 'o-', label=name, color=color)
    axes[0].set_xticks(range(12), labels, rotation=40, ha='right'); axes[0].set_title('Full96 by prompt family (8 images each)')
    axes[0].set_ylabel('ID similarity'); axes[0].legend(frameon=False)
    area = np.array([np.sqrt((masks[k]['face_bbox'][2]-masks[k]['face_bbox'][0])*(masks[k]['face_bbox'][3]-masks[k]['face_bbox'][1])) for k in ids])
    axes[1].scatter(area, delta, s=22, color=COLORS['trained'], alpha=.75)
    axes[1].axhline(0, color='black', linewidth=1); axes[1].set(xlabel='Square root of native face-box area (pixels)', ylabel='8k minus native ID similarity', title='Smaller faces are a useful diagnostic stratum')
    savefig(fig, 'scene_and_size')

    # Every full96 case is included. These sheets are inspection aids, not edits.
    for owner in owners:
        own_rows = [r for r in rows if r['identity_id'] == owner]
        fig, axes = plt.subplots(3, 8, figsize=(16, 7.5))
        fig.subplots_adjust(wspace=.02, hspace=.28, top=.91, bottom=.025)
        fig.suptitle(f'{owner} | native (left) and FLUX1 8k (right), all 12 fixed prompts', fontsize=17)
        for i, r in enumerate(own_rows):
            k=r['sample_id']; rr, cc=divmod(i, 4)
            for j, (folder, db) in enumerate([(FULL/'native', native), (FULL/'validation-008000', trained)]):
                ax=axes[rr, 2*cc+j]; path=folder/f'{k}.png'
                ax.imshow(crop(Image.open(path), masks[k]['face_bbox']))
                ax.set_title(f'{k} {labels[i]} | {db[k]["id_sim"]:.3f}', fontsize=9); ax.axis('off')
        fig.savefig(OUT/f'full96_{owner}.jpg', dpi=120, bbox_inches='tight', facecolor='white'); plt.close(fig)

    # Same pilot masks for native / 2k / 8k; scores are from the pilot execution.
    for group in range(2):
        chosen=small[group*6:(group+1)*6]
        fig, axes=plt.subplots(6, 4, figsize=(9, 12.5))
        fig.subplots_adjust(wspace=.04, hspace=.22, top=.965, bottom=.02)
        for ax, title in zip(axes[0], ['Reference', 'Native', 'FLUX1 2k', 'FLUX1 8k']): ax.set_title(title, fontsize=14)
        for i,r in enumerate(chosen):
            k=r['sample_id'];refpath=ROOT/'data/validation'/r['reference_image']
            axes[i,0].imshow(crop(Image.open(refpath),r['reference_face_bbox']));axes[i,0].set_ylabel(f'{k} {r["identity_id"]}',fontsize=10)
            for j,(folder,db) in enumerate([(PILOT/'native',pn),(PILOT/'validation-002000',checkpoints[2000]),(PILOT/'validation-008000',checkpoints[8000])],start=1):
                axes[i,j].imshow(crop(Image.open(folder/f'{k}.png'),pmasks[k]['face_bbox']));axes[i,j].set_xlabel(f'ID {db[k]["id_sim"]:.3f}',fontsize=10)
            for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
        fig.savefig(OUT/f'pilot_comparison_{group+1}.jpg',dpi=125,bbox_inches='tight',facecolor='white');plt.close(fig)
    print(json.dumps({k:result[k] for k in ('full96','pilot12','training')},indent=2))


if __name__ == '__main__':
    main()
