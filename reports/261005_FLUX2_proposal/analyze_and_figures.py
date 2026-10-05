"""CPU-only evidence audit and original figures for the Flux 2 design report.

AICODE-NOTE: Flux 2 is a proposal. This script never loads a generator, changes
training sources, downloads model weights, or contacts a GPU host.
Run: envs/report/bin/python reports/261005_FLUX2_proposal/analyze_and_figures.py
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / 'reports/261005_FLUX1a_results'
EVIDENCE = OUT / 'evidence'
FIG = OUT / 'figures'
for p in (EVIDENCE, FIG):
    p.mkdir(parents=True, exist_ok=True)

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

paths = [OLD / 'analysis.json', OLD / 'evidence/None_native_checks.json',
         OLD / 'evidence/None_routing_masks.json', OLD / 'selection.json',
         ROOT / 'reports/261004_FLUX1_review/evidence/full96_2k_8k_comparison.json',
         ROOT / 'reports/261004_FLUX1_review/evidence/reference_metric_audit.json',
         ROOT / 'runs/FLUX1a_vast9B_20261004/checkpoint-002000/manifest.json',
         ROOT / 'runs/FLUX1a_vast9B_20261004/comet_verified_006000.json']
csv_paths = {s: OLD / f'evidence/{s}_quality_per_image.csv' for s in (0, 2000, 4000, 6000)}
csv_paths['native'] = ROOT / 'reports/261004_FLUX1_review/evidence/native96_quality_per_image.csv'
paths.extend(csv_paths.values())
S = {}
for s, path in csv_paths.items():
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 96 and len({r['sample_id'] for r in rows}) == 96
    S[s] = {r['sample_id']: {k: v if k in ('sample_id', 'identity_id') else float(v)
                            for k, v in r.items()} for r in rows}
sample_ids = list(S[6000])
assert all(list(s) == sample_ids for s in S.values())
owners = sorted({r['identity_id'] for r in S[6000].values()})
assert len(owners) == 8
assert all(sum(r['identity_id'] == o for r in S[6000].values()) == 12 for o in owners)
metrics = {str(s): {k: float(np.mean([r[k] for r in rows.values()]))
                   for k in ('id_sim', 'text_sim')} for s, rows in S.items()}
old = read(OLD / 'analysis.json')
for s in metrics:
    for k, value in metrics[s].items():
        assert abs(value - old['metrics'][s][k]) < 1e-12

def paired(a, b):
    d = np.array([S[b][i]['id_sim'] - S[a][i]['id_sim'] for i in sample_ids])
    g = np.array([np.mean([d[j] for j, i in enumerate(sample_ids)
                          if S[b][i]['identity_id'] == owner]) for owner in owners])
    rng = np.random.default_rng(142)
    ci = np.quantile(g[rng.integers(0, 8, (100000, 8))].mean(1), [.025, .975])
    return dict(delta=float(d.mean()), wins=int((d > 0).sum()),
                identity_wins=int((g > 0).sum()), ci95=ci.tolist(),
                by_identity=dict(zip(owners, g.tolist())))

masks = read(OLD / 'evidence/None_routing_masks.json')['samples']
wh = np.array([[masks[i]['face_bbox'][2] - masks[i]['face_bbox'][0],
                masks[i]['face_bbox'][3] - masks[i]['face_bbox'][1]] for i in sample_ids])
short = wh.min(1)
checks = read(OLD / 'evidence/None_native_checks.json')
audit = {'date': '2026-10-05', 'status': 'Flux 2 unimplemented and unmeasured',
         'metrics': metrics, 'paired': {f'{b}-{a}': paired(a, b) for a, b in
                                      [('native', 4000), ('native', 6000), (4000, 6000)]},
         'face_geometry': {'short_side_quantiles_pixels': np.quantile(short, [0, .25, .5, .75, 1]).tolist(),
                           'short_side_below_112': int((short < 112).sum()),
                           'median_short_side_tokens': float(np.median(short) / 16),
                           'area_fraction_quantiles': np.quantile(wh.prod(1) / 768**2, [0, .25, .5, .75, 1]).tolist()},
         'admission': {k: checks[k] for k in ('trainable_parameters', 'native_off_exact',
                      'zero_mask_exact', 'all_frozen_parameters_exact', 'peak_reserved_gib',
                      'identity_gradient_norm', 'flow_gradient_norm_at_id_sigma',
                      'weighted_identity_to_flow_gradient_ratio')},
         'evidence_hashes': {str(p.relative_to(ROOT)): sha(p) for p in paths}}

sources = ['ba_dit/nn/isolated_reference.py', 'ba_dit/nn/masked_face_attention.py',
           'ba_dit/nn/online_identity_loss.py', 'ba_dit/backends/flux_runtime.py',
           'ba_dit/nn/masked_face_flow.py', 'scripts/online_face_ba.py',
           'patches/flux2_reference_branch_and_offload.patch']
audit['deployment_source_audit'] = {}
with tarfile.open(ROOT / 'scratch/FLUX1a_deploy/source.tar.gz') as archive:
    for name in sources:
        data = archive.extractfile(name).read()
        dest = EVIDENCE / 'deployed_sources' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        audit['deployment_source_audit'][name] = {
            'deployment_sha256': hashlib.sha256(data).hexdigest(),
            'local_sha256': sha(ROOT / name), 'local_equals_deployed': data == (ROOT / name).read_bytes()}
audit['checkpoint_identity'] = read(ROOT / 'runs/FLUX1a_vast9B_20261004/checkpoint-002000/manifest.json')['identity']
manifest6 = Path('/mnt/c/Users/ogure/FLUX1a_checkpoints/FLUX1a_vast9B_20261004/checkpoint-006000/manifest.json')
if manifest6.exists():
    m2 = read(ROOT / 'runs/FLUX1a_vast9B_20261004/checkpoint-002000/manifest.json')
    m6 = read(manifest6)
    assert m6['step'] == 6000 and m6['identity'] == m2['identity']
    assert m6['training_code_sha256'] == m2['training_code_sha256']
    audit['checkpoint_6k_manifest'] = {'path': str(manifest6), 'sha256': sha(manifest6),
                                      'identity_and_training_digest_match_2k': True}
    shutil.copyfile(manifest6, EVIDENCE / 'checkpoint_006000_manifest.json')
(OUT / 'analysis.json').write_text(json.dumps(audit, indent=2) + '\n')
for s, p in csv_paths.items():
    shutil.copyfile(p, EVIDENCE / f'{s}_quality_per_image.csv')
for sid in ('00', '65'):
    shutil.copyfile(OLD / f'sample_{sid}.jpg', EVIDENCE / f'sample_{sid}.jpg')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
TEAL, NAVY, ORANGE, RED = '#087f8c', '#203b53', '#b66c2b', '#b34747'
BLUE, PALE, GREY = '#e6f3f5', '#f1f4f7', '#647582'

def save(fig, name):
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=190, bbox_inches='tight', facecolor='white')
    plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.3, 3.3), layout='constrained')
for a, key, title in zip(ax, ('id_sim', 'text_sim'), ('Owner-matched ID similarity', 'CLIP text alignment (logits)')):
    x = [2000, 4000, 6000]
    y = [metrics[str(s)][key] for s in x]
    a.plot(x, y, 'o-', lw=2.3, color=TEAL, label='Flux 1A')
    a.axhline(metrics['native'][key], ls='--', lw=1.4, color=GREY, label='Native fixed96')
    for xx, yy in zip(x, y):
        a.annotate(f'{yy:.4f}' if key == 'id_sim' else f'{yy:.3f}', (xx, yy),
                   xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10)
    a.set(title=title, xlabel='Optimizer updates', xticks=x)
    a.grid(alpha=.16)
    a.margins(y=.4)
    a.legend(fontsize=9, loc='lower right')
save(fig, 'measured_trajectory')

fig, ax = plt.subplots(1, 2, figsize=(10.3, 4.3), layout='constrained')
yy = np.arange(8)
for step, delta, color in [(4000, -.16, TEAL), (6000, .16, ORANGE)]:
    values = [audit['paired'][f'{step}-native']['by_identity'][o] for o in owners]
    ax[0].barh(yy + delta, values, height=.3, color=color, label=f'{step//1000}K − native')
ax[0].set(yticks=yy, yticklabels=owners, xlabel='Mean owner-ID change', title='Gains differ across identities')
ax[0].axvline(0, color=GREY, lw=1)
ax[0].legend(fontsize=9)
ax[1].hist(short, bins=[40, 64, 80, 96, 112, 128, 160, 192, 256], color=TEAL, edgecolor='white')
ax[1].axvline(112, color=ORANGE, ls='--', lw=1.5)
ax[1].set(xlabel='Native owner-box short side (pixels)', ylabel='Images',
          title='59/96 faces are below 112 pixels')
save(fig, 'identity_and_resolution')

def canvas(height=6):
    fig, ax = plt.subplots(figsize=(11, height))
    ax.set(xlim=(0, 110), ylim=(0, height*10))
    ax.invert_yaxis()
    ax.axis('off')
    return fig, ax

def box(ax, x, y, w, h, title, body='', color=BLUE, edge=TEAL, fontsize=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.4,rounding_size=1',
                               linewidth=1.2, facecolor=color, edgecolor=edge))
    title_artist = ax.text(x+w/2, y+2.1, title, ha='center', va='top', fontsize=fontsize,
                          color=NAVY, weight='bold')
    artists = [title_artist]
    if body:
        artists.append(ax.text(x+w/2, y+h/2+1.4, body, ha='center', va='center', fontsize=fontsize-1,
                               color=NAVY, linespacing=1.35))
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    allowed = abs(ax.transData.transform((x+w-1, y))[0] - ax.transData.transform((x+1, y))[0])
    for artist in artists:
        while artist.get_window_extent(renderer).width > allowed:
            artist.set_fontsize(artist.get_fontsize() - .25)

def arrow(ax, p, q, color=GREY, label=None, style='-', rad=0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=13,
                                linewidth=1.5, color=color, linestyle=style,
                                connectionstyle=f'arc3,rad={rad}'))
    if label:
        ax.text((p[0]+q[0])/2, (p[1]+q[1])/2-1, label, ha='center', va='bottom',
                fontsize=10, color=color, bbox=dict(facecolor='white', edgecolor='none', pad=1))

fig, ax = canvas(6.5)
box(ax, 2, 3, 27, 12, 'Reference image', 'Clean latent + original RoPE\nConditioned on current σ')
box(ax, 40, 3, 29, 12, 'Image-only pass', 'No target or text tokens\nImmutable K/V at 8 sites')
box(ax, 81, 3, 27, 12, 'Reference read', 'Native Q + ΔQ\nBank K/V + ΔK/ΔV')
arrow(ax, (29, 9), (40, 9)); arrow(ax, (69, 9), (81, 9))
box(ax, 2, 29, 27, 14, 'Evolving target scene', 'Face starts from noise\nNative exterior is clamped', color=PALE, edge=GREY)
box(ax, 40, 29, 29, 14, 'Native joint backbone', 'Text + target + reference\n32 blocks; 8 BA sites', color=PALE, edge=GREY)
box(ax, 81, 29, 27, 14, 'Owned attention rows', 'Reference message replaces\nnative attention at those sites')
arrow(ax, (29, 36), (40, 36)); arrow(ax, (69, 36), (81, 36))
arrow(ax, (94, 15), (94, 29), TEAL)
arrow(ax, (67, 29), (84, 15), ORANGE, 'query features')
box(ax, 19, 52, 72, 10, 'Native residual + gate + MLP + later blocks remain',
    'Isolated memory does not give the branch exclusive control.', color='#fbf0e5', edge=ORANGE)
arrow(ax, (95, 43), (81, 52)); arrow(ax, (55, 43), (48, 52), ORANGE, 'residual path')
save(fig, 'flux1a_information_flow')

fig, ax = canvas(7.2)
box(ax, 2, 2, 32, 13, 'Native scene (frozen)', 'Prompt / seed / reference\nSave scene + owner mask', color=PALE, edge=GREY)
box(ax, 44, 2, 27, 13, 'Erase before encoding', 'Erase face + halo in RGB\nEncode sanitized scene', color=PALE, edge=GREY)
box(ax, 80, 2, 28, 13, 'Read-only context C', 'Sanitized exterior + text T\nNo native-face feedback', color=PALE, edge=GREY)
arrow(ax, (34, 8), (44, 8)); arrow(ax, (71, 8), (80, 8))
box(ax, 2, 27, 32, 16, 'Reference memory (fixed)', 'ArcFace → 4 ID tokens\nFace-crop ViT → 64 details\nReference image only')
box(ax, 44, 27, 64, 16, 'Persistent face stream F · all 32 blocks',
    'Fresh noise → own Q/K/V, residuals and MLP\nID/detail reads + ID modulation in every block\nShared frozen weights; no native-face state copied')
arrow(ax, (34, 35), (44, 35), TEAL)
arrow(ax, (94, 15), (94, 27), GREY, 'C/T → F only')
box(ax, 44, 55, 33, 13, 'Native head on F', 'Euler updates face latent\n20 steps · CFG 4')
box(ax, 86, 55, 22, 13, 'Decode + compose', 'Keep native pixels\noutside output support')
arrow(ax, (61, 43), (61, 55), TEAL); arrow(ax, (77, 61), (86, 61), TEAL)
ax.text(2, 54, 'Flux 2 proposal\nNo measured quality gain yet.\nBA-off dispatches to the exact\noriginal native implementation.', fontsize=12, color=ORANGE, va='top', linespacing=1.5)
save(fig, 'flux2_whole_model')

fig, ax = plt.subplots(figsize=(10.5, 4.9), layout='constrained')
matrix = np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 1, 0]])
ax.imshow(matrix, cmap=matplotlib.colors.ListedColormap(['#f6e8e7', '#dceff0']), vmin=0, vmax=1, aspect='auto')
ax.set(xticks=range(5), xticklabels=['Text T', 'Sanitized C', 'Face F', 'Reference R', 'Native face N'],
       yticks=range(4), yticklabels=['Text T queries', 'Context C queries', 'Face F queries', 'Reference encoder'])
ax.xaxis.tick_top()
for row in range(4):
    for col in range(5):
        text = 'READ' if matrix[row, col] else 'BLOCK'
        if row == 2 and col == 3:
            text = 'SEPARATE\nID / DETAIL READS'
        ax.text(col, row, text, ha='center', va='center', fontsize=12,
                color=TEAL if matrix[row, col] else RED)
ax.set_title('Allowed information flow at every layer\nReference memory remains read-only during denoising', pad=42)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0, pad=12)
save(fig, 'flux2_attention_firewall')

fig, ax = canvas(7.0)
box(ax, 2, 2, 25, 12, 'Face state Fℓ', 'Own persistent state\nNo native-state reset')
box(ax, 40, 2, 30, 12, 'Face-only modulation', 'Timestep + identity\nshift / scale / gate')
box(ax, 82, 2, 26, 12, 'Memory R', 'Fixed ID + detail tokens\nNo scene RoPE')
arrow(ax, (27, 8), (40, 8)); arrow(ax, (82, 8), (70, 8), TEAL)
box(ax, 2, 29, 30, 14, 'Scene self-attention', 'Q: F · K/V: F, C, T\nNative QK norm + RoPE', color=PALE, edge=GREY)
box(ax, 40, 29, 30, 14, 'ID cross-attention', 'Own softmax: 4 ID tokens\nTrainable Q/K/V/output')
box(ax, 80, 29, 28, 14, 'Detail cross-attention', 'Own softmax: 64 tokens\nIndependent strength')
arrow(ax, (49, 14), (18, 29)); arrow(ax, (56, 14), (55, 29)); arrow(ax, (64, 14), (94, 29))
arrow(ax, (91, 14), (59, 29), TEAL, style='--'); arrow(ax, (99, 14), (99, 29), TEAL, style='--')
box(ax, 15, 55, 82, 12, 'Owned residual → native MLP with identity modulation',
    'Separate ID/detail gates bypass the native attention gate.')
for x, end in [(17, 30), (55, 55), (94, 82)]:
    arrow(ax, (x, 43), (end, 55), TEAL if x != 17 else GREY)
save(fig, 'flux2_block')

fig, ax = canvas(5.6)
box(ax, 2, 3, 29, 13, 'Training photo + fresh noise', 'zσ = (1 − σ) z₀ + σ ε\nKeep pinned native σ sampler', color=PALE, edge=GREY)
box(ax, 42, 3, 27, 13, 'Flux 2 prediction vθ', 'Same patched backend\nReference-conditioned face stream')
box(ax, 80, 3, 28, 13, 'Masked native flow loss', 'Target velocity ε − z₀\nEvery original training row', color=PALE, edge=GREY)
arrow(ax, (31, 10), (42, 10)); arrow(ax, (69, 10), (80, 10))
box(ax, 8, 32, 36, 15, 'Differentiable clean estimate', 'ẑ₀ = zσ − σ vθ\nFrozen VAE + fixed training alignment\nFrozen recognizer keeps input gradients')
box(ax, 57, 32, 45, 15, 'Identity + contrastive supervision', 'Positive: same-ID target / reference\nNegatives: verified other-ID embedding bank\nLog weighted gradient ratios by σ bin')
arrow(ax, (50, 16), (26, 32), TEAL); arrow(ax, (44, 39), (57, 39), TEAL)
ax.text(55, 53, 'Low σ suppresses the identity-to-velocity gradient by σ. Test useful σ coverage; do not change the flow sampler silently.',
        ha='center', fontsize=10, color=ORANGE)
save(fig, 'training_objectives')

fig, ax = canvas(5.4)
for x, w, title, body, col in [
    (2, 33, 'Flux 2 · large GPU', '9B, 768×768 target\n32 sites, branch width 512\nFull-depth online denoising\n80GB / GB10-class admission', BLUE),
    (39, 32, 'Flux 2-Lite · local test', '4B, 768×768 target\n25 sites, branch width 128\nCheckpointing + staged encoders\n16GB fit must be measured', BLUE),
    (76, 32, 'Flux 2-Lite-ROI · fallback', '4B, 384×384 face window\nSame all-layer architecture\n256-token sanitized scene context\nSeparate named crop protocol', '#fbf0e5')]:
    box(ax, x, 4, w, 30, title, body, color=col, fontsize=13)
ax.text(55, 43, 'Shared contract: immutable reference memory • own face state • directed context • full native flow objective',
        ha='center', fontsize=11, color=NAVY)
ax.text(55, 49, 'Only historical Flux 1 / Flux 1A memory is measured. These profiles are design targets, not launch-ready configs.',
        ha='center', fontsize=11, color=ORANGE)
save(fig, 'hardware_profiles')
print(json.dumps({'metrics_recomputed': True, 'fixed96_order_verified': True,
                  'geometry': audit['face_geometry'], 'figures': 8}, indent=2))
