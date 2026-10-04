"""Build the completed one-ID architecture presentation from immutable run evidence.

CPU-only report generation. No inference, training, network calls or job changes.
"""
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image
from matplotlib.patches import Rectangle

import report_style as s
from report_style import (C, plt, PdfPages, page, text, para, box, arrow, card,
                          banner, rect, snippet, table)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flux4b_oneid_online_face_qkvo_r128_768_20261002'
SNAP = RUN / 'source_snapshot'
RESULT = ROOT / 'plans/260930/ONLINE_FACE_BA_RESULTS.json'
MODEL = ROOT / 'sources/ai-toolkit-flux/extensions_built_in/diffusion_models/flux2/src/model.py'
PDF = OUT / 'FLUX1_architecture_and_one_id_results.pdf'
E = json.loads(RESULT.read_text())
MASKS = json.loads((RUN / 'routing_masks.json').read_text())
NATIVE = Path(MASKS['source'])
ROWS = [json.loads(x) for x in (ROOT / 'data/datasets/one_id/validation_24_seeds01.jsonl').read_text().splitlines()]
IDS = [r['sample_id'] for r in ROWS]
N = json.loads((NATIVE / 'quality_summary.json').read_text())['metrics']
M = E['metrics']
COLORS = ['#657B85', C['orange'], '#4990A5', C['green'], C['purple']]
FOOT = 'Completed 2 Oct 2026 · fixed24 one-ID diagnostic · immutable run source and measured artifacts · rsrch_new'
used_files = {RESULT, RUN / 'identity.json', RUN / 'routing_masks.json', RUN / 'resolved_config.yaml', MODEL}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_scores(folder):
    p = folder / 'quality_per_image.csv'
    used_files.add(p)
    return {r['sample_id']: {k: float(v) for k, v in r.items() if k not in ('sample_id', 'identity_id')}
            for r in csv.DictReader(p.open())}


SCORES = {'native': read_scores(NATIVE), **{str(i): read_scores(RUN / f'validation-{i:06d}') for i in (0, 500, 1000, 2000)}}
D = np.array([SCORES['1000'][k]['id_sim'] - SCORES['native'][k]['id_sim'] for k in IDS])


def verify_evidence():
    for path, expected in E['source_sha256'].items():
        p = SNAP / path
        assert sha(p) == expected, path
        used_files.add(p)
    diff = subprocess.check_output(['git', '-C', str(ROOT / 'sources/ai-toolkit-flux'), 'diff', '--binary'])
    assert hashlib.sha256(diff).hexdigest() == E['model_identity']['source_patch_sha256']
    assert subprocess.check_output(['git', '-C', str(ROOT / 'sources/ai-toolkit-flux'), 'rev-parse', 'HEAD'], text=True).strip() == E['model_identity']['source_commit']
    for step in (0, 500, 1000, 2000):
        p = RUN / f'checkpoint-{step:06d}/adapters.safetensors'
        assert sha(p) == E['checkpoint_adapters_sha256'][str(step)]
        q = RUN / f'validation-{step:06d}/quality_summary.json'
        used_files.add(q)
        assert json.loads(q.read_text())['metrics'] == M[str(step)]
        report = json.loads((q.parent / 'validation.json').read_text())
        assert [x['sample_id'] for x in report['samples']] == IDS
        assert all(x['checkpoint_sha256'] == E['checkpoint_adapters_sha256'][str(step)] for x in report['samples'])
    assert len(IDS) == 24 and len(set(IDS)) == 24
    assert abs(N['id_sim'] - E['native_panel_id_sim']) < 1e-9
    for k in IDS:
        assert sha(RUN / 'native' / f'{k}.png') == MASKS['samples'][k]['baseline_image_sha256']
        assert sha(NATIVE / f'{k}.png') == sha(RUN / 'native' / f'{k}.png')
    assert np.count_nonzero(D > 0) == 23
    for metric, values in [('native', N), *M.items()]:
        for key in ('id_sim', 'text_sim'):
            assert abs(np.mean([SCORES[metric][k][key] for k in IDS]) - values[key]) < 1e-10


def new(title, sub, source=FOOT):
    return page('FLUX1 | ' + title, sub, source)


def source(rel):
    return SNAP / rel


def excerpt(ax, x, y, w, h, rel, first, last, size=14):
    p = MODEL if rel == 'model.py' else source(rel)
    used_files.add(p)
    snippet(ax, x, y, w, h, p, first, last, size)


def picture(fig, path, x, y, w, h, crop=None):
    used_files.add(path)
    im = Image.open(path).convert('RGB')
    if crop:
        im = im.crop(crop)
    original_size = im.size
    im.thumbnail((int(w * 1.5), int(h * 1.5)), Image.Resampling.LANCZOS)
    ax = fig.add_axes([x / 1600, 1 - (y + h) / 900, w / 1600, h / 900])
    ax.imshow(im, extent=(0, original_size[0], original_size[1], 0))
    ax.axis('off')
    return ax


def face_crop(key):
    x0, y0, x1, y1 = MASKS['samples'][key]['face_bbox']
    side = max(x1 - x0, y1 - y0) * 1.35
    return tuple(map(int, ((x0+x1-side)/2, (y0+y1-side)/2, (x0+x1+side)/2, (y0+y1+side)/2)))


def image_path(stage, key):
    return RUN / ('native' if stage == 'native' else f'validation-{int(stage):06d}') / f'{key}.png'


def mathline(ax, x, y, value, size=22, color=None):
    return text(ax, x, y, value, size, color)


def charts_axis(fig, x, y, w, h):
    a = fig.add_axes([x/1600, 1-(y+h)/900, w/1600, h/900], facecolor='white')
    a.spines[['top', 'right']].set_visible(False)
    a.spines[['left', 'bottom']].set_color(C['line'])
    a.tick_params(labelsize=12, colors=C['muted'])
    a.grid(axis='y', color=C['line'], alpha=.4)
    a.set_axisbelow(True)
    return a


def build():
    verify_evidence()
    # 1: identify the exact completed experiment and its strongest measured point.
    fig, ax = new('One identity, eight attention branches',
        'FLUX.2-klein Base 4B · online masked Q/K/V/O, rank 128 · completed architecture and results · 3 October 2026')
    text(ax, 58, 163, 'Reference attention inside\nthe complete FLUX denoiser', 31, C['teal'], 'bold')
    para(ax, 62, 282, 'Only branch-local low-rank projections learn. Native attention, modulation, MLPs and the velocity head remain in the computation.', 57, 21)
    card(ax, 55, 430, 440, 170, 'BEST ID SCORE: 0.4303', 'Native: 0.3314. Absolute gain +0.0989 at update 1,000.', size=18)
    card(ax, 525, 430, 440, 170, '23 / 24 PAIRED GAINS', 'Best checkpoint versus native, on the same prompts and seeds.', size=18)
    card(ax, 55, 625, 910, 171, 'SCOPE OF THE RESULT', '19 same-person training pairs; 12 prompts × 2 seeds. CLIP text score falls 28.0729 → 26.9670. This is identity fitting evidence; unseen-person generalization remains untested.', size=17)
    picture(fig, image_path('1000', 'oneid_06_s1'), 1010, 165, 535, 610)
    text(ax, 1275, 792, 'Measured output · BA 1,000 · oneid_06_s1', 13, C['muted'], ha='center')

    # 2: distinguish actual current architecture from the earlier report.
    fig, ax = new('Which branched-attention experiment is this?',
        'The present deck covers online_masked_qkvo_v1; earlier architectures are context, not interchangeable evidence.')
    table(ax, 45, 162, [265, 405, 410, 430], ['Path', 'Attention / learned path', 'Training computation', 'Initialization / evaluation'], [
        ('Earlier K/V delta', 'Native joint attention + γ(R1 − R0); K/V adapters.', 'Full transformer; rank-16 configuration in the first architecture deck.', 'Zero delta reproduces native. Earlier fixed12 diagnostic.'),
        ('Cached face heads', 'Separate final-feature reference heads; several width/depth variants.', 'Cached deep features; restricted training cases.', 'Generated-mask spatial protocol introduced; separate runs.'),
        ('This completed run', 'Masked native/reference mixture; train Q, K, V and output LoRA.', 'Full denoiser autograd; fresh noise and native timestep draws.', 'BA-on step 0 is a separate routed baseline. Fixed24 at 0/500/1k/2k.')
    ], row_h=162, size=16)
    banner(ax, 742, 'CL39-inspired principle: a separately controlled reference message. This implementation uses a different FLUX routing rule.', size=17)

    # 3: whole model, exportable vector diagram.
    fig, ax = new('Whole model: one frozen denoiser, eight trainable branches',
        '768 × 768 target · 512 × 512 validation reference · width 3,072 · 24 heads × 128 dimensions')
    box(ax, 50, 170, 325, 90, 'Prompt → frozen Qwen3-4B\n512 × 7,680 features', 'blue', 18)
    box(ax, 50, 312, 325, 95, 'Noisy target z(t)\n2,304 × 128 packed tokens', 'blue', 18)
    box(ax, 50, 463, 325, 95, 'Clean reference → frozen VAE\n1,024 × 128 tokens', 'blue', 18)
    box(ax, 450, 190, 315, 94, 'Native text/image input\nprojections → width 3,072', 'green', 18)
    box(ax, 450, 366, 315, 110, '5 double-stream blocks\nBA at 1, 2, 3, 4', 'green', 20)
    box(ax, 855, 366, 315, 110, '20 single-stream blocks\nBA at 3, 8, 13, 18', 'green', 20)
    box(ax, 1245, 366, 300, 110, 'Native velocity head\nkeep target rows', 'green', 18)
    box(ax, 1245, 584, 300, 95, 'Euler integration\n→ frozen VAE decode', 'blue', 18)
    box(ax, 860, 584, 305, 95, 'Native t / RoPE / gates\nshared by every block', 'blue', 17)
    box(ax, 445, 584, 320, 95, 'Trainable at each BA site\nΔQ, ΔK, ΔV, ΔO', 'orange', 20)
    arrow(ax, [(375, 215), (450, 215)])
    arrow(ax, [(375, 359), (405, 359), (405, 258), (450, 258)])
    arrow(ax, [(375, 510), (425, 510), (425, 270), (450, 270)])
    arrow(ax, [(607, 284), (607, 366)])
    arrow(ax, [(765, 421), (855, 421)])
    arrow(ax, [(1170, 421), (1245, 421)])
    arrow(ax, [(1395, 476), (1395, 584)])
    arrow(ax, [(605, 584), (605, 476)], 'orange')
    arrow(ax, [(765, 627), (814, 627), (814, 502), (1005, 502), (1005, 476)], 'orange')
    arrow(ax, [(1075, 584), (1075, 476)], 'teal')
    banner(ax, 756, 'Green = native frozen operations · orange = trainable adapters · one shared model for training and validation', size=17)

    # 4: exact placement and differentiability.
    fig, ax = new('Exact insertion sites and gradient path',
        'Zero-based block indices; the same eight sites are active in training and both validation CFG lanes.')
    text(ax, 55, 165, 'Double stream', 23, C['teal'], 'bold')
    for i in range(5):
        box(ax, 330+i*190, 160, 150, 100, str(i), 'orange' if i in (1,2,3,4) else 'palegreen', 25)
    text(ax, 55, 324, 'Single stream', 23, C['teal'], 'bold')
    for i in range(20):
        box(ax, 55+i*75, 390, 65, 88, str(i), 'orange' if i in (3,8,13,18) else 'palegreen', 20)
    arrow(ax, [(120, 520), (1490, 520)], 'teal')
    text(ax, 800, 545, 'Forward: native transformer with local attention routing', 17, C['teal'], ha='center')
    arrow(ax, [(1490, 618), (120, 618)], 'orange')
    text(ax, 800, 640, 'Backward: loss → frozen operations → all earlier trainable branches', 17, C['orange'], ha='center')
    banner(ax, 740, 'Frozen weights still transmit activation gradients. Deep reference states remain target-dependent through joint attention.', size=18)

    # 5: exact absolute indices.
    fig, ax = new('Token layout: preserve positions while selecting face support',
        'Measured validation layout: 512 text + 2,304 target + 1,024 reference = 3,840 joint tokens.')
    box(ax, 55, 179, 240, 95, 'Text\n[0, 512)', 'blue', 19)
    box(ax, 300, 179, 820, 95, 'Target image · 48 × 48\n[512, 2816)', 'palegreen', 21)
    box(ax, 1125, 179, 420, 95, 'Reference image · 32 × 32\n[2816, 3840)', 'paleorange', 19)
    table(ax, 55, 325, [330, 580, 580], ['Coordinate system', 'Face query indices', 'Reference key/value indices'], [
        ('Image stream / double', 'F = {i in [0, 2304): m[i] > 0}', '2304 + J'),
        ('Joint stream / single', '512 + F', '512 + 2304 + J'),
        ('Reference face support', 'Fractional target mask; 128-query chunks.', 'J: 88 of 1,024 tokens on this reference; cap 512.')
    ], row_h=106, size=17)
    para(ax, 65, 722, 'Native joint attention still sees every native token. The 512-key cap applies only to the extra reference read. Gather original RoPE rows for both queries and keys; never renumber gathered positions.', 128, 18)

    # 6: central BA graph, standalone export.
    fig, ax = new('Inside one branch: face queries read reference-face keys',
        'H is the block’s native normalized/modulated input; orange projections are the only learned components.')
    box(ax, 45, 167, 305, 95, 'Full native Q/K/V\ntext + target + reference', 'palegreen', 18)
    box(ax, 437, 167, 327, 95, 'Native joint attention N\nall native keys/values', 'green', 18)
    box(ax, 873, 167, 327, 95, 'Route: A = (1 − m)N + mR\non selected target rows', 'paleorange', 18)
    box(ax, 1280, 167, 275, 95, 'Native output projection\nP(A)', 'green', 17)
    arrow(ax, [(350, 214), (437, 214)])
    arrow(ax, [(764, 214), (873, 214)])
    arrow(ax, [(1200, 214), (1280, 214)])
    box(ax, 45, 375, 305, 120, 'Q(native face rows)\n+ ΔQ(H_face)\n→ native Q norm + RoPE', 'orange', 18)
    box(ax, 45, 589, 305, 122, 'K/V(native reference rows)\n+ ΔK/ΔV(H_ref)\n→ K norm + original RoPE', 'orange', 17)
    box(ax, 450, 425, 335, 145, 'R = SDPA(Q_face, K_ref, V_ref)\nsoftmax over selected\nreference-face keys only', 'teal', 18)
    box(ax, 883, 465, 310, 112, 'm · ΔO(R)\noutput LoRA', 'orange', 21)
    box(ax, 1280, 450, 275, 140, 'P(A) + m·ΔO(R)\n→ native modulation gate\n→ residual + native MLP', 'green', 17)
    arrow(ax, [(350, 435), (395, 435), (395, 471), (450, 471)], 'orange')
    arrow(ax, [(350, 650), (395, 650), (395, 534), (450, 534)], 'orange')
    arrow(ax, [(785, 485), (823, 485), (823, 320), (1025, 320), (1025, 262)], 'teal')
    arrow(ax, [(785, 530), (883, 530)], 'orange')
    arrow(ax, [(1193, 521), (1280, 521)], 'orange')
    arrow(ax, [(1417, 262), (1417, 450)])
    banner(ax, 763, 'm = 0 keeps native rows exactly at this seam; m = 1 fully routes to the reference read; 0 < m < 1 blends the two.', size=17)

    # 7: full math.
    fig, ax = new('Attention arithmetic and native block composition',
        'Native Q/K below are pre-normalization projection outputs; L_X(h) = (α/r) B_X A_X h, with α = r = 128.')
    mathline(ax, 65, 171, r'$Q_F = \mathrm{RoPE}_F(\mathrm{Norm}_Q(Q_F^{native}+L_Q(H_F)))$', 24)
    mathline(ax, 65, 235, r'$K_J = \mathrm{RoPE}_J(\mathrm{Norm}_K(K_J^{native}+L_K(H_J))),\quad V_J=V_J^{native}+L_V(H_J)$', 23)
    mathline(ax, 65, 311, r'$R=\mathrm{softmax}(Q_F K_J^T/\sqrt{128})V_J,\qquad A_F=(1-m_F)N_F+m_F R$', 24)
    mathline(ax, 65, 384, r'$A_{\mathrm{other}}=N_{\mathrm{other}},\qquad O_F=m_F L_O(R),\quad O_{\mathrm{other}}=0$', 24)
    card(ax, 55, 490, 720, 218, 'DOUBLE-STREAM BLOCK', 'Image update: h ← h + g_attn · [P_img(A) + O]. Then execute the native image MLP and its gate/residual. Text follows its native path.', size=19)
    card(ax, 810, 490, 735, 218, 'SINGLE-STREAM BLOCK', 'h ← h + g · [linear2(concat(A, native_MLP_activation)) + O]. The added output LoRA does not directly alter the MLP slice.', size=19)
    banner(ax, 757, 'The branch changes an internal attention message. The pretrained final velocity head remains active.', size=19)

    # 8: literal code from saved run.
    fig, ax = new('Code: adapt native projections, then apply norm and RoPE',
        'Verbatim executable lines from the completed run snapshot; line numbers refer to that snapshot.')
    excerpt(ax, 45, 156, 1085, 618, 'ba_dit/nn/masked_face_attention.py', 35, 52, 15)
    card(ax, 1160, 156, 390, 175, 'Q AND K', 'LoRA is added before the native per-head normalization and position rotation.', size=17)
    card(ax, 1160, 354, 390, 185, 'VALUES', 'V uses the native value slice plus its LoRA; values are not normalized or rotated here.', size=17)
    card(ax, 1160, 562, 390, 212, 'POSITION OWNERSHIP', 'pe.index_select uses original query and reference indices. Reference support stays inside the selected face.', size=17)

    # 9: explicit mask and target-only scatter.
    fig, ax = new('Code: fractional routing and exact zero-mask rows',
        'Attention replacement and output adaptation are separate, explicitly gated tensors.')
    excerpt(ax, 45, 165, 1510, 327, 'ba_dit/nn/masked_face_attention.py', 53, 62, 17)
    card(ax, 55, 535, 470, 235, 'QUERY SELECTION', 'Select target indices with m > 0. Only those query chunks perform the extra read; the result is scattered back into the native sequence.', size=18)
    card(ax, 560, 535, 470, 235, 'MASK VALUES', 'Fractional masks blend native and reference messages. torch.where restores exact native values where a batch row has zero mask.', size=18)
    card(ax, 1065, 535, 480, 235, 'GATE OWNERSHIP', 'The mask is supplied by the spatial protocol. γ is fixed at 1. There is no learned scalar router; native modulation gates remain frozen.', size=18)

    # 10: output-projection ordering is a critical invariant.
    fig, ax = new('Code: insert the output delta before the native gate',
        'The patched Toolkit model matches the recorded source commit and complete patch hash.')
    text(ax, 55, 159, 'Double-stream image path', 21, C['teal'], 'bold')
    excerpt(ax, 45, 206, 1510, 202, 'model.py', 494, 500, 17)
    text(ax, 55, 449, 'Single-stream attention + MLP path', 21, C['teal'], 'bold')
    excerpt(ax, 45, 496, 1510, 185, 'model.py', 382, 385, 18)
    banner(ax, 748, 'Native output projection, native gate and native residual each run once. ΔO(R) is added after projection, before the gate.', size=18)

    # 11: parameter inventory and initialization.
    fig, ax = new('What learns: 25,165,824 parameters in 64 tensors',
        'Eight sites × four projections × two low-rank matrices; native transformer, text encoder and VAE are frozen.')
    table(ax, 45, 161, [420, 340, 360, 390], ['Per projection', 'A shape', 'B shape', 'Parameter count'], [
        ('Q, K, V or output', '[128, 3072]', '[3072, 128]', '786,432'),
        ('Four per site', '4 A matrices', '4 B matrices', '3,145,728'),
        ('Eight sites', '32 A matrices', '32 B matrices', '25,165,824')
    ], row_h=75, size=17)
    excerpt(ax, 45, 481, 860, 295, 'ba_dit/nn/reference_read_delta.py', 10, 20, 14)
    card(ax, 940, 481, 610, 295, 'WHY STEP 0 DIFFERS FROM NATIVE', 'A is Kaiming-initialized and B is zero, so all LoRA deltas start at zero. But face routing is already enabled: its reference-only softmax differs from native joint attention. BA-off and a zero mask preserve native; BA-on step 0 does not.', size=19)

    # 12: measured mask illustration (no model generation).
    fig, ax = new('Two different masks, with explicit provenance',
        'Reference support chooses K/V; a target-space alpha chooses queries and controls the two-pass image protocol.')
    spec = importlib.util.spec_from_file_location('report_geometry', source('ba_dit/data/geometry.py'))
    geo = importlib.util.module_from_spec(spec); spec.loader.exec_module(geo)
    reference = ROOT / 'data/datasets/one_id/ref/51.jpg'
    used_files.add(reference)
    im, support, geometry = geo.reference_geometry(Image.open(reference), ROWS[0]['reference_face_bbox'], 'flux', 512)
    a = fig.add_axes([.036, .315, .28, .45]); a.imshow(im); a.axis('off')
    cells = np.repeat(np.repeat(support, 16, 0), 16, 1)
    overlay = np.zeros((*cells.shape, 4)); overlay[cells] = [1, .5, 0, .38]
    a.imshow(overlay)
    b = picture(fig, image_path('native','oneid_00'), 583, 210, 430, 410)
    x0,y0,x1,y1 = MASKS['samples']['oneid_00']['face_bbox']
    b.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=C['orange'],linewidth=2))
    y,x = np.mgrid[:768,:768]
    dist = np.maximum.reduce([x0-x, x-(x1-1), y0-y, y-(y1-1)])
    alpha = np.clip(1-dist/16,0,1)
    token = alpha.reshape(48,16,48,16).mean((1,3))
    c = fig.add_axes([.7,.315,.26,.45]); c.imshow(token,vmin=0,vmax=1,cmap='YlOrBr',interpolation='nearest');c.axis('off')
    text(ax, 268, 640, 'Reference: 88 selected / 1,024 tokens', 16, C['teal'], ha='center')
    text(ax, 797, 640, 'Frozen native image + reviewed face box', 16, C['teal'], ha='center')
    text(ax, 1330, 640, '48 × 48 query alpha; 16px feather', 16, C['teal'], ha='center')
    card(ax, 55, 698, 720, 126, 'TRAINING', 'Target-photo box, transformed exactly with target resize/crop, supplies face supervision.', size=16)
    card(ax, 810, 698, 735, 126, 'INFERENCE', 'Reviewed box from native generation. Target photographs are never loaded.', size=16)

    # 13: actual two-pass inference.
    fig, ax = new('Inference uses a frozen native scene at every denoising step',
        'First obtain the native image, clean latent and reviewed mask for the same prompt / reference / seed / sampler.')
    box(ax, 50, 180, 415, 103, 'Pass 1: native FLUX, BA off\nfreeze image, latent z_native and mask', 'green', 18)
    box(ax, 575, 180, 450, 103, 'Pass 2: initialize face state with noise ε\n20 Euler steps; CFG = 4', 'orange', 18)
    box(ax, 1135, 180, 410, 103, 'Decode then alpha-compose\nwith the original native pixels', 'green', 18)
    arrow(ax, [(465, 231), (575, 231)]); arrow(ax, [(1025, 231), (1135, 231)])
    mathline(ax, 90, 363, r'$b_t=(1-t)z_{native}+t\epsilon,\qquad z_t^{in}=(1-m)b_t+m z_t^{face}$', 27)
    mathline(ax, 90, 448, r'$v=v_{uncond}+4\,(v_{cond}-v_{uncond}),\qquad z_{t_{next}}^{face}=z_t^{face}+(t_{next}-t)v$', 25)
    mathline(ax, 90, 533, r'$I_{final}=\mathrm{round}[(1-a)I_{native}+a\,I_{decoded}]$', 27)
    para(ax, 90, 610, 'm is 16×16 average-pooled pixel alpha a. Both conditional and empty-prompt predictions run through the same patched branch. Pixel composition also removes VAE spill outside the mask support.', 120, 19)
    banner(ax, 756, 'Exact exterior preservation is imposed by latent mixing and pixel composition; it is not a learned background-locality result.', size=18)

    # 14: exact inference source.
    fig, ax = new('Code: fresh face noise, native exterior context and paired CFG',
        'The immutable one-ID controller asserts that validation rows contain no target photograph.')
    excerpt(ax, 45, 157, 1510, 437, 'scripts/online_face_ba.py', 178, 192, 15)
    excerpt(ax, 45, 612, 1510, 192, 'ba_dit/nn/masked_face_flow.py', 27, 30, 17)

    # 15: training objective.
    fig, ax = new('Training: fresh flow-matching inputs through the full denoiser',
        'Cached inputs are text embeddings and VAE latents only. Transformer activations and reference features are recomputed.')
    box(ax, 45, 163, 320, 102, '19 cross-view pairs\nclean target z₀ + reference', 'blue', 18)
    box(ax, 435, 163, 320, 102, 'Draw new ε and timestep\nfrom pinned native schedule', 'blue', 18)
    box(ax, 825, 163, 320, 102, 'Full patched transformer\nonly BA parameters learn', 'orange', 18)
    box(ax, 1215, 163, 340, 102, 'Face-weighted velocity MSE\nbackpropagate through FLUX', 'green', 18)
    for x in [365,755,1145]: arrow(ax, [(x,214),(x+70,214)])
    mathline(ax, 110, 302, r'$z_t=(1-t)z_0+t\epsilon,\quad v^*=\epsilon-z_0,\quad L=\frac{\sum_{p,c}m_p\,(v_{\theta,p,c}-v^*_{p,c})^2}{128\sum_p m_p}$', 22)
    excerpt(ax, 45, 370, 1510, 420, 'ba_dit/backends/flux_runtime.py', 92, 107, 14)
    text(ax, 65, 811, 'No ArcFace, perceptual or decoded-image auxiliary loss is used in this run.', 17, C['teal'], 'bold')

    # 16: resolved training controls.
    fig, ax = new('Resolved training recipe and execution contract',
        'The completed run stopped at 2,000 optimizer updates; all four scheduled panels were scored serially.')
    table(ax, 45, 154, [425,1085], ['Setting', 'Actual run'], [
        ('Inputs / order', '19 same-ID cross-view pairs; epoch shuffle seed 142; fresh noise each microbatch.'),
        ('Geometry / batch', '768px target, 512px reference; microbatch 1 × accumulation 4 = effective batch 4.'),
        ('Optimizer', 'AdamW, LR 5e−5, 100-update linear warmup, weight decay 0, global gradient clip 1.'),
        ('Precision / memory', 'Frozen BF16 denoiser; FP32 A/B parameters and Adam state; activation checkpointing.'),
        ('Validation schedule', 'Fixed24 at 0 / 500 / 1,000 / 2,000; 20 Euler steps, CFG 4, preserved native scene/masks.'),
        ('Save / resume', 'Checkpoints every 500: adapters, Adam, scheduler, CPU/CUDA/Python RNG and data cursor.'),
        ('Exposure / selection', '8,000 microbatches seen. Best ID_sim at update 1,000; final update 2,000 also retained.')
    ], row_h=79, size=17)

    # 17: metric definitions and fixed panel.
    fig, ax = new('Evaluation: named fixed24, same person and fixed references',
        'The original fixed96 benchmark remains a separate experiment. These 24 images are development evidence for one identity.')
    card(ax,45,159,730,211,'PANEL', '12 prompts × seeds 0 and 1, unchanged order. Scenes include subway, skiing, rain concert, gym, club, traffic, carnival and kitchen. Fixed reference 51.jpg. The identity is present in training.',size=19)
    card(ax,815,159,730,211,'THREE REQUIRED CONTROLS', 'Native BA-off; untrained routed BA at step 0; trained routed BA. All images use 768px, reference512, 20 steps, CFG4 and the same prompt/seed panel.',size=19)
    card(ax,45,407,730,235,'ID_SIM', 'InsightFace buffalo_l embedding cosine against the fixed identity embedding. Select the detected face by greatest overlap with the frozen native ownership box; IoU < 0.05 scores zero. Ambiguity margin: 0.02.',size=18)
    card(ax,815,407,730,235,'TEXT_SIM', 'CLIP ViT-L/14@336px image–prompt logits, averaged over 24 images. Higher is better. This is not cosine or a percentage. Face-quality metrics such as TOPIQ were not reported for this run.',size=18)
    banner(ax,733,'Missing-face, unowned-face, missing-mask and ambiguous fractions are zero at each scored BA checkpoint.',size=18)

    # 18: results table and charts from raw source values.
    fig, ax = new('Measured identity gain, with lower CLIP text alignment',
        'Best checkpoint is selected by mean ID_sim on this development panel; higher is better for both plotted metrics.')
    stages = ['0','500','1000','2000']; steps=[0,500,1000,2000]
    for x,key,title in [(100,'id_sim','Identity similarity'),(890,'text_sim','CLIP text score')]:
        a=charts_axis(fig,x,198,610,316)
        a.plot(steps,[M[k][key] for k in stages],'-o',color=C['green'],lw=3,ms=7,label='Routed BA')
        a.axhline(N[key],color=C['muted'],ls='--',lw=2,label='Native')
        a.set_title(title,fontsize=18,color=C['ink'],pad=14)
        a.set_xticks(steps); a.set_xlabel('Optimizer update',fontsize=13)
        a.legend(frameon=False,fontsize=12,loc='lower right' if key=='id_sim' else 'upper right')
    rows=[]
    for label,vals in [('Native',N),('BA 0',M['0']),('BA 500',M['500']),('BA 1,000 (best)',M['1000']),('BA 2,000',M['2000'])]:
        rows.append((label,f"{vals['id_sim']:.6f}",f"{vals['id_sim']-N['id_sim']:+.6f}",f"{vals['text_sim']:.6f}"))
    table(ax,80,590,[440,330,400,290],['Checkpoint','ID_sim','ID gain vs native','CLIP logits'],rows,row_h=36,size=14)

    # 19: paired gains do not hide failures.
    fig, ax = new('Paired results: 23 of 24 improve over native at update 1,000',
        'Computed directly from matched quality_per_image.csv rows. No independence or unseen-ID claim is attached to this count.')
    order=np.argsort(D)
    a=charts_axis(fig,120,187,1020,521)
    a.bar(np.arange(24),D[order],color=[C['green'] if d>0 else C['orange'] for d in D[order]])
    a.axhline(0,color=C['ink'],lw=1)
    a.set_xticks(np.arange(24));a.set_xticklabels([IDS[i].replace('oneid_','') for i in order],rotation=65,ha='right',fontsize=11)
    a.set_ylabel('BA 1,000 ID_sim − native ID_sim',fontsize=14)
    card(ax,1190,190,355,185,'MEAN CHANGE',f"{D.mean():+.6f}\nMedian {np.median(D):+.6f}",size=21)
    card(ax,1190,406,355,225,'ONE REGRESSION','03_s1: 0.526891 → 0.508440 (−0.018451). All 24 comparisons are shown in the appendix galleries.',size=18)
    para(ax,95,771,'At 2,000, mean ID falls by 0.011231 versus 1,000, although 14/24 individual images improve. One trajectory does not establish the cause of that change.',126,17)

    # 20: same full scenes across three checkpoints.
    fig,ax=new('Whole-scene comparison: native, best and final checkpoint',
        'Two fixed examples: prompt 00 (park bench) and prompt 06 (traffic), both seed 1. Exterior equality is imposed by composition.')
    for col,(stage,label) in enumerate([('native','Native'),('1000','BA 1,000'),('2000','BA 2,000')]):
        text(ax,510+col*395,153,label,19,C['teal'],'bold',ha='center')
        for row,key in enumerate(['oneid_00_s1','oneid_06_s1']):
            picture(fig,image_path(stage,key),350+col*395,197+row*318,310,278)
    for row,(key,title) in enumerate([('oneid_00_s1','Reading on a\npark bench'),('oneid_06_s1','Angry in a\ntraffic jam')]):
        text(ax,65,264+row*318,title,23,C['teal'],'bold')
        text(ax,65,349+row*318,key,15,C['muted'])
    text(ax,65,802,'Identity gains coexist with changes in facial detail, accessories and lighting; inspect crops at equal scale.',16,C['muted'])

    # 21: include best, limited and adverse cases with selection disclosed.
    fig,ax=new('Face crops: large gain, small gain and the native regression',
        'Selected by paired ID change at 1,000: largest gain (04_s1), smallest gain (07), only decline (03_s1).')
    for col,(stage,label) in enumerate([('native','Native'),('0','Untrained BA'),('1000','BA 1,000'),('2000','BA 2,000')]):
        text(ax,380+col*325,155,label,19,C['teal'],'bold',ha='center')
        for row,key in enumerate(['oneid_04_s1','oneid_07','oneid_03_s1']):
            picture(fig,image_path(stage,key),268+col*325,202+row*208,224,177,face_crop(key))
            text(ax,380+col*325,383+row*208,f"ID {SCORES[stage][key]['id_sim']:.4f}",13,C['muted'],ha='center')
    for row,key in enumerate(['oneid_04_s1','oneid_07','oneid_03_s1']):
        text(ax,55,243+row*208,key,17,C['teal'],'bold')
        val=SCORES['1000'][key]['id_sim']-SCORES['native'][key]['id_sim']
        text(ax,55,280+row*208,f'Δ ID {val:+.4f}',16,C['muted'])

    # 22/23: the entire fixed panel, with no reordering or selection.
    for seed in (0,1):
        fig,ax=new(f'Complete fixed24 gallery · seed {seed}',
            'All 12 prompts in manifest order; each pair is Native | BA 1,000. Same frozen crop coordinates; no per-image tuning.')
        for i,row in enumerate([r for r in ROWS if r['seed']==seed]):
            key=row['sample_id'];x=45+(i%4)*385;y=160+(i//4)*222
            rect(ax,x,y,368,211,'white')
            text(ax,x+12,y+9,key,13,C['teal'],'bold')
            text(ax,x+356,y+9,f"Δ {SCORES['1000'][key]['id_sim']-SCORES['native'][key]['id_sim']:+.3f}",13,C['green'],ha='right')
            for col,stage in enumerate(['native','1000']):
                picture(fig,image_path(stage,key),x+8+col*180,y+39,172,164,face_crop(key))

    # 24: actual measured implementation checks.
    fig,ax=new('Engineering evidence: native parity, learning and exact restart',
        'These checks support the implementation contract; image-quality claims come from the scored panel.')
    table(ax,45,158,[430,1080],['Check','Recorded outcome'],[
        ('Native / zero mask', 'Pretrained BA-off parity and zero-mask parity are exact, including after adapter updates.'),
        ('Trainable ownership', 'All 64 branch tensors update; first B gradients and second-step A/B gradients are finite/nonzero.'),
        ('Frozen weights', 'SHA256 over every frozen tensor is unchanged after the two real admission updates.'),
        ('Save / resume', 'Fresh-process replay is bit-exact for adapters, Adam moments, scheduler, RNG and sample cursor.'),
        ('Full-run health', 'All 2,000 logged losses and gradient norms are finite; 8,000 samples consumed.'),
        ('Training memory', 'Peak CUDA reserved 9.859 GiB = 61.65% of the 15.992 GiB laptop GPU; below the 90% ceiling.'),
        ('Inference / exterior', 'Final inference peak reserved 8.525 GiB; 24/24 final images have zero exterior pixel error.')
    ],row_h=79,size=17)

    # 25: conclusion and bounded interpretation.
    fig,ax=new('What the completed experiment establishes',
        'The measured result is successful one-person fitting with an explicit spatial protocol.')
    card(ax,45,163,725,256,'SUPPORTED BY THIS RUN','The branch learns through the full frozen denoiser. Best fixed24 ID_sim beats native by 0.0989 and improves 23/24 paired cases. Save/resume, branch gradients and parameter ownership are verified.',size=20)
    card(ax,815,163,730,256,'LIMITS OF THE EVIDENCE','Only one identity and one training trajectory. CLIP text alignment declines. Reference influence versus identity memorization is not isolated. Generated masks and native scenes are supplied at inference.',size=20)
    card(ax,45,459,725,282,'KEEP THE RIGHT BASELINES','Retain native BA-off, untrained routed BA and trained routed BA. Step 0 is already a strong intervention and a poor-quality baseline. Exact backgrounds reflect the composition rule.',size=20)
    card(ax,815,459,730,282,'NEXT CONTROLLED TESTS — PROPOSALS','Multi-ID training with identity-disjoint fixed96 validation; reference swaps and shuffled branch K/V controls; report paired identity, text alignment and face quality. These are not results from this one-ID run.',size=19)
    text(ax,65,794,'Best preserved checkpoint: update 1,000. Final checkpoint: update 2,000.',19,C['teal'],'bold')

    # 26: reproducibility and source ledger.
    fig,ax=new('Reproducibility: exact sources, weights, data and checkpoints',
        'Every code excerpt comes from the saved run snapshot or its hash-matched patched Toolkit checkout.')
    table(ax,45,154,[390,1120],['Artifact','Identity / location'],[
        ('Run','runs/flux4b_oneid_online_face_qkvo_r128_768_20261002'),
        ('Comet experiment',E['comet_key']+'  · project rsrch_new'),
        ('Toolkit source',E['model_identity']['source_commit']),
        ('FLUX Base4B weights',E['model_identity']['revisions']['flux4b']),
        ('Text encoder / VAE','Revisions recorded in source_audit.json and ONLINE_FACE_BA_RESULTS.json.'),
        ('Panel SHA256',E['validation_manifest_sha256']),
        ('Best adapter SHA256',E['checkpoint_adapters_sha256']['1000']),
        ('Source / evidence audit','source_audit.json: file hashes, snippets, metrics, paired scores, page index and visual artifact provenance.')
    ],row_h=61,size=15)
    link=text(ax,65,736,'Open the completed Comet experiment',19,C['teal'],'bold');link.set_url(E['comet_url'])
    text(ax,65,781,'Rebuild: envs/report/bin/python reports/261003_online_face_ba/build_report.py',16,C['muted'])


def export():
    OUT.mkdir(parents=True,exist_ok=True)
    checks=[]
    with PdfPages(PDF,metadata={'Title':'FLUX1 — FLUX Base 4B — Online masked Q/K/V/O branched attention: architecture and one-ID results',
                               'Author':'rsrch_new','Subject':'Completed single-identity experiment; 3 October 2026'}) as pdf:
        for i,(fig,title) in enumerate(s.pages,1):
            fig.canvas.draw()
            for artist,(x,y,w,h) in s.fitted_texts:
                if artist.figure is not fig: continue
                for _ in range(3):
                    bb=artist.get_window_extent(fig.canvas.get_renderer())
                    a,b=artist.axes.transData.transform([[x,y],[x+w,y+h]])
                    ratio=min(abs(b[0]-a[0])/max(bb.width,1),abs(b[1]-a[1])/max(bb.height,1),1)
                    if ratio>=1:break
                    artist.set_fontsize(artist.get_fontsize()*ratio*.985)
            fig.canvas.draw()
            for artist in s.texts:
                if artist.figure is not fig:continue
                bb=artist.get_window_extent(fig.canvas.get_renderer())
                assert bb.x0>=-1 and bb.y0>=-1 and bb.x1<=fig.bbox.width+1 and bb.y1<=fig.bbox.height+1,(i,artist.get_text())
            checks.append({'page':i,'title':title,'page_text_bounds':'pass'})
            pdf.savefig(fig)
            if i in (3,6,13):
                name={3:'whole_model',6:'branched_attention',13:'inference_protocol'}[i]
                svg = OUT/f'FLUX1_{name}.svg'
                fig.savefig(svg)
                svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
            plt.close(fig)
    doc=pymupdf.open(PDF)
    doc.set_toc([[1,title,i] for i,(_,title) in enumerate(s.pages,1)]);doc.saveIncr()
    for i,p in enumerate(doc,1):
        p.get_pixmap(matrix=pymupdf.Matrix(100/72,100/72)).save(OUT/f'page_{i:02d}.png')
    for number,name in [(3,'whole_model'),(6,'branched_attention'),(13,'inference_protocol')]:
        d=pymupdf.open();d.insert_pdf(doc,from_page=number-1,to_page=number-1);d.save(OUT/f'FLUX1_{name}.pdf',garbage=4,deflate=True);d.close()
    assert len(doc)==26
    extracted='\n'.join(p.get_text() for p in doc)
    for required in ['25,165,824','0.430254','0.331399','23 of 24','BA 2,000','source_audit.json']:
        assert required in extracted,required
    (OUT/'extracted_text.txt').write_text(extracted)
    doc.close()
    audit={'report_date':'2026-10-03','run':str(RUN.relative_to(ROOT)),'completed_at_utc':E['completed_at_utc'],
           'comet_url':E['comet_url'],'configuration':E['configuration'],'model_identity':E['model_identity'],
           'native_metrics':N,'ba_metrics':M,'best_checkpoint':E['checkpoint_adapters_sha256']['1000'],
           'paired_id_gains_vs_native':{k:float(d) for k,d in zip(IDS,D)},
           'positive_pairs':int((D>0).sum()),'mean_gain':float(D.mean()),'median_gain':float(np.median(D)),
           'source_snapshot_hashes_verified':E['source_sha256'],
           'artifact_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(used_files)},
           'report_generator_sha256':sha(Path(__file__)),'report_style_sha256':sha(OUT/'report_style.py'),
           'pdf_sha256':sha(PDF),'checks':checks,'scope':E['scope'],
           'experiment_family':'FLUX1',
           'note':'Current working tree contains later edits. Run snapshot, not live HEAD alone, defines the reported implementation.'}
    (OUT/'source_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (OUT/'validation_report.json').write_text(json.dumps({'pages':26,'source_hashes_verified':20,'text_bounds_pass':True,
        'metrics_match_raw_csv':True,'native_images_match_frozen_hashes':True,'checkpoint_hashes_match':True,
        'no_inference_or_training_performed':True,'pdf_sha256':sha(PDF)},indent=2)+'\n')
    for start in (0,12,24):
        count=min(12,len(s.pages)-start);sheet=Image.new('RGB',(1280,180*((count+3)//4)),'#ccd7dc')
        for i in range(count):
            im=Image.open(OUT/f'page_{start+i+1:02d}.png');im.thumbnail((320,180));sheet.paste(im,((i%4)*320,(i//4)*180))
        sheet.save(OUT/f'contact_sheet_{start//12+1}.png')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>FLUX1 — architecture and one-ID results</title>'
        '<style>body{background:#dce5e9;margin:24px;font:18px sans-serif}img{display:block;max-width:1400px;width:100%;margin:20px auto}a{color:#006982}</style>'
        '<a href="'+PDF.name+'">Download the 26-page PDF</a>'+''.join(f'<img src="page_{i:02d}.png" alt="Slide {i}">' for i in range(1,27)))
    (OUT/'rebuild.txt').write_text('FLUX1 / Base4B online masked Q/K/V/O — completed one-ID architecture and results\n'
        'Rebuild from the project root:\n  envs/report/bin/python reports/261003_online_face_ba/build_report.py\n\n'
        'Uses the existing report environment and immutable local run artifacts. No GPU or network calls.\n'
        'Sources: saved source_snapshot, recorded patched Toolkit, native/BA images and per-image scores.\n'
        'Includes searchable code, 26 slides, PDF bookmarks, 3 standalone vector schemes and source audit.\n'
        'The original fixed96 benchmark and current multi-ID deployment are outside this report.\n')
    print(json.dumps({'pdf':str(PDF),'pages':len(s.pages),'bytes':PDF.stat().st_size,'paired_gains':int((D>0).sum())}))


if __name__=='__main__':
    build()
    export()
