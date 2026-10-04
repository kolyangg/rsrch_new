"""Final measured deck: envs/report/bin/python reports/261001_deep_identity/build_report.py."""
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / '261001_reference_refiner'))
import report_style as s
from report_style import plt, PdfPages, page, text, para, box, arrow, card, banner, snippet, image_panel, table, C

RUN = ROOT / 'runs/flux4b_deep_identity1024_det_20261001'
PARENT = ROOT / 'runs/flux4b_reference_refiner1024_b128_20261001'
FLOW = ROOT / 'ba_dit/nn/deep_identity_flow.py'
ALIGN = ROOT / 'ba_dit/nn/deterministic_identity_loss.py'
PRECEDENT = Path('/tmp/codex-remote-attachments/01a0f18d-112a-7520-8ae2-cd6285ecf1e5/d225a994-d286-47e7-977a-c5315ed32378/2-branched_attention_short_combined_E13_CL14_Hardcase_CL19_CL23_CL39.pdf')
COMET = 'https://www.comet.com/nikolay-2104/rsrch-new/75e5d8961bdf40efab63ffd2770711ca'
READ_BYTES = {}


def read(path):
    READ_BYTES[path] = path.read_bytes()
    return json.loads(READ_BYTES[path])


def per_image(step):
    path = RUN / f'validation-{step:06d}/quality_per_image.csv'
    READ_BYTES[path] = path.read_bytes()
    return {r['sample_id']: float(r['id_sim']) for r in csv.DictReader(READ_BYTES[path].decode().splitlines())}


def main():
    identity = read(RUN / 'identity.json')
    status = read(RUN / 'status.json')
    decision = read(RUN / 'convergence.json')
    assert status['status'] == 'completed' and decision['plateau'], 'Final report requires a completed scored plateau'
    scores = {int(p.parent.name.split('-')[1]): read(p)['metrics'] for p in sorted(RUN.glob('validation-*/quality_summary.json'))}
    latest, best = max(scores), decision['best_step']
    summary = read(RUN / 'training_summary.json')
    training_audit = read(RUN / 'final_training_audit.json')
    review = read(RUN / 'comparison_summary.json')
    assert review['latest_step'] == latest and summary['steps'] == training_audit['steps'] == latest
    best_record = read(RUN / 'best_checkpoint.json')
    assert hashlib.sha256(Path(best_record['file']).read_bytes()).hexdigest() == best_record['sha256']
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in identity['source_sha256'].items())
    probes = read(RUN / 'probes.json')
    admission = read(RUN / 'admission.json')
    resume = read(RUN / 'resume_check.json')
    replay = read(RUN / 'deterministic_replay.json')
    audits = [read(p) for p in sorted(RUN.glob('inference_audit_*.json'))]
    metrics_path = RUN / 'metrics.jsonl'
    READ_BYTES[metrics_path] = metrics_path.read_bytes()
    logs = [json.loads(line) for line in READ_BYTES[metrics_path].decode().splitlines()]
    peak = max(r['hardware/peak_reserved_gib'] for r in logs)
    mean_seconds = np.mean([r['train/seconds'] for r in logs if r['step'] > 500])
    before, at_best, final = per_image(0), per_image(best), per_image(latest)
    improved = sum(at_best[k] > before[k] for k in before)
    worse = sum(final[k] < at_best[k] for k in before)
    cutoff = datetime.now(timezone.utc).isoformat(timespec='seconds')
    source = 'Evidence: immutable run sources, scored fixed 24 panels and checkpoint audits; full hashes in source_audit.json.'
    evidence = f'Evidence captured {cutoff} · deterministic continuation · completed development stopping rule'

    fig, ax = page('Three-read BA with direct face identity supervision',
                   'FLUX.2-klein Base 4B · one identity · local 16 GB GPU · 1 October 2026', source)
    card(ax, 45, 155, 485, 222, 'BEST GENERATED ID SCORE',
         f"{scores[best]['id_sim']:.5f} at +{best:,} updates. Initial {scores[0]['id_sim']:.5f}; {improved}/24 images improve. Native reference-conditioned baseline: .33140.", size=20)
    card(ax, 558, 155, 485, 222, 'TRAINABLE ARCHITECTURE',
         '30.83M BA parameters, width 1024, 16 heads and three reference reads. Frozen 512-wide BA core plus fully frozen native FLUX.', accent='orange', size=20)
    card(ax, 1070, 155, 485, 222, 'FINAL RESULT',
         f"Stopped after +{latest:,}: ID {scores[latest]['id_sim']:.5f}. Two checks without a .003 best-score gain. The earlier best checkpoint is preserved.", size=20)
    box(ax, 65, 442, 400, 110, 'Frozen native features', fill='green', size=25)
    box(ax, 590, 415, 430, 164, 'Frozen BA core\n+ three trainable reference reads', fill='orange', size=23)
    box(ax, 1140, 442, 395, 110, 'BA face velocity\ninside native scene mask', fill='teal', size=22)
    arrow(ax, [(465, 497), (590, 497)]); arrow(ax, [(1020, 497), (1140, 497)])
    card(ax, 65, 642, 710, 143, 'ESTABLISHED', 'BA updates change generated faces and give an early ID gain. Background pixels remain exact after compositing.', size=19)
    card(ax, 805, 642, 730, 143, 'LIMIT', 'Later training regresses. Visible facial artifacts remain, and the best BA score is below native. This panel does not establish generalization.', size=19)

    fig, ax = page('Whole model: native background and a new BA face trajectory',
                   'Green: frozen weights · orange: trainable BA · blue: saved inputs and spatial routing', source)
    box(ax, 45, 163, 285, 105, 'Prompt + ID reference', fill='blue')
    box(ax, 390, 145, 490, 142, 'Frozen text encoder + VAE\nNative FLUX generation · CFG 4', fill='green', size=21)
    box(ax, 1020, 145, 530, 142, 'Native scene + detected face mask\nSaved once for all checkpoints', fill='blue', size=21)
    arrow(ax, [(330, 216), (390, 216)]); arrow(ax, [(880, 216), (1020, 216)])
    box(ax, 45, 360, 285, 115, 'Seeded face noise\n+ native background at sigma', fill='blue')
    box(ax, 400, 340, 490, 148, 'Frozen FLUX: 5 dual + 20 single blocks\nFinal attention hook captures\ntarget Q and reference-face K/V', fill='green', size=20)
    arrow(ax, [(330, 417), (400, 417)])
    arrow(ax, [(1285, 287), (1285, 313), (187, 313), (187, 360)], color='teal')
    box(ax, 1020, 340, 530, 148, 'Frozen 512 BA core\n+ three-read 1024 BA refiner\n128-channel face velocity', fill='orange', size=22)
    arrow(ax, [(890, 417), (1020, 417)])
    box(ax, 1055, 577, 460, 104, '20-step Euler face trajectory\nFace CFG 1', fill='teal', size=22)
    arrow(ax, [(1285, 488), (1285, 577)])
    arrow(ax, [(1055, 629), (951, 629), (951, 525), (187, 525), (187, 475)], color='teal', dashed=True)
    box(ax, 395, 577, 480, 104, 'Blend final latent → frozen VAE\nComposite native exterior pixels', fill='green', size=21)
    arrow(ax, [(1055, 629), (875, 629)])
    banner(ax, 753, 'New prompted images from ID reference and noise. No target photograph is loaded during validation.', size=19)

    fig, ax = page('Connection point: final single-stream Q / K / V',
                   'The same patched feature seam is used to build the training cache and to generate validation images.', source)
    box(ax, 50, 163, 370, 112, 'Native sequence\n[text | target | reference]', fill='green', size=22)
    box(ax, 510, 163, 450, 112, 'Native modulation + QKV projection\nQ/K normalization and RoPE', fill='green', size=22)
    box(ax, 1060, 151, 490, 138, 'Target Q + reference-face K/V\n24 × 128 → 3072 features\nDetach from the backbone graph', fill='blue', size=21)
    arrow(ax, [(420, 219), (510, 219)]); arrow(ax, [(960, 219), (1060, 219)])
    snippet(ax, 50, 325, 880, 304, ROOT / 'ba_dit/nn/face_crop_flow.py', 55, 65, size=17)
    card(ax, 970, 326, 580, 301, 'EXPLICIT OWNERSHIP', 'Target indices exclude text/reference tokens. K/V use the ID-reference face mask. Native Q/K rotation is already applied: the BA projections do not add a second RoPE. All target queries are evaluated before spatial blending.', size=19)
    card(ax, 50, 662, 730, 135, 'BRANCH ON', 'The capture exception exits before the native final projections. Face velocity is supplied by BA; native face velocity is not added.', size=18)
    card(ax, 815, 662, 735, 135, 'BRANCH OFF', 'The hook is bypassed. Actual native outputs before and after installation match bit-for-bit.', size=19)

    fig, ax = page('Branched attention: three trainable reads and a frozen core',
                   'Three reads refers to the trained refiner; the frozen 512 core has its own reference read.', source)
    box(ax, 45, 157, 330, 113, 'Captured target Q\nReference-face K / V', fill='green', size=22, badge='A')
    box(ax, 480, 149, 585, 130, 'Frozen 512-wide conditioned BA core\n10k checkpoint · 5.33M parameters\noutputs base face velocity', fill='green', size=21, badge='B')
    arrow(ax, [(375, 212), (480, 212)])
    box(ax, 45, 353, 330, 135, 'Trainable Q / K / V\n3072 → 1024\n16 heads', fill='orange', size=22, badge='C')
    arrow(ax, [(210, 270), (210, 353)], color='orange')
    box(ax, 470, 331, 570, 171, 'Read 1: bounded cosine attention\nPooled reference V → identity modulation\nContext = Q + sigma + noisy latent\nNonlinear feature mix', fill='orange', size=21, badge='D')
    arrow(ax, [(375, 420), (470, 420)], color='orange')
    box(ax, 475, 561, 265, 121, 'Read 2\nReference attention\n+ residual MLP', fill='orange', size=19, badge='E')
    box(ax, 810, 561, 265, 121, 'Read 3\nReference attention\n+ residual MLP', fill='orange', size=19, badge='F')
    arrow(ax, [(603, 502), (603, 561)], color='orange'); arrow(ax, [(740, 622), (810, 622)], color='orange')
    box(ax, 1180, 563, 370, 119, 'Output projection\n+ reference-gated noise path\nΔ face velocity', fill='orange', size=20, badge='G')
    arrow(ax, [(1075, 622), (1180, 622)], color='orange')
    box(ax, 1235, 330, 290, 152, 'ADD\nfrozen base + Δ\n= BA face velocity', fill='teal', size=22, badge='H')
    arrow(ax, [(1065, 213), (1380, 213), (1380, 330)])
    arrow(ax, [(1365, 563), (1365, 482)], color='orange')
    card(ax, 45, 560, 340, 124, 'REFERENCE MEMORY', 'Reads 2/3 reuse projected K/V from C, with their own learned projections.', size=16)
    banner(ax, 754, 'At step zero, the added reads 2/3 are exact identity residuals. The trained parent prediction is preserved.', size=18)

    fig, ax = page('First read: reference attention and identity-conditioned context',
                   'This trained single-read refiner is inherited from the parent; the new experiment continues to update it.', source)
    snippet(ax, 45, 151, 960, 646, FLOW, 59, 72, size=18)
    card(ax, 1045, 157, 505, 183, 'ATTENTION', 'Target Q attends to face-only reference K/V. Per-head cosine logits use a learned bounded temperature; the read is normalized.', size=19)
    card(ax, 1045, 373, 505, 196, 'CONTEXT', 'Current target query, diffusion sigma and noisy latent form the local context. Mean reference V predicts bounded gain and shift.', size=19)
    card(ax, 1045, 601, 505, 195, 'COMBINATION', 'Multiply the reference read by activated context, add pooled identity, then apply the nonlinear mix. Reads 2/3 further refine these features.', size=19)

    fig, ax = page('Two extra reference reads — implementation',
                   'Each target query is independent; sampled-token training matches full-scene head evaluation.', source)
    snippet(ax, 45, 151, 930, 376, FLOW, 30, 37, size=18)
    snippet(ax, 45, 555, 930, 249, FLOW, 71, 77, size=18)
    card(ax, 1015, 158, 535, 187, 'BOUNDED COSINE ATTENTION', 'Normalize each head’s Q/K. The learned temperature starts at 8 and stays within [1,16]. Each new read has its own Q/K/V and output.', size=19)
    card(ax, 1015, 373, 535, 198, 'INITIALIZATION', 'Only the new read output matrices and final MLP projections start at zero. Parent refiner weights are already trained. Initial generated images match the parent exactly.', size=19)
    card(ax, 1015, 598, 535, 207, 'ABLATION SCOPE', 'reference_read=False returns the frozen core. It disables the new trainable refiner; the core and native captured features still contain reference conditioning.', size=19)

    fig, ax = page('Training objective: flow accuracy plus decoded face identity',
                   'The identity path uses only training-photo landmarks and the fixed training reference.', source)
    box(ax, 45, 164, 290, 110, 'BA face velocity v\nfrom cached features', fill='orange', size=22)
    box(ax, 420, 151, 460, 139, 'x₀ estimate = noisy − sigma × v\nContiguous training latent crop\n48 px surrounding context', fill='blue', size=21)
    box(ax, 985, 164, 270, 110, 'Frozen FLUX2 VAE\ndecode RGB', fill='green', size=20)
    box(ax, 1320, 154, 235, 132, 'Bilinear aligned\n112 px face', fill='blue', size=21)
    arrow(ax, [(335, 219), (420, 219)]); arrow(ax, [(880, 219), (985, 219)]); arrow(ax, [(1255, 219), (1320, 219)])
    box(ax, 1040, 355, 500, 132, 'Frozen ArcFace → identity embedding\nL_ID = 1 − cosine(embedding, reference)', fill='green', size=21)
    arrow(ax, [(1435, 286), (1435, 355)])
    snippet(ax, 45, 347, 935, 249, ALIGN, 26, 32, size=17)
    card(ax, 45, 641, 935, 153, 'OPTIMIZER LOSS', 'Flow MSE on batch 256 every update, plus 0.5 × L_ID every second update using one full training-face region. Backprop reaches BA through frozen VAE and ArcFace.', size=19)
    card(ax, 1020, 529, 530, 265, 'INTERPRETATION LIMITS', 'ArcFace uses the same recognition family as ID_sim. Crop decoding has different context from full-scene decoding. Lower auxiliary loss alone does not establish better prompted faces.', accent='orange', size=19)

    fig, ax = page('Deterministic recovery preserves the trained state',
                   'Original failure and evidence remain unchanged; the continuation has its own run identity and Comet key.', source)
    card(ax, 45, 151, 730, 203, 'OBSERVED FAILURE', 'After the first 500-step panel, resumed step 502 differed by 4.37e−7 in a parameter near zero. Step 501 was exact. CUDA grid_sample backward is nondeterministic; the prior trainer had not enforced its configured determinism.', size=19)
    card(ax, 815, 151, 735, 203, 'EVIDENCE-BASED REPAIR', 'Equivalent four-neighbor bilinear gathers replace grid_sample. Deterministic algorithms, cuDNN and cuBLAS workspace are enforced. Both BA weights and Adam state resume from 500.', size=19)
    snippet(ax, 45, 390, 880, 387, ALIGN, 11, 23, size=17)
    card(ax, 965, 390, 585, 175, 'NUMERICAL CHECKS', 'CUDA crop forward error 2.38e−7 versus grid_sample; repeated input gradients exact. Two fresh-process 501/502 replays match bit-for-bit, including the identity update.', size=18)
    card(ax, 965, 594, 585, 183, 'PROVENANCE', f'Historical 0/500 images are imported unchanged. Later panels are newly generated. Last production resume check: step {resume["step"]}, max parameter error {resume["max_abs"]:.1f}.', size=19)

    fig, ax = page('Parameter ownership and exact experiment settings',
                   'This is a standalone BA flow head; its width is not a native-attention LoRA rank.', source)
    table(ax, 45, 156, [410, 315, 785], ['Component', 'Parameters / setting', 'Role'], [
        ('Native FLUX Base 4B', '3072 width · 24 heads', 'Frozen features and native reference/text conditioning. No native-attention LoRA trains.'),
        ('Conditioned BA core', '5,327,360 frozen', 'Retained 10k checkpoint; 11 tensors remain bit-exact.'),
        ('Three-read BA refiner', '30,831,747 trainable', '35 updated tensors: initial read/context/output plus two cross-attention/MLP reads.'),
        ('Frozen decoder + recognizer', 'FLUX2 VAE + ArcFace', 'Their operations transmit identity gradients to BA; their weights receive no optimizer updates.'),
        ('Optimizer', 'AdamW · batch 256', 'LR 5e−5; 100-step warmup; weight decay .01; gradient clip 1. Core and backbone excluded.'),
        ('Validation', '24 images · 768×768', '12 prompts × 2 seeds; one ID; 20 Euler steps; native CFG 4; face CFG 1; unchanged masks.'),
    ], row_h=85, size=16)
    banner(ax, 763, 'Original deep run resets Adam; deterministic continuation preserves its step 500 weights and optimizer state.', size=18)

    results_pages(identity, status, decision, scores, latest, best, summary, review, probes, admission,
                  logs, peak, mean_seconds, before, at_best, final, worse, audits, evidence, source)
    finish(identity, status, scores, decision, cutoff)


def results_pages(identity, status, decision, scores, latest, best, summary, review, probes, admission,
                  logs, peak, mean_seconds, before, at_best, final, worse, audits, evidence, source):
    fig, ax = page('Training data and measured batch scaling',
                   'Cached BA optimization excludes frozen-backbone cache creation and image validation.', source)
    box(ax, 45, 158, 415, 136, '19 training photos · one fixed ID\nFlow: 1140 fit + 114 noise probes\n64 sampled face tokens per case', fill='blue', size=21)
    box(ax, 575, 158, 435, 136, 'Identity: 76 full-face cases\n19 photos × sigmas .2/.4/.6/.8\nFrozen VAE + ArcFace backward', fill='green', size=21)
    box(ax, 1120, 158, 430, 136, 'Batch 256 flow every update\n1 identity case every 2 updates\nOnly BA optimizer parameters', fill='orange', size=21)
    arrow(ax, [(460, 226), (575, 226)]); arrow(ax, [(1010, 226), (1120, 226)])
    chart = fig.add_axes([.075, .225, .44, .365])
    bm = admission['benchmarks']
    chart.plot([r['batch_size'] for r in bm], [r['flow_cases_per_second'] for r in bm], 'o-', color=C['orange'])
    chart.set(xlabel='Flow cases per update', ylabel='Cached flow cases / second', xticks=[64, 128, 256])
    chart.grid(alpha=.2)
    card(ax, 925, 342, 625, 184, 'ADMISSION BENCHMARK', 'Batch 64/128/256: 314/506/732 flow cases/s, with the largest identity crop every second update. Batch 256 peaks at 11.06 GiB. This is the initial nondeterministic admission benchmark.', size=19)
    card(ax, 925, 553, 625, 209, 'ACTUAL CONTINUATION', f'Mean update time 501–{latest}: {mean_seconds:.3f}s; about {256/mean_seconds:.0f} cached flow cases/s. All-history peak reserved {peak:.3f} GiB. Sampled active GPU utilization 96–100%; not an end-to-end FLUX throughput claim.', size=19)
    para(ax, 85, 750, 'Finite cached cases are reused. Noise probes share training photographs; no held-out identity evaluation is claimed.', 72, 16)

    fig, ax = page('Spatial ownership: native scene outside, BA inside',
                   'The user explicitly requested masks detected from native generated images for this named experiment.', source)
    image_panel(fig, RUN / 'mask_overlays.png', 45, 151, 860, 633)
    card(ax, 940, 153, 610, 206, 'MASK ALIGNMENT', 'All 24 masks are tied to native image hashes. Full face-box weight with 16 px outer feather. Pixel masks are average-pooled into 48×48 packed latent grids at 768 px resolution.', size=20)
    snippet(ax, 940, 389, 610, 176, ROOT / 'ba_dit/nn/masked_face_flow.py', 27, 30, size=16)
    card(ax, 940, 594, 610, 190, 'EXACT EXTERIOR IS ENFORCED', f"All {review['exact_background_images']} reviewed composites preserve exterior pixels exactly. The raw VAE output may affect neighboring pixels; final compositing restores the native exterior. Raw outputs remain saved.", size=19)

    fig, ax = page('CL14 and CL39: shared reference attention, different placement',
                   'Comparison grounded in the supplied precedent report, pages 2/5/7/10/14.', source)
    table(ax, 45, 151, [275, 555, 680], ['Topic', 'Earlier PhotoMaker / SDXL', 'Current local FLUX diagnostic'], [
        ('CL14 face ownership', 'Hard query split and reference-face replacement in self-attention; native cross-attention remains active.', 'Same face/background ownership intent. A separate two-pass face-velocity generator replaces the face region.'),
        ('CL39 native anchor', 'Output-projected native message N plus a routed reference correction derived from R−N.', 'Frozen learned BA core plus trainable correction. Native face velocity is absent; native features still carry conditioning.'),
        ('CL39 router', 'Soft spatial routing, temporal low/high frequency gains and detached entropy confidence.', 'Saved generated-native mask, latent blending and pixel compositing. No CL39 frequency or confidence router.'),
        ('Integration depth', 'Branched attention processors inside U-Net self-attention layers.', 'One final FLUX feature hook; three extra BA reads outside native blocks. Native attention remains frozen.'),
        ('Training objective', 'Primary face diffusion objective; lineage variants add auxiliary objectives.', 'Sampled-token native flow MSE plus decoded ArcFace identity loss through frozen VAE.'),
        ('Parameter meaning', 'Ranked branch Q/K/V inside the attention processor.', '1024 is the standalone BA hidden width, not native LoRA rank. Only BA refiner weights train.'),
    ], row_h=92, size=16)
    banner(ax, 774, 'This is a CL14-inspired face-generation diagnostic with reference attention, not an exact CL39 port.', size=18)

    fig, ax = page('Generated identity scores: an early gain, then regression', evidence, source)
    old = {int(p.parent.name.split('-')[1]): read(p)['metrics'] for p in sorted(PARENT.glob('validation-*/quality_summary.json'))}
    chart = fig.add_axes([.075, .28, .50, .52])
    chart.plot([x/1000 for x in old], [old[x]['id_sim'] for x in old], 'o-', color=C['muted'], label='Previous single-read refiner')
    chart.plot([x/1000 for x in scores], [scores[x]['id_sim'] for x in scores], 'o-', color=C['orange'], label='Three reads + identity objective')
    chart.axhline(.3313986754, ls='--', color=C['teal'], label='Native generated images')
    chart.scatter([best/1000], [scores[best]['id_sim']], s=115, facecolors='none', edgecolors=C['green'], linewidths=2)
    chart.set(xlabel='Additional updates within each named experiment (thousands)', ylabel='Mean generated-face ID similarity')
    chart.grid(alpha=.2); chart.legend(fontsize=10)
    rows = [(f'{step:,}' + (' (best)' if step == best else ''), f"{scores[step]['id_sim']:.6f}", f"{scores[step]['text_sim']:.4f}") for step in scores]
    table(ax, 995, 157, [185, 180, 185], ['Update', 'ID_sim ↑', 'CLIP ↑'], rows, row_h=80, size=17)
    card(ax, 995, 555, 555, 220, 'READING THE CURVE', f"Best gain over initial: {scores[best]['id_sim']-scores[0]['id_sim']:+.5f}. Final falls {scores[best]['id_sim']-scores[latest]['id_sim']:.5f} below best; {worse}/24 images worsen. Native remains higher.", size=20)
    para(ax, 80, 755, 'Depth, identity loss, batch and LR changed together. The experiment supports their combined early gain, not a causal attribution to depth alone.', 78, 16)

    fig, ax = page('Lower cached losses do not guarantee better generated faces',
                   'Training photographs and finite cached noise cases differ from the prompted generation trajectory.', source)
    left = fig.add_axes([.075, .31, .40, .47])
    for split, color in [('fit', C['orange']), ('probe', C['teal'])]:
        left.plot([p['step']/1000 for p in probes], [p[f'probe/{split}/mse'] for p in probes], 'o-', color=color, label=split)
    left.set(xlabel='Additional BA updates (thousands)', ylabel='Flow MSE'); left.grid(alpha=.2); left.legend()
    right = fig.add_axes([.565, .31, .37, .47])
    active = [r for r in logs if r['step'] > 500 and r['train/identity_active']]
    cycles = [active[i:i+76] for i in range(0, len(active)-75, 76)]
    right.plot([np.mean([r['step'] for r in group])/1000 for group in cycles],
               [np.mean([r['train/identity_loss'] for r in group]) for group in cycles], 'o-', color=C['purple'])
    right.set(xlabel='Additional BA updates (thousands)', ylabel='Auxiliary identity loss', title='Complete 76-case cycles'); right.grid(alpha=.2)
    card(ax, 60, 667, 718, 142, 'MEASURED', f"Flow fit/probe MSE: {probes[0]['probe/fit/mse']:.4f}/{probes[0]['probe/probe/mse']:.4f} initially → {probes[-1]['probe/fit/mse']:.4f}/{probes[-1]['probe/probe/mse']:.4f} finally. Identity loss also falls.", size=18)
    card(ax, 817, 667, 733, 142, 'INTERPRETATION', 'The loss-to-generation gap remains. Finite caches, crop context and approximate one-step clean reconstruction are candidate causes; these results do not isolate one.', size=18)

    fig, ax = page('All 24 cases: paired identity changes',
                   'Fixed manifest order; same prompts, seeds, reference, mask and scoring definitions at every checkpoint.', source)
    names = list(before)
    chart = fig.add_axes([.07, .28, .86, .50])
    index = np.arange(len(names))
    chart.bar(index-.2, [at_best[n]-before[n] for n in names], .4, color=C['green'], label=f'Best +{best} minus initial')
    chart.bar(index+.2, [final[n]-at_best[n] for n in names], .4, color=C['orange'], label=f'Final +{latest} minus best')
    chart.axhline(0, color=C['muted'], lw=1)
    chart.set_xticks(index, [n.replace('oneid_', '') for n in names], rotation=45, ha='right', fontsize=9)
    chart.set(ylabel='Change in ID similarity', xlabel='Panel sample ID'); chart.grid(axis='y', alpha=.2); chart.legend(fontsize=11)
    banner(ax, 751, f"Best improves {sum(at_best[k]>before[k] for k in names)}/24 over initial; final improves {sum(final[k]>at_best[k] for k in names)}/24 over best. Paired images follow in fixed order.", size=18)

    fig, ax = page('Measured invariants and retained evidence',
                   'Checks cover actual checkpoints and full-backbone inference, not just a synthetic module test.', source)
    table(ax, 45, 153, [410, 1100], ['Check', 'Final evidence'], [
        ('Finite training / updates', f"All {len(logs):,} sequential loss/gradient records finite; all 35 trainable tensors changed."),
        ('Frozen weights / BA-off', 'All 11 core tensors exact; frozen native backbone excluded from optimizer. Actual native BA-off output is bit-exact.'),
        ('Cache vs live backbone', f'{len(audits)} scored panels with exact cached/live BA prediction checks on real pretrained features.'),
        ('Process resume', '501/502 replay in two fresh processes is exact; production 2001/2002 exact, including identity loss.'),
        ('Memory below90%', f'Training peak {peak:.3f} GiB (77.7%); inference {max(a["hardware/peak_reserved_gib"] for a in audits):.3f} GiB.'),
        ('Checkpoint provenance', '23 immutable source hashes; strict config/cache/mask identities; final checkpoint hashes recorded in every generated sample.'),
        ('Exterior / ownership', f'{review["exact_background_images"]} composites preserve native exterior exactly; original ID_sim and CLIP definitions unchanged.'),
    ], row_h=79, size=17)
    banner(ax, 779, 'Best checkpoint and optimizer state are retained. Old failed and regressed experiments are preserved without restart.', size=18)

    fig, ax = page('Stopping decision, limitations and reproducible artifacts', evidence, source)
    card(ax, 45, 153, 730, 212, 'DEVELOPMENT STOPPING RULE', f'At {latest:,}, two consecutive best-score gains are below .003. This is an operational plateau of the best score, with regression in later checkpoints; it is not monotonic or statistical convergence.', size=20)
    card(ax, 815, 153, 735, 212, 'PRESERVED RESULT', f"Best +{best}: ID {scores[best]['id_sim']:.6f}, CLIP {scores[best]['text_sim']:.4f}. Current continuation checkpoint-{best:06d}/branch.safetensors includes the frozen core and refiner; optimizer.pt is retained.", size=20)
    card(ax, 45, 403, 730, 221, 'LIMITATIONS AND VISIBLE DEFECTS', 'One ID and 24 development images; finite cached features and a shared recognizer family. Persistent eye/mouth defects are evident in 01_s1, 05_s1 and 10_s1. Goggles are damaged in 02/02_s1. Native conditioning already carries reference information.', size=19)
    card(ax, 815, 403, 735, 221, 'NEXT HYPOTHESIS — UNTESTED', 'Fresh online noise/context and trajectory-consistent identity supervision may reduce the loss-to-generation gap. Multi-identity evaluation and an independent recognizer would be required to test broader identity transfer.', size=19)
    para(ax, 65, 666, 'Reproduce: plans/260930/IDENTITY_FLOW.md; scripts.run_deterministic_identity_flow; scripts.review_identity_flow. All recorded stages are complete; do not restart old runs.', 100, 16)
    para(ax, 65, 738, COMET, 140, 15, color=C['teal'])
    text(ax, 65, 787, 'Native source '+identity['base']['source_commit']+'  |  weights '+identity['base']['revisions']['flux4b'], size=11)

    masks = read(RUN / 'routing_masks.json')['samples']
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    columns = [('Native', RUN / 'native'), ('Initial', RUN / 'validation-000000'),
               (f'Best +{best}', RUN / f'validation-{best:06d}'), (f'Final +{latest}', RUN / f'validation-{latest:06d}')]
    for group in range(6):
        keys = list(masks)[4*group:4*(group+1)]
        panel = Image.new('RGB', (4*260, 4*230+40), 'white'); draw = ImageDraw.Draw(panel)
        for col, (label, _) in enumerate(columns): draw.text((col*260+8, 8), label, fill='black', font=font)
        for row, key in enumerate(keys):
            x0, y0, x1, y1 = masks[key]['face_bbox']; cx, cy = (x0+x1)/2, (y0+y1)/2
            side = max(x1-x0, y1-y0)*1.35
            crop = tuple(map(int, (cx-side/2, cy-side/2, cx+side/2, cy+side/2)))
            for col, (_, folder) in enumerate(columns):
                image = Image.open(folder / f'{key}.png').crop(crop).resize((207, 207), Image.Resampling.LANCZOS)
                panel.paste(image, (col*260+26, row*230+39))
                draw.text((col*260+6, row*230+247), key, fill='black', font=font)
        path = OUT / f'paired_faces_{group+1}.png'; panel.save(path)
        fig, ax = page(f'Paired faces · fixed panel items {group*4+1}–{group*4+4} of 24',
                       'Native | initial | best | final · identical crop per case · all 24 cases shown without quality selection', source)
        image_panel(fig, path, 45, 152, 1120, 668)
        rows = [(k.replace('oneid_', ''), f'{at_best[k]-before[k]:+.4f}', f'{final[k]-at_best[k]:+.4f}') for k in keys]
        table(ax, 1175, 160, [110, 132, 133], ['Case', 'Best−init', 'Final−best'], rows, row_h=77, size=14)
        card(ax, 1175, 565, 375, 231, 'VISUAL REVIEW', 'Inspect face geometry, eyes, mouth and texture alongside ID_sim. Higher embedding similarity can coexist with blur or structural artifacts.', size=19)


def finish(identity, status, scores, decision, cutoff):
    target = OUT / 'flux4b_deep_identity1024_architecture.pdf'
    with PdfPages(target, metadata={'Title': 'Three-read BA with direct face identity supervision', 'Author': 'rsrch_new'}) as pdf:
        for number, (fig, _) in enumerate(s.pages, 1):
            fig.canvas.draw()
            for artist, (x, y, w, h) in s.fitted_texts:
                if artist.figure is not fig: continue
                for _ in range(3):
                    bb = artist.get_window_extent(fig.canvas.get_renderer())
                    a, b = artist.axes.transData.transform([[x, y], [x+w, y+h]])
                    ratio = min((b[0]-a[0])/bb.width, (a[1]-b[1])/bb.height, 1)
                    if ratio >= 1: break
                    artist.set_fontsize(artist.get_fontsize()*ratio*.96)
            fig.canvas.draw()
            for artist in s.texts:
                if artist.figure is not fig: continue
                bb = artist.get_window_extent(fig.canvas.get_renderer())
                assert bb.x0 >= -1 and bb.y0 >= -1 and bb.x1 <= fig.bbox.width+1 and bb.y1 <= fig.bbox.height+1, (number, artist.get_text())
            pdf.savefig(fig); fig.savefig(OUT / f'page_{number:02d}.svg'); plt.close(fig)
    doc = pymupdf.open(target)
    doc.set_toc([[1, title, i] for i, (_, title) in enumerate(s.pages, 1)]); doc.saveIncr()
    for number, name in [(2, 'whole_model'), (4, 'branched_attention')]:
        single = pymupdf.open(); single.insert_pdf(doc, from_page=number-1, to_page=number-1)
        single.save(OUT / f'{name}.pdf'); single.close()
    thumbs = Image.new('RGB', (1280, 190*((len(doc)+3)//4)), '#DAE2E7')
    for i, p in enumerate(doc):
        p.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(OUT / f'page_{i+1:02d}.png')
        im = Image.open(OUT / f'page_{i+1:02d}.png'); im.thumbnail((320, 180))
        thumbs.paste(im, ((i%4)*320, (i//4)*190))
    thumbs.save(OUT / 'contact_sheet.png'); doc.close()
    files = {Path(__file__), Path(s.__file__), PRECEDENT, *s.source_files, *(ROOT/p for p in identity['source_sha256']),
             ROOT/'sources/ai-toolkit-flux/extensions_built_in/diffusion_models/flux2/src/model.py'}
    audit = {'evidence_cutoff': cutoff, 'run': str(RUN), 'comet': COMET, 'identity': identity, 'status': status,
             'image_scores': scores, 'decision': decision, 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             'note': 'Uncommitted implementation is identified by immutable source snapshots and hashes, not git HEAD alone.',
             'source_hashes': {str(p): hashlib.sha256(READ_BYTES.get(p, p.read_bytes())).hexdigest() for p in files | READ_BYTES.keys()},
             'pdf_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
             'pages': [{'number': i, 'title': t} for i, (_, t) in enumerate(s.pages, 1)]}
    (OUT / 'source_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({'pdf': str(target), 'pages': len(s.pages), 'latest_step': max(scores), 'best_step': decision['best_step'], 'sha256': audit['pdf_sha256']}))


if __name__ == '__main__':
    main()
