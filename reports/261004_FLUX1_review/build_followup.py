"""Build the shortcut audit and three concrete FLUX1 follow-up recipes, CPU only."""
import csv
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pymupdf

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
evidence=OUT/'evidence'
faces=json.loads((evidence/'native_face_similarity.json').read_text())
masks=json.loads((evidence/'shortcut_masks.json').read_text())
structure=json.loads((evidence/'shortcut_structure.json').read_text())
identity=json.loads((evidence/'identity_cpu_check.json').read_text())
checks=json.loads((evidence/'followup_local_checks.json').read_text())
PARTS=[]


def table(headings,rows):
    return '<table><tr>'+''.join(f'<th>{x}</th>' for x in headings)+'</tr>'+''.join(
        '<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</table>'


def page(title,body):
    body=re.sub(r'\b(full|fixed|rank|alpha|sigma|saved|all|at|historical|original|width|double|single|seed|microbatch|accumulation|warmup|clip|every|below|with|only|in)([0-9])', r'\1 \2', body, flags=re.IGNORECASE)
    PARTS.append(f'<section><h1>{title}</h1>{body}</section>')


def plots():
    comparisons={row['sample_id']:row for row in csv.DictReader((OUT/'per_image_comparison.csv').open())}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9.3,3.5),layout='constrained')
    values=[r['native_trained_identity_cosine'] for r in faces['samples']]
    gains=[float(comparisons[r['sample_id']]['delta_id_sim']) for r in faces['samples']]
    axes[0].scatter(values,gains,s=22,color='#087f8c',alpha=.72)
    axes[0].axhline(0,color='#526574',lw=.8)
    axes[0].set(xlabel='Cosine: native face vs FLUX1 8k face',ylabel='Target-ID gain over native',title='96 paired faces: similarity is not fidelity')
    percentages=[100*r['fraction_face_pixels_in_fractional_tokens'] for r in masks['samples']]
    axes[1].hist(percentages,bins=14,color='#d48145',edgecolor='white')
    axes[1].axvline(np.mean(percentages),color='#744523',ls='--',lw=1)
    axes[1].set(xlabel='Face-box pixels in fractional latent cells (%)',ylabel='Images',title='Native blending reaches face boundaries')
    fig.savefig(OUT/'shortcut_measurements.png',dpi=180)
    fig.savefig(OUT/'shortcut_measurements.pdf')
    plt.close(fig)


page('FLUX1 shortcut audit and next experiments',f'''
<p class="eyebrow">Vast9B · 4 October 2026 · Addendum to the architecture/results report</p>
<div class="callout"><b>Run FLUX1a next.</b> It removes two concrete shortcuts and gives BA an explicit identity objective while keeping the native backbone frozen. FLUX1b tests the objective's contribution; FLUX1c tests more adapter capacity. These are implemented proposals, not measured quality improvements.</div>
<p><b>Your concern has a structural basis.</b> Current FLUX1 reads reference K/V from a joint stream whose reference tokens already depend on the target. During training the target is a noised ground-truth photo; during inference it is the evolving generated scene. Selecting reference positions does not make their contents target-independent. Detaching these tensors would stop gradients but retain contaminated values.</p>
<p><b>A second path is spatial.</b> Average-pooled face masks create fractional latent cells. These cells blend the generated native face back into the second denoising pass. In the saved96 masks, {100*masks['mean_fraction_face_pixels_in_fractional_tokens']:.1f}% of face-box pixels fall in fractional cells; the face-pixel average native-clean coefficient at sigma0 is {100*masks['mean_native_clean_coefficient_at_sigma0']:.2f}%. This coefficient is not an estimate of causal identity influence.</p>
{table(['Priority','Experiment','Purpose'],[
 ['1','FLUX1a','Isolated reference-image bank + binary face-token ownership + rank128 Q/K/V/O + identity auxiliary. Highest-potential first run.'],
 ['2','FLUX1b','Same bank, ownership and rank128; flow loss only. Establish whether the identity auxiliary helps.'],
 ['3','FLUX1c','Same as FLUX1a, rank256/alpha256. Test extra BA capacity after fixing information flow.']])}
<p>The requested-identity score on full96 is 0.2742 native versus 0.2762 at8k. Native-to-FLUX1 face cosine averages {faces['mean_native_trained_identity_cosine']:.4f}, median {faces['median_native_trained_identity_cosine']:.4f}. Faces are related but not identical; this comparison has no universal same-person threshold. Merely making them less like native would be the wrong objective.</p>
<p class="small">The original20-page report and its full96 visual appendix remain valid measured evidence. This addendum supersedes its generic experiment shortlist with executable FLUX1a/b/c recipes. The historical2k full96 replay is running separately; no new training was started.</p>
''')

page('Where information can bypass reference identity','''
<div class="flow"><b>Current FLUX1</b><br>target + text + reference image<br>↓ frozen joint attention, including target→reference edges<br>mixed reference features → selected reference-face K/V → BA target-face read<br>↑ later layers can also contain feedback from earlier BA writes</div>
<div class="flow"><b>FLUX1a / b / c</b><br>reference image only, original position IDs, current sigma, zero text/target tokens<br>↓ separate frozen native-backbone forward<br>immutable per-prediction bank → face K/V + trainable K/V deltas<br>target's native Q + Q delta → native SDPA reference read → native output projection + O delta</div>
'''+table(['Route','Audit conclusion','Handling'],[
 ['Target→reference hidden state','Confirmed structural dependency at all8 tested sites, including before the first BA read.','Build the reference bank in an independent image-only forward.'],
 ['Native clean latent→face boundary','Confirmed fractional token interpolation in the saved96 masks.','Use binary support for all cells touched by the existing pixel mask.'],
 ['Final RGB compositor→face interior','No direct copy: native pixel coefficient is zero inside each saved face box.','Keep the existing exterior feather and exact background pixels.'],
 ['Target queries, residuals, MLPs and unmodified blocks','Native priors remain; they can carry pose, scene and identity information.','Preserve the native backbone; test BA authority with a donor-bank intervention.'],
 ['Native stream consumes the original reference','Legitimate existing native conditioning; it is held fixed in the causal probe.','No removal or new fusion of native conditioning.'],
 ['Validation target photographs','None loaded by the audited validation path.','Training-only landmarks and target embeddings never enter generation.']])+'''
<p><b>Interpretation:</b> the first route is a shortcut in a bank that is meant to convey the reference identity. It is not evidence of train/validation dataset overlap. The training noised target itself is required by the flow objective. The causal magnitude of these routes in the trained9B checkpoint remains unmeasured.</p>
<p class="small">The original implementation plan already described FLUX reference computation as joint and target-dependent. The defect is relying on that stream as an isolated BA identity source. Isolation here preserves native attention math; it does not impose a new causal mask on the native joint stream.</p>
''')

page('Measurements and their limits',f'''
<img src="shortcut_measurements.png" width="510">
<p class="caption">Left: all96 saved native/8k face pairs, using the same owner detector and recognizer. Right: exact mask geometry, before any model intervention. These plots are descriptive, not causal attribution.</p>
{table(['Measurement','Result','Scope'],[
 ['Target-only perturbation, legacy reference Q/K/V','Mean absolute change0.00437 at double2;0.01097 at single22; nonzero at all8 sites.','CPU full-depth9B topology with width16, random weights. Numerical scale is not a9B model estimate.'],
 ['Same perturbation, isolated K/V','Maximum absolute change exactly0 at all8 sites.','Reference, text and sigma fixed. Also checked hidden features and RoPE positions.'],
 ['Reference-only perturbation','Changes isolated-bank features and target output with the native input held byte-identical.','Structural authority demonstrated; trained semantic identity authority still needs images.'],
 ['Native/8k face cosine',f"Mean{faces['mean_native_trained_identity_cosine']:.4f}; median{faces['median_native_trained_identity_cosine']:.4f}; range{faces['min_native_trained_identity_cosine']:.4f}–{faces['max_native_trained_identity_cosine']:.4f}.",'All96 owner-matched pairs. Not a requested-identity score.'],
 ['Face-box RGB change','Mean absolute difference26.20 on the0–255 scale.','Affected by expression, lighting and geometry as well as identity.'],
 ['Native copy inside final face rectangle','Maximum coefficient0.','Exact saved pixel compositor, all96 masks.']])}
<p>Full96 at8k has47 wins and49 losses against native. The mean gain is+0.00203 and only two of eight identity means improve. The identity-cluster bootstrap95% interval is−0.0223 to+0.0324. This does not establish broad identity improvement.</p>
<p>The12-image pilot rises from0.3816 native to0.4341 at2k, then drops to0.4165 at8k. Its2k→8k change is−0.0176, with an identity-cluster interval−0.0636 to+0.0260. Its narrow scene coverage and uncertainty prevent calling this general overfitting. Keep the requested full96 replay at2k as the decisive missing comparison.</p>
''')

page('Exact recipe: stronger BA without output fusion','''
<p><b>Shared architecture.</b> Frozen FLUX.2-klein Base9B, width4096,32 heads,8 double and24 single blocks. BA sites remain double2/4/6/7 and single4/10/16/22. Q/K/V deltas enter before native Q/K normalization and RoPE; the O delta follows the native output projection and precedes the native modulation gate. No new attention kernel or replacement velocity head is introduced.</p>
<p><b>Isolated bank.</b> At each prediction, run the same frozen backbone on the original reference-image tokens alone, at the current sigma, with zero text tokens. Capture the pre-projection hidden state and native pre-normalization K/V at each site, retaining original reference RoPE coordinates. Select reference-face keys using the unchanged mask/cap512. Reference image context can influence those face features; target, prompt and BA feedback cannot.</p>
<p><b>Full token ownership.</b> Let A be the existing pixel alpha mask and M=1[AvgPool16(A)&gt;0]. At selected sites, M=1 chooses the reference attention message fully; M=0 retains native attention. In the spatial denoising loop, owned latent cells come wholly from the face trajectory starting at fresh noise. The clean native scene fills unowned cells. Only the final outer pixel feather interpolates RGB; the face-box interior has full generated weight.</p>
<p><b>FLUX1a objective.</b> Preserve face-normalized native flow MSE, fresh noising and timestep draws; add a small identity term on usable training targets when sigma≤0.5:</p>
<div class="eq">x̂0 = xσ − sigma · vθ(xσ)<br>L = Lflow +0.05 · [1−cos(ArcFace(Align(VAEdecode(x̂0))), e_target)]</div>
<p>Use the actual FP32 noising coefficient for x̂0; preserve native BF16 rounding for model conditioning. Alignment uses frozen training-target landmarks, not a detector on the generated prediction. The recognizer and VAE weights stay frozen while gradients reach BA through both. Decoder activations are checkpointed; alignment uses the existing deterministic bilinear gather implementation. Invalid labels skip only the auxiliary, retaining every flow-training row.</p>
'''+table(['Experiment','Rank / alpha','Parameters','Identity term'],[
 ['FLUX1a','128 /128','33,554,432 in64 tensors','0.05, sigma≤0.5'],
 ['FLUX1b','128 /128','33,554,432 in64 tensors','None'],
 ['FLUX1c','256 /256','67,108,864 in64 tensors','0.05, sigma≤0.5']])+'''
<p class="small">All are fresh initializations. Zero-initialized LoRA deltas do not make BA-on equal native: the reference-read routing itself is already active. Preserve and evaluate each BA-on step0 alongside its BA-off control. Rank256 is an ablation, not evidence that capacity was the original bottleneck.</p>
''')

page('Launch setup and admission','''
<p>Use a separate GB10 checkout, e.g. <code>/workspace/rsrch_FLUX1abc</code>. Keep the current historical evaluation under <code>/workspace/rsrch_9b_8k</code> unchanged. Install the pinned Toolkit source at <code>ecee894ed2b1f3716d9d7326693061ec1a3105bb</code> with the new recorded patch; do not reuse the old patched source directory.</p>
<p>Reuse verified weights, datasets and environment paths. Copy the full Large manifest/import audit with valid image paths; new supervision and runs belong to the new checkout. FLUX1a/c require <code>onnx==1.23.1</code> in the FLUX environment and the existing metrics/InsightFace environment for CPU label preparation. Freeze buffalo_l weight hashes. Full preparation has47,341 pairs; only a two-pair CPU wiring check has been executed here.</p>
<pre>scripts/run_FLUX1a.sh                   # plan only
scripts/run_FLUX1a.sh --action prepare  # training labels
scripts/run_FLUX1a.sh --action run \\
  --native-bundle /workspace/rsrch_9b_8k/runs/flux9b_8000_fixed96_20261004</pre>
<p>Use <code>run_FLUX1b.sh</code> and <code>run_FLUX1c.sh</code> for the later runs. b needs no identity preparation; c reuses a's verified training labels. Append <code>--resume</code> only for an interrupted run of the same recipe, with its immutable source/config intact.</p>
'''+table(['Setting','All three recipes'],[
 ['Data / ordering','Full pinned Large pairs; no filtering of flow rows; seed142 and the same order.'],
 ['Optimization','4,000 updates; microbatch1; accumulation1; AdamW lr5e-5; warmup100; weight decay0; clip1.'],
 ['Validation','Original fixed96 at0,2000,4000;768px; reference512;20 steps;CFG4;batch2.'],
 ['Saving / logging','Checkpoint every500; independent immutable Comet key under rsrch_new per run. Original ID/CLIP metric definitions.'],
 ['Frozen comparison inputs','Hash-verified native PNGs, latents and masks from the completed8k fixed96 bundle.'],
 ['Memory admission','Measured peak CUDA reserved below85% in the actual pretrained configuration; no automatic fallback.']])+'''
<p><b>Before long training, the launcher requires:</b> native-off and zero-mask parity; frozen-weight equality; finite gradients and updates in all64 adapters; identity-only gradient and ONNX parity for a/c; same-batch weighted-ID/flow gradient-norm ratio; largest-reference/full-routing memory stress; exact fresh-process optimizer/RNG/cursor replay. If any check fails, it stops.</p>
<p class="small">Bank extraction adds one frozen reference-only forward per prediction, including each CFG branch. ID supervision additionally decodes eligible training estimates and backpropagates through the recognizer. Exact throughput and peak memory remain unmeasured; the previous52.3GiB FLUX1 peak is not a guarantee for these recipes.</p>
''')

page('How to tell whether BA has become stronger','''
<p><b>Primary comparison:</b> use the full96 paired requested-ID changes against the frozen native baseline and the historical FLUX1 checkpoints. Report the overall mean, all eight identity means, win count, missing/ambiguous faces and an identity-cluster uncertainty interval. Inspect all96 images for expression, pose, accessories and artifacts; full-frame CLIP can hide face-only regressions because backgrounds are fixed.</p>
<p><b>BA authority:</b> the supplied <code>scripts/probe_flux1_reference.py</code> generates a separate, named eight-image diagnostic (first original-panel sample per identity). Three arms use the same native conditioning, prompt, seed, native latent, spatial mask, sampler and checkpoint:</p>
'''+table(['Arm','What changes','Interpretation'],[
 ['BA-off second pass','Disable the branch while retaining the complete second-pass procedure.','Controls for the two-pass procedure and native conditioning.'],
 ['Own isolated bank','Enable BA with the intended reference image.','Measure own-ID gain over BA-off.'],
 ['Different-person bank','Swap only the independent bank to the next identity; native reference inputs stay fixed.','Donor-ID gain together with own-ID drop supports semantic control by the BA bank.']])+'''
<pre>envs/flux-toolkit/bin/python -m scripts.probe_flux1_reference \\
  generate --run runs/FLUX1a_vast_9b --step 2000 \\
  --output runs/FLUX1a_vast_9b/probe_swap8_002000
envs/metrics/bin/python -m scripts.probe_flux1_reference \\
  score --output runs/FLUX1a_vast_9b/probe_swap8_002000</pre>
<p>The diagnostic records input/checkpoint hashes and own/donor ID scores using the original subject-v2 prototypes and owner matching. It writes separate files and never replaces fixed96 scores or Comet curves. Run it only when the validation GPU lock is free.</p>
<p><b>Decision order.</b> Start a because it addresses information flow and the missing identity objective together. Run b next to isolate the auxiliary's benefit. Run c third if a shows useful reference control and capacity remains a plausible limit. If a gives weak/no donor response, inspect bank feature quality and gradient norms before paying for more rank or an8k continuation.</p>
<p>There is no loss rewarding arbitrary distance from native. A successful checkpoint should move toward the requested identity while retaining prompt-directed expression and natural appearance. Predeclare a practical improvement threshold before viewing new results; for example+0.02 mean owner-ID with improvements in at least6/8 identities is a useful engineering target, not a significance test.</p>
<p class="small">The auxiliary and historical ID metric use the same ArcFace family. An improvement can overfit that recognizer. Before a broader claim, add a separately named independent-recognizer audit and a new held-out panel; do not alter the established96-item benchmark.</p>
''')

page('Validation evidence, caveats and sources',f'''
{table(['Completed locally','Evidence'],[
 ['Six new information-flow tests','Target-invariant isolated bank; own/donor response; native and zero-mask parity; two outstanding checkpointed graphs; batch/single consistency; exact native-input preservation during donor swaps.'],
 ['Seven existing FLUX invariants','Native integration and layout, adapter registration, checkpoint identity, optimizer/RNG/sample-order replay.'],
 ['Five additional focused tests','Existing masked-attention/ownership, spatial flow and alignment checks, plus the new aligned identity-gradient/noise-gate regression.'],
 ['Real ArcFace CPU wiring',f"Two real training targets accepted. ONNX vs differentiable max error{identity['arcface_max_abs_onnx_error']:.8f}; prediction gradient norm{identity['prediction_gradient_norm']:.6f}. Decoder was a deterministic surrogate, not the pretrained VAE."],
 ['Recipe/script checks','All three plans, exact parameter counts, shell syntax and Python syntax pass. No recipe has pretrained GPU admission or quality results yet.']])}
<p><b>Main risks.</b> The image-only/zero-text feature pass is a proposed use of the pretrained joint model and may have a feature-distribution mismatch. Full token ownership enlarges effective BA support and may harm boundary geometry. One-step x̂0 can be unreliable even below sigma0.5. Rank256 can cost more without improving identity. Training-target landmarks are fixed and can become misaligned if predicted faces move.</p>
<p><b>Research basis.</b> <a href="https://arxiv.org/html/2404.16022v2">PuLID, §§3.3–3.5</a> explains why ID supervision on a noisy one-step clean estimate can be inaccurate and uses a short generation path for cleaner supervision. Our low-noise gate is a cheaper, incomplete mitigation; this implementation does not reproduce PuLID's architecture or objectives.</p>
<p><a href="https://arxiv.org/abs/2510.14975">WithAnyone</a> emphasizes paired identities, copy-paste failure modes, and balancing identity fidelity with variation. That motivates retaining cross-view pairs and examining expression/pose alongside ID. Our simple target-aligned cosine auxiliary is not its contrastive training procedure.</p>
<p><b>Reproducibility.</b> Source entry points: <code>ba_dit/nn/isolated_reference.py</code>, <code>online_identity_loss.py</code>, <code>masked_face_attention.py</code>, <code>masked_face_flow.py</code>, <code>ba_dit/backends/flux_runtime.py</code>, and the recorded Toolkit patch. Configs: <code>configs/FLUX1a_vast_9b.yaml</code>, b and c equivalents. Operational instructions: <code>docs/FLUX1_DEPLOYMENTS.md</code>.</p>
<p>Raw audit receipts are in <code>reports/261004_FLUX1_review/evidence/</code>: <code>shortcut_structure.json</code>, <code>shortcut_masks.json</code>, <code>native_face_similarity.json</code>, <code>identity_cpu_check.json</code>, <code>followup_local_checks.json</code> and <code>followup_source_identity.json</code>. Charts are also saved as a standalone vector PDF.</p>
<p class="small">Scope is Vast9B. The HSE4B run, its precision and its current DDP continuation are separate experiments. No commits, pushes, rentals or new training launches were performed for these recipes.</p>
''')

CSS='''body{font-family:sans-serif;font-size:10pt;line-height:1.28;color:#20323f}
h1{font-size:21pt;color:#076775;line-height:1.12;margin:0 0 12pt}p{margin:7pt 0}
.eyebrow,.small,.caption{font-size:8.8pt;color:#546774}.callout,.flow,.eq{background:#eaf4f4;padding:10pt;margin:10pt 0}
.callout{border-left:4pt solid #087f8c}.flow{font-size:10.3pt}.eq{font-family:monospace;font-size:9pt}
table{border-collapse:collapse;width:100%;font-size:9.1pt;margin:10pt 0}th{background:#076775;color:white;text-align:left;padding:6pt}
td{padding:5pt;border-bottom:.5pt solid #d6e0e5;vertical-align:top}tr:nth-child(even){background:#f3f6f8}
a{color:#087f8c}pre{font-family:monospace;font-size:7.5pt;background:#f1f5f7;padding:8pt;white-space:pre-wrap}code{font-family:monospace;font-size:8pt}
'''


def build():
    plots()
    body=''.join(PARTS)
    (OUT/'FLUX1_shortcut_audit_and_next_experiments.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>FLUX1 shortcut audit and next experiments</title><style>'+CSS+'body{max-width:980px;margin:28px auto;padding:0 24px;font-size:16px}section{padding:22px 0;border-bottom:1px solid #ccd8de}img{max-width:100%;height:auto}table{font-size:14px}pre{font-size:12px}</style></head><body>'+body+'</body></html>')
    pdf=pymupdf.open(); sections=[]
    for index,part in enumerate(PARTS):
        story=pymupdf.Story(html=part,user_css=CSS,archive=pymupdf.Archive(str(OUT)))
        doc=story.write_with_links(lambda _,__: (pymupdf.Rect(0,0,595,842),pymupdf.Rect(42,51,553,792),None))
        sections.append({'section':index+1,'start':len(pdf)+1,'pages':len(doc)})
        pdf.insert_pdf(doc);doc.close()
    for index,p in enumerate(pdf):
        p.insert_text((42,28),'FLUX1 | shortcut audit and executable follow-up proposals',fontsize=8,color=(.32,.40,.45))
        p.insert_text((42,818),'4 October 2026 | CPU checks completed; pretrained follow-up quality unmeasured',fontsize=7.5,color=(.32,.40,.45))
        p.insert_text((515,818),f'{index+1}/{len(pdf)}',fontsize=8,color=(.32,.40,.45))
    pdf.set_metadata({'title':'FLUX1 shortcut audit and next experiments','author':'rsrch_new','subject':'Isolated reference features, full BA token ownership and three prioritized recipes'})
    pdf.save(OUT/'FLUX1_shortcut_audit_and_next_experiments.pdf',garbage=4,deflate=True)
    bounds=[]
    for index,p in enumerate(pdf):
        for block in p.get_text('dict')['blocks']:
            if block['type']!=0:continue
            for line in block['lines']:
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    if x0<0 or x1>596 or y0<0 or y1>843:bounds.append({'page':index+1,'text':span['text']})
    audit={'pages':len(pdf),'sections':sections,'bounds_violations':bounds,'links':sum(len(p.get_links()) for p in pdf),
           'page_text_characters':[len(p.get_text()) for p in pdf]}
    (OUT/'followup_document_validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    for index in (0,1,3,4,6):
        if index<len(pdf):pdf[index].get_pixmap(matrix=pymupdf.Matrix(1.25,1.25)).save(OUT/f'followup_preview_{index+1:02d}.png')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    build()
