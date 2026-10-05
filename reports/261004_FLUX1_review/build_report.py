"""Build a self-contained PDF and a linked HTML research report from analysis.json.

Run analyze.py first, then this file with envs/report/bin/python. CPU only.
"""
import html
import json
from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
A = json.loads((OUT / 'analysis.json').read_text())
PARTS = []


def table(headers, rows):
    return '<table><thead><tr>'+''.join(f'<th>{x}</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def image(name, caption, width=510):
    return f'<div class="figure"><img src="{name}" width="{width}"><p class="caption">{caption}</p></div>'


def page(title, body):
    PARTS.append(f'<section><h1>{title}</h1>{body}</section>')


def f(x, digits=4):
    return f'{x:.{digits}f}'


def link(url, label):
    return f'<a href="{url}">{label}</a>'


native = A['full96']['native']
trained = A['full96']['trained8000']
pilot = A['pilot12']

page('FLUX1: architecture, results and next experiments', f'''
<p class="eyebrow">Research review · 4 October 2026 · rsrch_new</p>
<div class="callout"><b>FLUX1 demonstrably learns to denoise through its added branch. Broad improvement in held-out identity fidelity is not yet established.</b> The completed full96 evaluation is more informative than the 12-image training monitor: at 8k it is almost tied with the native baseline.</div>
{table(['Evaluation', 'Native ID', 'FLUX1 ID', 'Change'], [
    ['Pilot12, 2k (best observed pilot)', '0.3816', '0.4341', '+0.0525'],
    ['Pilot12, 8k', '0.3816', '0.4165', '+0.0349'],
    ['Full96, 8k', '0.2742', '0.2762', '+0.0020']])}
<p><b>The apparent peak is real on the saved pilot outputs, but its cause is unresolved.</b> ID similarity falls by 0.0176 from 2k to 8k; five images improve and seven regress. A descriptive 95% bootstrap interval, resampling the eight identities as groups, is −0.0636 to +0.0260. This is insufficient to call the decline general overfitting, and full96 has not been evaluated at 2k.</p>
<p><b>The pilot is narrow in scene coverage.</b> It contains eight identities but only “reading on a park bench” and “angry in traffic” scenes, all at seed 0. Full96 has twelve prompts per identity. In the full96 execution, the pilot rows gain +0.0307 on average; the other 84 rows lose 0.0021.</p>
<p><b>Good visual consistency and modest face ID can coexist.</b> Full96 has zero missing or unowned faces, yet only 47/96 improve and six of eight identity averages fall. Background pixels are copied exactly from the native scene by design. Strong expressions, goggles, small faces and recognizably different facial details occur in both native and FLUX1 images.</p>
<p><b>Recommended order:</b> finish 2k full96, isolate the branch’s contribution, then test lower LR, identity supervision and stronger reference features separately. Per the requested design direction, keep native attention unchanged and strengthen the existing reference read without adding native/reference output fusion. Keep 2k as the pilot-selected checkpoint, without calling it the full96 winner.</p>
<p class="small">Scope: Vast9B FLUX1, completed native/8k full96 and earlier local4B one-ID evidence. HSE-cluster4B is separate. “FLUX1” is the user’s experiment name, not BFL’s FLUX.1 family. The newly requested 2k full96 evaluation is running on GB10 with the exact 8k inputs and Comet ID; this report’s quality tables contain completed results only.</p>
''')

page('What the current architecture actually does', '''
<p>FLUX1 uses the frozen FLUX.2-klein <b>Base 9B</b> denoiser: width 4096, 32 attention heads of width 128, eight double-stream blocks and 24 single-stream blocks. It trains branch-local Q/K/V/output low-rank projections at eight sites. The original final velocity head, MLPs, modulation, attention output projections, text encoder and VAE remain frozen.</p>
'''+table(['Property', 'Vast9B implementation'], [
    ['Sites, zero based', 'Double: 2, 4, 6, 7. Single: 4, 10, 16, 22.'],
    ['Trainable parameters', '33,554,432 in 64 tensors. Four rank128 projections per site, with A and B matrices; alpha128 gives scale 1.'],
    ['Native LoRA', 'Disabled. The rank8 lora config entry is an inactive schema field.'],
    ['Input geometry', '768×768 target: 48×48 = 2304 packed 128-channel tokens. Reference uses at most 512² pixels with preserved aspect ratio, not a mandatory 512×512 crop.'],
    ['Reference support', 'Clean VAE reference tokens enter native joint attention. The added branch selects face-box tokens, capped at 512 using deterministic stratified indices.'],
    ['Branch mask', 'Target face box plus 16px outer feather, average-pooled to the 16px token grid. Gamma is fixed at 1; there is no learned branch-strength gate.']])+'''
<p><b>Token ownership is explicit.</b> In a double block’s image stream, target positions are [0, N<sub>t</sub>) and reference positions start at N<sub>t</sub>. The joint native attention also contains text. In a single block the sequence is [text, target, reference], so target/reference indices add N<sub>text</sub>. Only selected target-face query rows are overwritten at the insertion site.</p>
<div class="eq">Q′ = RoPE<sub>target</sub>(Norm<sub>Q</sub>(Q<sub>native</sub> + ΔQ(h<sub>target</sub>)))<br>
K′ = RoPE<sub>reference</sub>(Norm<sub>K</sub>(K<sub>native</sub> + ΔK(h<sub>reference</sub>)))<br>
V′ = V<sub>native</sub> + ΔV(h<sub>reference</sub>)<br>
R = Attention(Q′, K′, V′)<br>
A<sub>routed</sub> = (1 − m) A<sub>native</sub> + m R</div>
<p>Each Δ projection is BAh, with FP32 trainable weights and B initialized to zero. Q/K deltas are added <b>before native Q/K normalization and the original positional rotations</b>. Reference positions are retained when gathering K/V; they are not renumbered as target positions.</p>
<p>The native output projection receives A<sub>routed</sub>. The additional output term mΔO(R) is then added <b>before the native modulation gate</b>. In single blocks the native projection jointly consumes attention and MLP channels; its MLP input is preserved.</p>
<p class="small">Evidence: frozen source_snapshot/ba_dit/nn/masked_face_attention.py; backends/flux2_native.py; patches/flux2_reference_branch_and_offload.patch. Twenty-six parent snapshot files were rehashed successfully.</p>
''')

page('The central design tradeoff: replacement and two passes', '''
<div class="callout"><b>Zero LoRA deltas do not make FLUX1’s BA-on path equal to native.</b> At step zero, the branch still replaces a face query’s joint text/target/reference attention message with an attention read over reference-face tokens only. This is why “untrained BA” is a separate and much worse baseline.</div>
<p>Inside a fully weighted face region, m=1, so the selected attention message becomes R. The residual stream, MLP, previously accumulated prompt information and unmodified blocks remain active; FLUX1 does not erase all text conditioning. Nevertheless, it removes the current native joint-attention contribution at those sites. The large 0→2k improvement therefore combines repair of this disruption with any additional identity benefit.</p>
<p><b>BA-off and locality mean different things.</b> When BA is disabled, the patched transformer follows native attention. With BA enabled, zero-mask rows at an insertion site keep their native values exactly. Later native joint attention can still propagate face changes into text, reference and other image states. Exact locality of the final picture is supplied by the inference protocol, not by a proof of globally isolated hidden states.</p>
<p><b>Inference is a spatial face-refinement procedure:</b></p>
<ol>
<li>Generate a native image from the prompt/reference and seed. Detect/freeze a face box and retain the native latent and image.</li>
<li>Run the same patched full denoiser with the trained BA, starting the face from noise. At every noise level, supply a noised version of the frozen native scene outside the face mask.</li>
<li>Decode the final latent and blend its face region into the original native pixels. Every exterior pixel is preserved exactly.</li>
</ol>
<div class="eq">z<sub>input</sub>(σ) = m z<sub>face</sub>(σ) + (1−m)[(1−σ)z<sub>native</sub> + σ ε]<br>
v<sub>CFG</sub> = v<sub>negative</sub> + 4(v<sub>positive</sub> − v<sub>negative</sub>)<br>
z<sub>face,next</sub> = z<sub>face</sub> + (σ<sub>next</sub> − σ)v<sub>CFG</sub></div>
<p>Both conditional and negative-prompt calls use the same reference conditioning and trained branch. Sampling uses 20 Euler steps. Target photographs are never loaded at validation; inference masks originate from the frozen native generation. Full96’s inference audit records exact latent exterior preservation, no target-photo loads and peak reserved memory of 20.373 GiB.</p>
<p><b>A missing control matters:</b> native one-pass generation versus trained two-pass FLUX1 measures the entire pipeline change. To isolate the branch, also run the second pass with BA disabled, using the identical frozen scene, masks, noise and sampler. A branch-only reference swap, leaving native reference conditioning unchanged, is then needed to show that improvements depend on the added reference route.</p>
<p class="small">Evidence: scripts/online_face_ba.py infer/decode; nn/masked_face_flow.py; completed 8k full96 inference/background audits. All 96 saved exterior-pixel audit maxima are zero. The new 2k replay uses the same protocol.</p>
''')

page('Training is healthy, but optimizes a different target', '''
<p>The full denoiser runs on every training example with fresh noise and native timestep draws. Targets and references are distinct views. Only the 64 branch tensors receive optimizer updates. The objective is native velocity error normalized over the training-photo face mask:</p>
<div class="eq">z<sub>σ</sub> = (1−σ)z<sub>GT</sub> + σ ε<br>
L<sub>flow</sub> = Σ m ‖v<sub>θ</sub>(z<sub>σ</sub>, σ, text, reference) − (ε−z<sub>GT</sub>)‖² / (C Σm)</div>
<p><b>There is no ArcFace/identity loss, contrastive identity loss, decoded perceptual loss or adapter EMA.</b> Face-local velocity accuracy can improve while recognition similarity of the final generated face worsens. The training context is a noisy real target scene; validation uses a separately generated frozen native scene. The masks share a spatial contract, but the context distribution differs.</p>
'''+table(['Executed setting', 'Value'], [
    ['Optimizer', 'AdamW, LR 5×10⁻⁵; 100-update warmup, then constant LR; weight decay 0; clip norm 1.'],
    ['Effective batch', '1 (microbatch1 × accumulation1). The earlier batch8 attempt is a different run.'],
    ['Data exposure', '8,000 examples from 47,341 eligible pairs: 16.9% of a first shuffled traversal. Cursor=8000; no complete epoch yet.'],
    ['Precision / memory', 'BF16 backbone, FP32 branch parameters/state; checkpointing off. Measured peak reserved 52.285 GiB, 43.0% of reported device capacity.'],
    ['Continuation integrity', '2k→8k changed only endpoint and validation cadence. Recorded full optimizer/RNG/cursor restoration; all 64 adapter tensors changed and remain finite.']])+image('training_curves.png', 'Saved 8,000 unique optimizer updates. Curves average changing training examples and noise levels; they are not held-out denoising losses.', width=465)+'''
<p>Mean loss is 0.8218 over updates 1001–2000 and 0.8028 over 6001–8000. All logged losses and gradients are finite; late pre-clipping gradient norms average 0.1749. Effective low-rank weight norms grow between 2k and 8k (Q 19.07→31.79, K 17.80→43.14, V 4.29→10.76, O 11.23→22.54; Frobenius norms pooled over sites). These establish ongoing adaptation, not a measured causal explanation for identity regression.</p>
<p>Repeated-data memorization is not established by “8k steps”: this run has not traversed the dataset once. Objective mismatch, constant-LR drift, limited scene coverage and variation among identities are better-supported first questions. Effective batch1 also makes a batch8 continuation an exposure-changing experiment unless samples seen are matched.</p>
''')

page('What the 12-image curve does and does not show',
table(['Checkpoint', 'ID similarity', 'CLIP text', 'ID change vs native'], [
    ['Native', f(pilot['native']['id_sim'],6), f(pilot['native']['text_sim'],4), '—'],
    *[[f'FLUX1 {s}', f(pilot['checkpoints'][str(s)]['id_sim'],6), f(pilot['checkpoints'][str(s)]['text_sim'],4),
       f(pilot['checkpoints'][str(s)]['id_sim']-pilot['native']['id_sim'],6)] for s in [0,1000,2000,4000,6000,8000]]])+
image('pilot_trajectory.png', 'Same fixed12 rows within this trajectory, unchanged sampling settings. The native baseline is BA-off; BA-on at zero is not a parity control.')+'''
<p><b>Measured:</b> the peak among sampled checkpoints is 2k. The 2k→8k identity drop is 0.01761 (4.1% of the 2k score). The 8k checkpoint remains 0.03487 above native on this pilot. CLIP does not decline monotonically; it rebounds at 6k before falling at 8k. Avoid summarizing every metric as a monotonic post-2k collapse.</p>
<p><b>The decline is uneven.</b> Jennie loses 0.1149 on average between 2k and 8k, Marion loses 0.0997, while Jensen gains 0.1121. Five individual images improve, seven regress. The identity-cluster 95% interval for the mean image change is −0.0636 to +0.0260. With only eight identity groups and one training run/validation seed, this interval is descriptive, not strong evidence about population generalization. Selecting the best of six checkpoints on these same rows adds selection bias.</p>
<p><b>Scene coverage is the larger problem.</b> Eight rows are prompt index0 (reading) and four index6 (angry traffic); Eddie, Jennie, Jisoo and Lex each appear twice, the other identities once. Full96 weights identities equally and includes skiing, drumming, boxing, dancing, crying, laughing and other scenes. All seeds are 0.</p>
<p>The prior local4B one-ID run also peaked and then dipped: native0.3314, BA0 0.1460, BA500 0.3658, BA1000 0.4303, BA2000 0.4190 on its separate fixed24 panel. This supports checking optimization and the objective, but a single training identity, two validation seeds and a different backbone do not establish the same cause in Vast9B.</p>
''')

page('The completed full96 result is close to neutral',
image('full96_results.png', 'All 96 fixed validation rows, with 12 prompts for each of eight held-out identities. Native and FLUX1 8k scores are paired within the full96 execution.')+
table(['Identity label', 'Native ID', '8k ID', 'Mean change', 'Improved / 12'], [
    [i, f(r['native']['id_sim']), f(r['trained']['id_sim']), f(r['paired']['mean_delta']), str(r['paired']['improved'])]
    for i,r in A['per_identity'].items()])+'''
<p>The full-panel gain is <b>+0.00203</b>, with an identity-cluster 95% bootstrap interval of <b>−0.02233 to +0.03245</b>. The median paired change is −0.00131. Forty-seven images improve and 49 regress; only Jensen and Keanu improve on average. This does not establish an overall improvement over native generation.</p>
<p>CLIP text score changes from 29.4402 to 29.3671 (−0.0731). It is a CLIP logit score, not a percentage. Since the pipeline fixes most of the image to the native result, a small whole-image CLIP change is weak evidence of preserved face-level expression, accessories or identity.</p>
<p><b>Keep the two executions separate.</b> On overlapping rows, the full96 8k mean is 0.4112, versus 0.4165 in the pilot. Four native images and three face boxes differ across executions; validation batching/grouping can alter numerical results. The maximum per-row 8k ID difference is 0.0411. The cause was not replayed here, so no cross-panel numbers are pooled. The full96 subset comparison uses its own native and trained outputs.</p>
<p class="small">Raw baseline CSVs were retrieved read-only from the original native12/native96 Comet experiments. Their means reproduce the saved summaries. Trained CSVs reproduce saved summaries; manifests/order and the full96 adapter hash match the recorded run. Full96 at 2k remains the decisive missing comparison.</p>
''')

page('Why good-looking images can have low face ID',
image('scene_and_size.png', 'Descriptive full96 strata. Scene and face-size associations do not establish causation; each scene has only eight examples.')+'''
<p><b>What visual inspection showed.</b> I inspected all 96 native/8k face pairs and all 12 native/2k/8k pilot comparisons. At 8k, faces are coherent and scene/expression continuity is strong. The dominant problems are changes to distinguishing facial details and difficult baseline faces, rather than a broad return to the malformed step-zero outputs. Goggles hide identity-bearing regions; hard shouting/crying expressions, rain, shadows and small faces are common in both lanes.</p>
<p>There are clear counterexamples to any single story: Jensen improves across 10/12 scenes, including laughing 0.217→0.422; Jennie’s drumming case falls 0.183→0.034; Jisoo’s chef case falls 0.199→approximately0 despite a plausible face. Marion’s drumming case falls 0.202→−0.010 while her dancing case rises 0.259→0.427. These are selected explanatory examples; the appendix includes every row.</p>
'''+table(['Full96 stratum', 'n', 'Native ID', '8k ID', 'Change'], [
    [label, str(A['subsets'][key]['n']), f(A['subsets'][key]['native']['id_sim']), f(A['subsets'][key]['trained']['id_sim']),
     f(A['subsets'][key]['trained']['id_sim']-A['subsets'][key]['native']['id_sim'])]
    for label,key in [('Pilot rows in this execution','pilot_rows_in_full96'),('Other 84 rows','other84'),
                      ('Face size <128px','face_under128'),('Face size 128–200px','face128_to200'),('Face size ≥200px','face_over200')]])+'''
<p class="small">Face size means square root of the frozen native face-box area; 53/96 fall below128px. These bins were chosen during analysis and are exploratory.</p>
<p><b>Consistency has several meanings.</b> Pixel/scene consistency is imposed by composition. A recognizable identity across images is a separate property. Facial attractiveness, realistic texture and matching hairstyle do not guarantee high cosine similarity to the supplied reference embedding. Some low scores reflect actual facial mismatch; the current evidence does not justify dismissing them all as a metric artifact.</p>
''')

page('Metric and reference-input audit', '''
<p><b>The headline metric is owner-matched ID similarity.</b> The evaluator uses InsightFace buffalo_l. It selects the detected output face with greatest IoU against a frozen native-derived owner box (minimum IoU0.05), then computes cosine similarity to <code>id_embeds_manual_val_subject_v2</code>. Ambiguity is flagged when another qualifying face is within0.02 IoU, rather than automatically excluded. This differs from choosing whichever face has the best identity score.</p>
<p>Full96 native and 8k have zero no-face, unowned, missing-mask or ambiguity flags. Native averages2.865 detected faces/image and 8k2.885; multiple faces therefore exist, but failed detection does not explain the low mean. The frozen owner is selected as the largest detected native face, not through semantic subject understanding. Visual review of challenging multi-person frames remains necessary even when the flags are clean.</p>
<p><b>Reference prototypes passed a direct CPU check.</b> Re-embedding the eight supplied reference images gives cosine approximately1.0000 against each matching subject_v2 prototype. There is no detected prototype-label mismatch in the current <code>id_sim</code>. The legacy-best metric is problematic for Eddie: its older prototype has cosine−0.0078 to the intended reference face, whereas subject_v2 matches at1.0000. Eddie’s reference has two detected faces. Keep the historical metric for continuity, but use owner-matched <code>id_sim</code> for conclusions.</p>
'''+table(['Reference label', 'Approx. face pixels after preprocessing', 'Selected face tokens'], [
    [i, ' × '.join(str(round(v)) for v in r['face_pixels_approx']), str(r['selected_face_tokens'])]
    for i,r in A['reference_geometry'].items()])+'''
<p><b>More keys alone will not fix this validation panel.</b> It uses only63–252 reference-face tokens, so the512-key cap is never reached. A larger cap cannot restore details already lost when the full reference was resized. Native reference hidden states are also updated by joint attention with the current target; they are not a fixed, identity-only feature bank.</p>
<p><b>Add calibration without changing the historical metric.</b> On a separate panel, score real same-person cross-view photos and different-person pairs under the same recognition/alignment procedure; stratify pose, expression and face size. Add a second recognition model, crop quality, and blinded side-by-side identity/expression review. A reference self-score of1 only verifies the prototype—it is not a real cross-view ceiling. No universal “good ID score” follows from these data.</p>
''')

page('Most plausible explanations, ranked by evidence',
table(['Explanation', 'Evidence and limit', 'Discriminating check'], [
    ['1. Denoising / identity objective mismatch', 'Confirmed architecture fact: face velocity MSE only. Training loss improves while pilot ID falls; full96 is neutral. Causality is not proven.', 'Matched flow-only versus flow + carefully aligned identity supervision, with a second recognizer and expression checks.'],
    ['2. Reference-only routing starts with a large quality penalty', 'Confirmed: zero deltas still replace joint attention; BA0 ID0.1544 versus native0.3816. Later gains include recovery from this change.', 'Keep native attention intact; strengthen reference features within the existing read. Test site placement separately, without mixing attention outputs.'],
    ['3. Pilot selection and scene mismatch', 'Measured: only two scene families, eight identities, one seed. Other84 show no 8k gain. Full96 2k is missing.', 'Score saved2k on full96 and compare paired identity/scene changes.'],
    ['4. Constant-LR drift / noisy updates', 'LR remains5e-5 after100 updates, batch1, no decay or EMA; effective adapter norms grow. No loss/gradient blow-up.', 'Fork from2k: matched2k further samples at5e-5 versus1e-5. Separately test EMA; keep Adam/RNG/order.'],
    ['5. Reference/detail and context mismatch', 'Reference-face support can be only63–96 tokens. Inference freezes a generated exterior while training sees noisy real scenes.', 'Independent ID token bank or additional face crop; separately test training context augmentation.'],
    ['6. Numerical failure or optimizer not updating', 'Disfavored:8,000 finite log rows,64 finite changed tensors, continued loss decrease, measured memory margin.', 'Retain existing parity/gradient/reload checks. No evidence warrants a numerical-repair project.']])+'''
<p><b>Not established:</b> global overfitting after2k, a universal2k stopping point, saturation of the512-key cap, or a need for a larger rank/backbone. Loss curves are averaged across changing examples and noise, so they do not measure a train/validation loss gap. The dataset has not completed a single pass, although imbalance, near-duplicates and objective-specific over-adaptation remain possible.</p>
<p><b>Data qualifications.</b> Import logic excludes held-out identities via aliases and exact reference-image hashes, rejects same-content target/reference pairs and pairs each Large target with the next available view. The matching Large cluster metadata audit reports47,341 pairs,110 identity-excluded pairs,49 same-content pair exclusions and zero validation-image overlaps. The local mirror contains2,557 identity labels. This is not a complete audit of aliases, near-duplicates or erroneous identity labels in the exact Vast sample stream. A stratified cross-view/label review is useful before scaling training.</p>
''')

page('Architecture changes to try first', '''
<p class="eyebrow">Proposals below are unimplemented and unmeasured.</p>
<div class="callout"><b>Design constraint:</b> preserve native attention, its projections, conditioning and RoPE. Improve what the existing reference read receives and learns. Do not add a residual native/reference mixture, a second attention-output fusion or a learned interpolation between those outputs.</div>
<h2>1. Put identity-discriminative features into the existing reference read</h2>
<p>The current branch sees appearance-rich VAE features, but receives no explicit recognition feature. Encode the supplied reference face with a frozen identity encoder; a small trainable projector produces, for example, 8–16 identity K/V tokens. Include these in the <b>same reference attention operation</b> as the current face K/V tokens:</p>
<div class="eq">K<sub>ref</sub> = concatenate(K<sub>face</sub>, K<sub>ID</sub>)<br>
V<sub>ref</sub> = concatenate(V<sub>face</sub>, V<sub>ID</sub>)<br>
R = Attention(Q′, K<sub>ref</sub>, V<sub>ref</sub>)</div>
<p>Keep the current query projection, native attention operator, branch output routing, eight sites and rank128. Native text/target/reference conditioning remains as it was. Only the reference bank and its small projector change. Identity tokens are independent of the noisy target, providing a stable signal across steps; there is one branch attention output, with no extra output blend.</p>
<p>Specify positional treatment explicitly: retain original coordinates for image tokens and use a separately defined nonspatial encoding for identity keys, without changing native RoPE. Match key/value scales and inspect attention mass to avoid the short ID bank dominating or being ignored. Added keys change softmax normalization even if their values start at zero, so <b>do not claim zero-init BA-on parity</b>. BA-off native parity remains required.</p>
<p>Compare against a fresh unchanged FLUX1 control at the same exposure; report added parameters/compute. Check branch-only identity-token swaps, expression/pose freedom and an independent recognizer. Risks are recognition-encoder bias, reference-expression copying and competition between detail and identity tokens.</p>
<h2>2. Improve face-detail tokens as a separate ablation</h2>
<p>Reference faces currently occupy only63–252 packed tokens; the512-key cap is inactive. Give the branch a higher-detail reference-face crop encoded by the existing frozen VAE, while leaving native full-image reference conditioning intact. Replace the branch’s face bank in this experiment; do not simultaneously add ID tokens or change the loss.</p>
<p>Record crop/context margins and an explicit coordinate mapping for its K/RoPE positions; preserve target positions and native attention. More crop pixels may recover eyes, nose and mouth detail, but cannot create resolution absent from the input. Measure reference swapping, output identity and pose/expression copying. Prefer this controlled input change to increasing the unused key cap or blindly raising rank.</p>
<p class="small">PuLID, InstantID and WithAnyone motivate recognition features and identity supervision; they do not establish this proposed single-reference-read implementation. The proposed K/V integration is our adaptation to the requested architecture constraint, not a reproduction of their fusion mechanisms.</p>
''')

page('Identity supervision, optimization and later ablations', '''
<h2>3. Add identity supervision while retaining native flow loss</h2>
<p>Under the current flow convention, form a clean-latent estimate ẑ<sub>0</sub>=z<sub>σ</sub>−σv<sub>θ</sub>. Decode a bounded fraction of training examples, align the predicted and real target face using <b>training-target landmarks</b>, and add a small identity cosine loss. Target landmarks are training supervision only; evaluation remains reference/prompt/native-mask conditioned.</p>
<div class="eq">L = L<sub>flow</sub> + λ<sub>ID</sub>[1−cos(E<sub>ID</sub>(aligned decoded ẑ<sub>0</sub>), E<sub>ID</sub>(target))]</div>
<p>Start at lower noise levels where the decoded estimate is usable, or use a short differentiable rollout on a bounded subset. There is no demonstrated four-step Base9B rollout quality here. Freeze identity-network weights but retain gradients through its generated-image input; the present NumPy/ONNX evaluation code cannot provide this training gradient. Log alignment/quality rejection rates and loss-gradient magnitudes before choosing λ. Keep the main native noise/timestep distribution unchanged.</p>
<p>Then consider an identity contrastive term against other training identities, with multiple positive views if available. Effective batch1 needs a detached bank of negative reference embeddings; it cannot supply in-batch negatives by itself. Evaluate with an independent recognizer and expression/pose diversity to avoid optimizing a single metric or copying the reference’s expression. These are separate stages so failure can be attributed.</p>
<h2>4. Small optimization controls before larger capacity</h2>
<p>Continue a copy of the2k state for the same next2,000 examples at constant5e-5 versus1e-5, preserving the original checkpoint, Adam moments, cursor and RNG. Record the LR change as a new experiment. Test adapter EMA separately (a candidate decay such as0.999 is a proposal), saving both raw and EMA adapters. A simple reduction in drift is cheaper to establish than a new9B capacity experiment.</p>
<p>If increasing accumulation, match total examples/noise draws and document the changed optimizer-update count. Batch8 for8k updates sees eight times the data of the present run and is not a clean stability comparison. Rank/site-count reductions can become later regularization/efficiency tests; larger rank is not the first response to33.6M branch parameters and a neutral full96 result.</p>
<h2>Later, only if the first tests support them</h2>
<p><b>Context alignment:</b> train with controlled corrupted or teacher-generated exterior context and the same face target/native velocity objective. <b>Site placement:</b> measure which existing sites affect identity versus expression; compare a later set of the same number of sites with the original set, preserving the attention operation and direct routing. <b>Spatial granularity:</b> test central face versus hair/accessory support. <b>Sampling:</b> use a separately named50-step/CFG study; the official Base9B example uses50 steps, versus20 here. Generate matching native scenes/masks for a changed sampler.</p>
<p>Do not combine a new loss, token bank, routing rule, sampler, rank and data recipe in the first follow-up. A successful combined model would leave the useful change unidentified.</p>
''')

page('A concrete next experiment sequence',
table(['Priority / name', 'Change and comparison', 'Decision it enables'], [
    ['0 · FLUX1_9B_2k_full96 — running', 'Saved2k adapter, exact full96 native bundle, prompts, seed0, owner boxes,20 steps, CFG4 and grouping. Same Comet as8k; log at step2000.', 'Whether2k actually wins broadly. No training required.'],
    ['1 · FLUX1_9B_protocol_controls', 'On a named stratified diagnostic panel, then full96: BA-off second pass; trained BA; branch-only shuffled/donor reference. Native conditioning stays fixed.', 'How much improvement comes from the branch and whether its reference features causally matter.'],
    ['2 · FLUX1_9B_lr_ablation', 'From copied2k state,5e-5 versus1e-5 for the identical next2k samples. Keep architecture and objective unchanged.', 'Whether the post-2k behavior is cheaply corrected by optimization.'],
    ['3 · FLUX1_9B_idloss', 'Existing native attention and FLUX1 architecture; add one identity-loss term with bounded decoding and a gradient audit.', 'Whether the objective is the bottleneck for identity fidelity.'],
    ['4 · FLUX1_9B_idkeys', 'Add reference identity K/V tokens inside the existing reference read; preserve native attention and current output routing.', 'Whether explicit identity features help without a new attention-output mixture.'],
    ['5 · FLUX1_9B_reference_detail', 'Change only the branch reference-face detail bank. Leave native reference conditioning, loss and output routing unchanged.', 'Whether insufficient reference detail limits the existing attention.']])+'''
<p><b>Budget and stopping:</b> use matched2k pilot training segments, with full96 at step0 and every2k optimizer updates. The historical one-ID schedule is a different exception. Bounded diagnostic panels must be named and include difficult scenes; do not replace the original96 benchmark. Use a second training seed for the most promising comparison and a separately locked, more diverse identity/reference/seed panel before a generalization claim.</p>
<p><b>Success should be paired and multi-dimensional:</b> primary owner-matched ID change by identity; no increase in detection/ownership failures; blinded identity and face-artifact review; expression/pose/accessory compliance and reference-copying checks. Preserve historical CLIP and ID definitions, adding supplementary metrics under distinct names. Choose meaningful acceptance margins before running, based on reference calibration and human review—not a borrowed universal cosine threshold.</p>
<p><b>Engineering admission remains necessary:</b> exact native BA-off parity, explicit token/mask/RoPE ownership, finite gradients and branch updates, measured peak reserved memory below90%, save/resume and serial validation on the patched backend. Benchmark added decoding/identity features before scheduling a long run.</p>
<p><b>Current checkpoint choice:</b> retain both2k and8k, with2k the provisional pilot selection. The8k model is a useful coherent face-refinement baseline and a useful full96 audit target, but neither an overall identity breakthrough nor evidence that this family should be abandoned. The immediate uncertainty is evaluation coverage, followed by route/objective design.</p>
<p class="small">Only priority0 has been launched, following the user’s request and GB10 restart. Other rows remain proposals. Supervisor rsrch_9b_2k_fixed96 runs inference→decode→scoring→Comet read-back; no training or machine shutdown is included.</p>
''')

sources = [
    ('IP-Adapter (2023)', 'https://arxiv.org/html/2308.06721v1', 'Separates image conditioning from the pretrained text path and adds image attention to it. That output-fusion mechanism is outside the requested follow-up direction. Its feature-resolution and fidelity/diversity observations remain useful background; it is not the recommended routing change.'),
    ('PuLID (NeurIPS 2024)', 'https://arxiv.org/html/2404.16022v2', 'Combines identity features with contrastive alignment and identity loss on a generated-image branch. It explicitly discusses unreliable identity supervision on noisy single-step estimates. Transfer the lesson about preserved native behavior and reliable supervision; its SDXL-Lightning training branch is not interchangeable with undistilled Klein Base9B.'),
    ('InstantID (2024)', 'https://arxiv.org/abs/2401.07519', 'Provides an independent example of identity-specific face features and spatial conditioning for tuning-free generation. It motivates an explicit identity bank as a separate experiment, without adopting reference landmarks as unrequested inference geometry.'),
    ('WithAnyone (2025 / ICLR 2026)', 'https://arxiv.org/html/2510.14975v1', 'Combines paired multi-view data, face-recognition and visual features, identity contrastive supervision, and target-landmark alignment for its training ID loss. Its benchmark separates identity fidelity from reference copy-paste. For FLUX1, test target-aligned ID supervision and a negative bank separately; do not compare its published cosine values directly with this owner-matched eight-identity panel.'),
    ('DynaIP (2025)', 'https://arxiv.org/html/2512.09814v1', 'Studies reference-specific versus reference-agnostic information in multimodal DiT conditioning, and hierarchical visual features. It motivates probing which blocks/timesteps carry identity versus pose/style and improving reference features. The paper does not validate our Klein mask routing or the proposed single-read identity bank.'),
    ('Official FLUX.2-klein Base9B card', 'https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B', 'Its example uses50 inference steps and CFG4 at1024px. Our measured protocol is768px/20 steps/CFG4. A longer-sampling comparison is reasonable but must be separately named with its own native masks; it does not retroactively invalidate the matched current baseline.')]

page('Research basis and limits of transfer', ''.join(f'<p><b>{link(url,name)}.</b> {text}</p>' for name,url,text in sources)+'''
<p>These are primary papers, official repositories/model documentation, checked4October2026. The proposed FLUX1 changes are inferences from their mechanisms and our implementation audit. None of these papers measures this exact architecture, fixed96 protocol or checkpoint trajectory; no published benchmark scores are presented as expected gains here.</p>
<p><b>Evidence levels used in this report:</b> “measured” refers to the saved run artifacts or the new CPU audit; “documented” refers to execution/admission notes not rerun here; “hypothesis” connects observations without establishing causality; “proposal” describes work not launched.</p>
''')

page('Reproducibility and variant boundaries', '''
<p><b>Authoritative Vast runs:</b> <code>flux9b_large_qkvo_r128_fast5h_b1</code> (0→2k), <code>flux9b_large_qkvo_r128_to8k</code> (full-state continuation), and <code>flux9b_8000_fixed96_20261004</code>. Historical paths and names remain unchanged.</p>
'''+table(['Record', 'Value'], [
    ['Canonical Comet', link('https://www.comet.com/nikolay-2104/rsrch-new/cfbd6f879e3d4158959bbbef7fb8e894','cfbd6f879e3d4158959bbbef7fb8e894')],
    ['Full96 Comet: 8k and new 2k', link('https://www.comet.com/nikolay-2104/rsrch-new/2b3eda1b1f364ecfaa584ffe273102a1','2b3eda1b1f364ecfaa584ffe273102a1')],
    ['Native12 / native96 Comet', '1acaa7707e184bad9176008e78f97ea6 /<br>6abec2cdaa104924a563e8b53f5a622c'],
    ['Frozen Vast runtime commit', '276fbf4e2fc8e75ec5eb38dd57044a4367587c35'],
    ['Pinned Toolkit commit', 'ecee894ed2b1f3716d9d7326693061ec1a3105bb'],
    ['9B weight revision', '32773329fbe7e81a90ef971740e8ba4b0364ecf3'],
    ['9B text encoder revision', 'b968826d9c46dd6066d109eabc6255188de91218'],
    ['VAE revision', '3f679cf232e6d91d28396522d9502c31e8f7ccbe'],
    ['2k adapter SHA256', 'eb27ca1c63780a50ae090cc65c746b6a7225<br>ea66e776400f1b5bd131606857fd'],
    ['8k adapter SHA256', '7cdc530e228ad76f310b8eb2f1aa9cdc6e<br>502a58691c626adade6e4be6e2ed60']])+'''
<p><b>Other FLUX1 variants:</b> local4B one-ID uses width3072,25,165,824 branch parameters and sites double1/2/3/4, single3/8/13/18. Its fixed24 result is documented separately. HSE-cluster4B uses selectiveFP16 and a separate frozen encoder GPU; the locally mirrored checkpoint is500 and the only scored panel available here is untrained step0 (full96 ID0.13005). This is not a live cluster-status report and supplies no trained4B-vs9B quality comparison. Older cached face heads and rank16 experiments are not FLUX1.</p>
<p><b>Artifacts in this report folder:</b> <code>analysis.json</code> has exact means, intervals, per-identity/prompt/size strata and audits. <code>per_image_comparison.csv</code> has all96 paired scores. <code>source_audit.json</code> hashes analyzed scores, manifests, source and the8k adapter. <code>evidence/</code> retains Comet source IDs/baseline CSVs plus CPU reference/checkpoint audit results. Charts and all-image contact sheets are separate files.</p>
<p>Rebuild with the existing report environment from the repository root:</p>
<div class="eq">envs/report/bin/python reports/261004_FLUX1_review/analyze.py<br>
envs/report/bin/python reports/261004_FLUX1_review/build_report.py</div>
<p class="small">The scripts read private photographs and saved outputs already in the workspace; Git alone does not contain those assets. No credentials are included. Bootstrap:100,000 draws of identity groups with replacement, RNG seed142; the primary mean weights all images in each sampled group. Intervals do not capture training-seed variation or best-checkpoint selection. Raw findings are descriptive on this fixed benchmark.</p>
''')

for i in (1,2):
    page(f'Visual appendix: pilot reference / native / 2k / 8k ({i}/2)', image(f'pilot_comparison_{i}.jpg',
        'All pilot rows, in their fixed order. Crops share the frozen pilot face box with context padding. Labels are original owner-matched pilot scores. Reference crops are supplied-input photographs, not model outputs.', width=362))
owners=list(A['per_identity'])
for i in range(0,len(owners),2):
    page('Visual appendix: all full96 pairs', ''.join(image(f'full96_{owner}.jpg',
        f'{owner}: native left, FLUX1 8k right in each pair; prompt order0–11. Scores are full96 owner-matched ID similarity. All cases included.',width=510) for owner in owners[i:i+2]))

CSS = '''
body { font-family: sans-serif; font-size: 10.1pt; line-height: 1.30; color: #21323f; }
h1 { font-size: 21pt; color: #076775; margin: 0 0 13pt; line-height: 1.14; }
h2 { font-size: 13pt; color: #076775; margin: 13pt 0 5pt; }
p { margin: 7pt 0; } b { color: #142b3b; }
.eyebrow { color: #677685; font-size: 10pt; }
.callout { padding: 11pt; border-left: 4pt solid #087f8c; background-color: #e8f4f3; margin: 10pt 0; }
.eq { font-family: sans-serif; padding: 9pt; background-color: #f0f4f7; font-size: 10pt; margin: 10pt 0; }
table { border-collapse: collapse; width: 100%; margin: 9pt 0; font-size: 9.5pt; }
th { text-align: left; background-color: #076775; color: white; padding: 6pt; }
td { padding: 5pt; border-bottom: 0.5pt solid #d6e0e5; vertical-align: top; }
tr:nth-child(even) { background-color: #f3f6f8; }
li { margin: 4pt 0; } ol { padding-left: 18pt; }
a { color: #087f8c; text-decoration: underline; }
.figure { margin: 11pt 0; text-align: center; }
.caption { font-size: 8.5pt; color: #5d6c78; text-align: left; margin-top: 5pt; }
.small { font-size: 9pt; color: #5d6c78; }
code { font-family: monospace; font-size: 8.3pt; }
'''


def build():
    website_css = CSS + '''
    body { max-width: 1000px; margin: 28px auto; padding: 0 22px; font-size: 16px; }
    section { padding: 26px 0 38px; border-bottom: 1px solid #d6e0e5; }
    img { max-width: 100%; width: 100%; height: auto; }
    table { font-size: 14px; } .caption, .small { font-size: 14px; }
    code { font-size: 13px; overflow-wrap: anywhere; } h1 { font-size: 30px; }
    @media print { section { break-before: page; } body { margin: 0; } }
    '''
    (OUT/'FLUX1_research_report.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>FLUX1 architecture and results review</title><style>'+website_css+'</style></head><body>'+''.join(PARTS)+'</body></html>')
    pdf=pymupdf.open()
    rect=pymupdf.Rect(0,0,595,842)
    box=pymupdf.Rect(42,52,553,795)
    section_pages=[]
    for n,part in enumerate(PARTS):
        story=pymupdf.Story(html=part,user_css=CSS,archive=pymupdf.Archive(str(OUT)))
        doc=story.write_with_links(lambda _, __: (rect,box,None))
        section_pages.append({'section':n+1,'pages':len(doc),'starts_at':len(pdf)+1})
        pdf.insert_pdf(doc);doc.close()
    for index,p in enumerate(pdf):
        p.insert_text((42,28),'FLUX1 | architecture and results review',fontsize=8,color=(.32,.40,.45))
        p.insert_text((42,818),'4 October 2026  |  measured results and explicitly marked proposals',fontsize=8,color=(.32,.40,.45))
        p.insert_text((516,818),f'{index+1}/{len(pdf)}',fontsize=8,color=(.32,.40,.45))
    pdf.set_metadata({'title':'FLUX1: architecture, results and next experiments','author':'rsrch_new research review','subject':'Vast9B 0–8k and fixed96 evaluation; evidence and proposed ablations'})
    pdf.save(OUT/'FLUX1_research_report.pdf',garbage=4,deflate=True)
    text='\n\n'.join(p.get_text() for p in pdf)
    (OUT/'extracted_text.txt').write_text(text)
    audit={'pages':len(pdf),'sections':section_pages,'text_characters':len(text),
           'links':sum(len(p.get_links()) for p in pdf),'tiny_pages':[i+1 for i,p in enumerate(pdf) if len(p.get_text().strip())<180],
           'page_text_bounds_violations':[]}
    for i,p in enumerate(pdf):
        for block in p.get_text('dict')['blocks']:
            if block['type'] != 0:continue
            for line in block['lines']:
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    if x0<0 or x1>596 or y0<0 or y1>843:audit['page_text_bounds_violations'].append({'page':i+1,'text':span['text']})
    (OUT/'document_validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    for i in [0,1,4,5,8,10,12,len(pdf)-1]:
        if i<len(pdf):pdf[i].get_pixmap(matrix=pymupdf.Matrix(1.3,1.3)).save(OUT/f'preview_{i+1:02d}.png')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    build()
