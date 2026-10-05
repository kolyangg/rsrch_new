"""Build the detailed, source-linked Flux 2 proposal as PDF and HTML.

Run analyze_and_figures.py first, then run this with envs/report/bin/python.
All new architectures, losses, parameter budgets and profiles are proposals.
"""
from pathlib import Path
import base64
import hashlib
import html
import json
import re

import pymupdf

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
A = json.loads((OUT / 'analysis.json').read_text())
PAGES = []
SOURCES = {
    'R1': ('AnyPhoto', '16 March 2026; arXiv v1', 'https://arxiv.org/html/2603.14770v1'),
    'R2': ('WithAnyone', '16 October 2025; arXiv v1; ICLR 2026 project', 'https://arxiv.org/html/2510.14975v1'),
    'R3': ('InfiniteYou', '20 March 2025; ICCV 2025', 'https://arxiv.org/html/2503.16418v1'),
    'R4': ('DreamO', '2025; arXiv v3; SIGGRAPH Asia 2025', 'https://arxiv.org/html/2504.16915v3'),
    'R5': ('IdGlow', '28 February 2026; revised 8 May 2026, v2', 'https://arxiv.org/html/2603.00607v2'),
    'R6': ('SpatialID: Inject Where It Matters', 'February 2026; arXiv v1', 'https://arxiv.org/html/2602.13994v1'),
    'R7': ('RefDiT', '4 September 2026; arXiv v1', 'https://arxiv.org/html/2609.04976v1'),
    'R8': ('ReGain', '30 September 2026; arXiv v1', 'https://arxiv.org/html/2609.38680v1'),
    'R9': ('PuLID', '2024; NeurIPS 2024', 'https://arxiv.org/abs/2404.16022'),
    'R10': ('IP-Adapter', '2023; decoupled cross-attention', 'https://arxiv.org/abs/2308.06721'),
    'R11': ('DINOv2 official implementation', 'Frozen visual-feature encoder candidate; accessed 5 October 2026', 'https://github.com/facebookresearch/dinov2'),
}

def cite(key):
    name, _, url = SOURCES[key]
    return f'<a href="{url}">[{key}] {name}</a>'

def table(headers, rows):
    return '<table><tr>' + ''.join(f'<th>{s}</th>' for s in headers) + '</tr>' + ''.join(
        '<tr>' + ''.join(f'<td>{s}</td>' for s in row) + '</tr>' for row in rows) + '</table>'

def fig(name, caption='', width=510):
    return f'<img src="figures/{name}.png" width="{width}">' + (f'<p class="caption">{caption}</p>' if caption else '')

def page(title, body):
    PAGES.append((title, f'<h1>{title}</h1>{body}'))

def eq(text):
    return '<div class="equation">' + text + '</div>'

def callout(text):
    return '<div class="callout">' + text + '</div>'

page('Flux 2: give the reference lasting control', '''
<p class="eyebrow">Architecture review and experimental design · 5 October 2026</p>
''' + callout('''<b>Recommendation:</b> build Flux 2 around a persistent face stream, immutable identity memory, and a one-way scene-context boundary. Add identity conditioning at every block and strengthen the learning signal with calibrated identity and contrastive losses. Retain a full 4B version for local tests and a crop-based 16GB fallback.''') + '''
<p>Flux 1A solved a real information-flow problem: its reference bank no longer reads target or text states. It did <i>not</i> give that bank control of the complete face-generation computation. Native query features, residual state, native MLPs, timestep gates and 24 unmodified blocks remain. The next design should address these routes without discarding the pretrained generator's ability to produce a natural face.</p>
<p><b>The evidence does not establish a remaining reference-bank leak.</b> At 4K, Flux 1A improves owner-ID by only +0.0201 over native; at 6K, the gain is +0.0104. The best historical Flux 1 fixed96 score is higher. Similarity to the native face can reflect a weak identity signal, small spatial support, a shared generative prior, or scene-conditioned geometry. These explanations need distinct tests.</p>
''' + table(['Decision', 'Recommended design'], [
    ['Large GPU', '<b>Flux 2-9B:</b> full 768px scene protocol, identity/detail adapters at all 32 blocks, read-only sanitized context and face-specific modulation. First target: 80GB or a GB10-class host, subject to admission.'],
    ['16GB local GPU', '<b>Flux 2-Lite-4B:</b> the same information-flow rules at all 25 blocks, narrow adapters, activation checkpointing and staged encoders. <b>Flux 2-Lite-ROI384</b> reduces active spatial tokens and decoder cost when needed.'],
    ['First scientific question', 'Does changing only the reference cause the owned face to follow that identity, while an erased native-face perturbation has no effect?'],
    ['Evidence standard', 'Beat native and Flux 1A 4K on the original fixed96; also compare the historical Flux 1 2K result. Require face quality, expression and independent-recognizer checks.']]) + '''
<p><b>Status:</b> research and design only. Flux 2 is not implemented, trained or benchmarked. Hardware fit and quality gains are hypotheses. This report recomputes existing scores and mask geometry on CPU; it does not launch training or change existing model code.</p>
<p class="small">Naming: Flux 1, Flux 1A and Flux 2 are project experiment names. The underlying generator is BFL FLUX.2-klein Base 4B/9B. Flux 2 here is not a new BFL model release. The HSE 4B run is separate from the completed Vast9B run reviewed here.</p>
''')

metrics = A['metrics']
page('What the completed 6K run actually shows', table(
    ['Model / checkpoint', 'Owner-ID ↑', 'CLIP logits ↑', 'Interpretation'],
    [['Native fixed96', f"{metrics['native']['id_sim']:.4f}", f"{metrics['native']['text_sim']:.3f}", 'Frozen comparison scene'],
     ['Flux 1A, step 0', f"{metrics['0']['id_sim']:.4f}", f"{metrics['0']['text_sim']:.3f}", 'Active, untrained replacement branch'],
     ['Flux 1A, 2K', f"{metrics['2000']['id_sim']:.4f}", f"{metrics['2000']['text_sim']:.3f}", '96 images'],
     ['Flux 1A, 4K', f"{metrics['4000']['id_sim']:.4f}", f"{metrics['4000']['text_sim']:.3f}", 'Best measured Flux 1A ID'],
     ['Flux 1A, 6K', f"{metrics['6000']['id_sim']:.4f}", f"{metrics['6000']['text_sim']:.3f}", 'Completed 6K training and validation'],
     ['Historical Flux 1, 2K', '0.3174', 'See original run', 'Same fixed96 metric; different training recipe']]) +
    fig('measured_trajectory', 'Measured fixed96 results. Step 0 is omitted from the plot to make the small trained differences readable; its score remains in the table.') + '''
<p>Flux 1A 4K exceeds native by <b>0.02014</b> (7.34% relative), with 61/96 image wins but only 3/8 identity-mean wins. At 6K the gain is <b>0.01040</b> (3.79%), with 55/96 image wins and 6/8 identity wins. Identity-group bootstrap intervals are respectively [−0.00329, +0.04648] and [−0.00401, +0.02658].</p>
<p>The 4K→6K change is −0.00973; its grouped interval is [−0.02621, +0.00587]. This is evidence against assuming that more steps alone will fix identity, but it does not prove overfitting or a specific failure mechanism. One training seed and eight identities provide limited precision.</p>
<p class="small">Recomputed from the saved per-image CSVs, with all 96 sample IDs/order checked. Bootstrap: 100,000 resamples of eight identity groups, seed 142. Intervals omit training-seed variability and checkpoint-selection uncertainty. The large 0→4K gain mainly includes recovery from a disruptive untrained path.</p>
''')

page('Where the aggregate score hides the difficulty',
    fig('identity_and_resolution', 'Left: observed per-identity change, not separate independent trials. Right: native owner-box geometry from the unchanged routing-mask manifest.') + '''
<p><b>Small faces are a concrete bottleneck.</b> The native owner-box short side ranges from 50 to 245 pixels, with quartiles 77 / 105 / 123.5 pixels. In 59/96 images it is below 112 pixels. The median face spans approximately <b>6.56 native 16px tokens</b> across its short dimension. A 112px recognition crop enlarges many of these faces; it cannot recover details absent from generation.</p>
<p>The median face box occupies only 2.47% of the full image. Full-frame CLIP barely moves because almost all scene pixels are explicitly preserved. Good body cohesion is useful, but the unchanged exterior and inherited face position explain part of it; it is not an independent demonstration of learned integration.</p>
<p><b>Metric ownership matters.</b> The primary number is the existing owner-matched ArcFace cosine against the subject-aligned reference embedding. No-face, unowned, ambiguous and missing-mask rates at 4K/6K are all zero. Mean detected faces are 2.875 per image. Mean owner-box IoU changes from 0.8887 at 4K to 0.8628 at 6K; this is a useful localization diagnostic, not proof that movement caused the ID decline.</p>
<p>The earlier metric audit found a wrong-person legacy embedding for the reference labeled eddie; the corrected subject-v2 embedding is already used by these primary scores. Do not quietly switch between legacy best-face and owner-ID results. Keep the legacy metric as a separately labeled historical column.</p>
<p><b>Implication for Flux 2:</b> identity memory and loss can improve semantic control, while an explicit high-resolution face window can improve spatial capacity. They are different interventions and should be tested separately.</p>
''')

page('Visual evidence: coherent changes, incomplete identity', '''
<p>These two panels are copied unchanged from the existing report: sample 00 and sample 65 from its deterministic identity-index/prompt-index selection. They are illustrative, not selected as quantitative extremes. No new images were generated.</p>
<img src="evidence/sample_00.jpg" width="510">
<p class="caption">Sample 00. Native and trained faces retain related geometry and scene lighting. The 4K score increases slightly, then declines at 6K. Appearance alone cannot identify which internal route caused this.</p>
<img src="evidence/sample_65.jpg" width="510">
<p class="caption">Sample 65. Identity score improves considerably while the overall pose, expression and scene remain similar. Therefore “looks different from native” is not a valid objective by itself.</p>
<p><b>Desired behavior:</b> change the person's identity toward the supplied reference while retaining prompt-driven pose, expression, lighting and body integration. Never reward arbitrary dissimilarity from the native face; native may already match the requested identity well.</p>
''')

page('Flux 1A: exact computational structure',
    fig('flux1a_information_flow') + table(['Item', 'Completed Vast9B implementation'], [
    ['Backbone / sites', 'Width 4096, 32 heads, 8 double + 24 single blocks. BA at double {2,4,6,7}, single {4,10,16,22}, zero based.'],
    ['Trainable tensors', 'Q/K/V/output low-rank deltas, rank=alpha=128: 8 × 8 × 4096 × 128 = 33,554,432 parameters in 64 tensors. All native weights frozen.'],
    ['Reference bank', 'The clean reference latent is passed through the frozen model with zero text/target tokens. Current σ conditions this pass; the reference itself is not newly noised. Pre-normalization K/V and input hidden states are captured at BA sites.'],
    ['Attention', 'Target Q comes from the live native target features plus ΔQ. Reference K/V come from the independent bank plus ΔK/ΔV. Native QK norm and original target/reference RoPE are retained. At most 512 face keys are selected.'],
    ['Spatial routing', 'Binary latent support M = 1[AvgPool16(pixel alpha) > 0]. Owned attention rows are replaced at eight sites. Native joint reference conditioning remains in the main stream.']]) + '''
<p class="small">Source basis: original deployment snapshot plus unchanged patch/runtime files, not an assumption that today's working tree matches the run. The local isolated-bank and identity-loss files have subsequently diverged; the archived versions are included under evidence/deployed_sources.</p>
''')

page('Flux 1A: equations and routes that remain', eq('''
Q<sub>F</sub> = norm(Q<sub>native</sub>(h<sub>F</sub>) + ΔQ(h<sub>F</sub>)); apply target RoPE<br>
K<sub>R</sub> = norm(K<sub>isolated</sub> + ΔK(h<sub>R</sub>)); apply reference RoPE<br>
A<sub>R</sub> = softmax(Q<sub>F</sub>K<sub>R</sub><sup>T</sup>/√d) · (V<sub>isolated</sub> + ΔV(h<sub>R</sub>))<br>
A<sub>routed</sub> = (1 − M) A<sub>native</sub> + M A<sub>R</sub><br>
h′<sub>F</sub> = h<sub>F</sub> + g<sub>native</sub> · [W<sub>O</sub>A<sub>R</sub> + ΔO(A<sub>R</sub>)]
''') + '''
<p>The last line shows the double-stream attention residual on an owned token; its normal native MLP residual follows. A single-stream block instead jointly projects attention and its parallel MLP features through linear2, adds the output delta, then applies the native gate. In either case, the original face residual remains.</p>
''' + table(['Route', 'What is established', 'Consequence'], [
    ['Target → isolated bank', 'Removed by construction. Its inputs contain no target/prompt states.', 'Further detaching the bank is not a new solution.'],
    ['Target state → branch Q', 'Direct, at every selected BA site.', 'Reference retrieval is steered by the current face representation.'],
    ['Native residual / MLP / other blocks', 'All retained; 24/32 blocks have no BA.', 'A reference-only attention message is not exclusive face-state ownership.'],
    ['Native gate on branch output', 'Both native projection and ΔO pass through the same native modulation gate.', 'Some identity messages may be attenuated. Gate strength has not been measured across the run.'],
    ['Joint text / exterior / raw reference tokens', 'Interact in native attention. Target information can enter them and return later.', 'No all-layer directional boundary exists around the face stream.'],
    ['Frozen native scene', 'Unowned latents are clamped to the noised native scene; RGB exterior is restored.', 'Nearby scene latents and fixed mask geometry can constrain the face.']]) + '''
<p><b>Important distinction:</b> the live target face is generated from fresh noise; it is not simply the saved native face copied into an interior residual. Native-looking output is not proof of copying. Likewise, frozen weights still contain a generative face prior. Flux 2 can remove data paths and give reference conditioning more authority; it cannot remove that prior merely by renaming a stream.</p>
<p class="small">Double-block image indices are target [0,Nt), reference [Nt,Nt+Nr). Single-block indices add the text offset. At 768px, Nt=48×48=2304; full native reference tokens may be 1024 even when the BA read is capped at 512.</p>
''')

page('Ranked diagnosis: what is known and what is not', table(
    ['Priority / hypothesis', 'Supporting evidence', 'Discriminating test'], [
    ['1. Sparse authority and continued native processing', 'Only 8/32 attention outputs change; residuals, MLPs and native gates remain.', 'Compare 8-site and all-layer identity conditioning at matched capacity and compute. Inspect donor-reference response.'],
    ['2. Identity supervision is weak or poorly timed', 'Admission weighted-ID/flow gradient ratio is 0.0444; identity activates only at σ≤0.5.', 'Log gradient ratios and activation fraction by σ over actual training; sweep calibrated weights with the flow sampler fixed.'],
    ['3. Reference representation mismatch', 'Image-only, empty-text native pass; native Q/RoPE reused across nonaligned face geometry.', 'Compare existing bank with direct recognition + crop detail memory, then test dedicated reference projections.'],
    ['4. Insufficient face detail', 'Median short side 105px; 59/96 below 112px.', 'Run the separately named ROI-resolution variant with paired outputs and identical final canvas.'],
    ['5. Context and training/inference mismatch', 'Training denoises a real photo; inference clamps a generated exterior. Global streams can relay face information.', 'All-layer one-way context plus RGB redaction; train-only context robustness ablation.'],
    ['6. Data identity ambiguity / narrow target loss', 'Accepted target labels do not establish correct cross-view reference/target pairing.', 'Audit actual pair provenance and identity agreement; use reliable other-ID negatives and alternate same-ID references.'],
    ['7. Capacity or optimization', 'Rank 128 already gives 33.6M parameters; later loss reduction did not ensure higher owner-ID.', 'Rank increase only after reference causality improves. Keep lower-rank controls and fresh-seed reruns.']]) + '''
<p><b>The gradient clue is not a diagnosis.</b> At admission, weighted identity gradient norm is 0.002956 versus flow 0.066534. That 4.44% ratio comes from one test point, not the 6K average. Gradient direction also matters: a small gradient can help, and a larger one can damage realism.</p>
''' + eq('''ẑ₀ = zσ − σ vθ &nbsp;&nbsp; ⇒ &nbsp;&nbsp; ∂L<sub>ID</sub>/∂vθ = −σ · ∂L<sub>ID</sub>/∂ẑ₀''') + '''
<p>Very low-noise estimates receive an additional σ factor on the identity-to-velocity gradient. High-noise one-step estimates can be unreliable. The useful supervision window must be measured. Do not simply divide by σ near zero or assume that σ≤0.5 means half of the native nonuniform schedule is supervised.</p>
<p>These hypotheses can coexist. The recommended design addresses the first four, while the ablation sequence prevents an expensive all-at-once change from becoming uninterpretable.</p>
''')

page('Recent research: directly useful mechanisms', table(
    ['Primary source', 'Mechanism worth considering', 'What transfers to this project'], [
    [cite('R1'), 'Persistent identity-adaptive modulation, location-aligned tokens, and restricted multi-ID attention.', 'Add face-only identity modulation. Its star mask still allows shared image/text relays, so do not treat it as strict target/reference isolation.'],
    [cite('R2'), 'Paired different-image supervision, target-aligned ID loss and contrastive identity loss with extended negatives.', 'Use a verified negative embedding bank at microbatch 1. Keep cross-view positives. Its published gains do not predict this fixed96 score.'],
    [cite('R3'), 'A trainable InfuseNet injects identity through residual connections; staged training includes diverse images of the same person.', 'Provide identity an independent residual route. A full auxiliary denoiser is costly; use small cross-attention modules first.'],
    [cite('R4'), 'Routing supervision links reference conditions and spatial regions in a DiT.', 'Audit where identity reads land. Explicit ownership is already available here, so learned routing is secondary to correct information flow.'],
    [cite('R5'), 'Task-specific timestep weighting of identity constraints, with a later preference-optimization stage.', 'Measure noise-dependent identity gradients. Its timestep conventions and group-generation matching do not replace our owner-matched evaluation.'],
    [cite('R6'), 'SpatialID adapts where and when PuLID-style identity features are injected.', 'Schedule identity/detail strength within owned support. Do not import its late global mask floor: our exterior must remain preserved.']]) + '''
<p><b>Synthesis:</b> use explicit face recognition features for identity, a separate appearance channel for detail, recurrent identity modulation and discriminative training. Attention connectivity must be audited across all layers; a single direct mask does not rule out a two-hop relay through shared tokens.</p>
<p class="small">Search cutoff: 5 October 2026. Sources were opened on arXiv, official project repositories or publisher pages. Recent arXiv proposals are treated as research evidence, not independently reproduced improvements. Most use FLUX.1 or other generators, not this pinned Klein runtime.</p>
''')

page('Newest work and limits of transfer', table(
    ['Source', 'Relevant lesson', 'Decision'], [
    [cite('R7') + '<br>4 September 2026', 'Reference attributes can be represented locally instead of being entangled in one global identifier.', 'Supports separating identity from appearance/attributes. This paper studies local attribute transfer; it is not direct evidence of face-ID gains on Klein.'],
    [cite('R8') + '<br>30 September 2026', 'Synthetic-image personalization can inflate CFG differences, particularly at high frequencies; a sampling correction is proposed.', 'Audit data provenance and conditional/unconditional velocity norms. Do not import its frequency correction or published gains as a Flux 2 result.'],
    [cite('R9'), 'Combines identity supervision with alignment and a fast generation branch to preserve fidelity while limiting disruption.', 'Short-rollout identity supervision is a later option when one-step estimates are unreliable. It changes compute and needs its own comparison.'],
    [cite('R10'), 'Separates image and text cross-attention instead of forcing them to compete in one softmax.', 'Use independent identity/detail reads. Isolation of reference inputs still needs explicit enforcement.'],
    [cite('R11'), 'Official frozen vision backbone family with small-model options and patch features.', 'Candidate for face-crop detail memory; it is not a face-ID metric. Exact variant/revision/preprocessing must be locked before implementation.']]) + '''
<p>The latest papers broaden the design space, but recency is not a reason to add every mechanism. The initial Flux 2 should not add DPO, a second full trainable denoiser, learned spatial routing, frequency fusion and a new sampler simultaneously.</p>
<p><b>CL39 relationship:</b> retain the useful project principle of a separately controlled reference message and explicit spatial ownership. The earlier reports distinguish CL39's projected reference-minus-native/frequency route from this project's evolving FLUX variants. Flux 2 does not claim to reproduce CL39 or use its results as a benchmark. Its new contribution is complete state provenance plus persistent identity conditioning.</p>
<p><b>What not to conclude from paper leaderboards:</b> published identity cosines, CLIP values and “copy-paste” scores use different datasets, alignment, face assignment and recognition systems. They cannot be plotted beside our 0.2943 as if they were the same metric.</p>
<p class="small">The report's Flux 2 equations, connectivity rules, parameter budgets and thresholds below are proposed engineering choices. They are not claimed as the exact architectures or validated hyperparameters of the cited papers.</p>
''')

page('Flux 2: whole-model design',
    fig('flux2_whole_model') + '''
<p><b>Pass A:</b> keep the frozen native scene generation and its saved mask/image provenance. This gives the requested composition, body and background. Reuse the historical native bundle for controlled comparisons.</p>
<p><b>Pass B:</b> denoise the owned face with a persistent face stream F. It uses the pretrained backbone's weights and velocity head, but has its own hidden state throughout the complete depth. There is no parallel native-face hidden state to copy into F. Identity and detail memory are read at every block.</p>
<p><b>Context:</b> erase the owned native face and an additional halo in RGB before encoding the scene used as context. Read-only context C and text T cannot receive F updates. Never obtain C by taking “non-face rows” from a native hidden sequence that already attended to the native face.</p>
<p><b>Output:</b> write F's predicted velocity only to owned latent positions, integrate the same Euler trajectory, decode and retain the established final exterior compositor. Face-box interior has generated RGB weight one. An independent native BA-off path remains available for exact parity.</p>
<p class="small">The new BA-on topology and conditioning intentionally differ from Flux 1A; this is a fresh named experiment. Do not describe it as preserving native attention on the active path. Weight sharing means one frozen weight set, not a second 9B model resident in memory.</p>
''')

page('An all-layer information boundary',
    fig('flux2_attention_firewall', 'Rows are readers; columns are permitted sources. Native face N is excluded from the trainable path. R is encoded independently and never updated from denoising states.') + '''
<p><b>Indices:</b> maintain original target-grid positions I={0,…,Nt−1}. Let F be positions owned by the unchanged binary support M. Let D be an RGB exclusion region covering the owner box, its output feather and a proposed extra 32px halo. C contains only target-grid cells outside D; the halo gap is omitted from context reads. Text has its original positions. Reference-memory tokens use a separate cross-attention namespace.</p>
<p><b>Connectivity at every block:</b> T reads T; C reads C and T; F reads F, C and T through native projections, then separately reads identity and detail memory. No F→C, F→T, target→R, or native-face→F edge exists. This also prevents the F→text/context→F relay. C and T can be computed under no_grad because they do not depend on trainable F updates; F must retain full autograd.</p>
<p><b>Why erasure must precede the VAE:</b> an exterior latent can contain information from nearby face pixels. Masking VAE latents or late hidden rows is weaker than encoding an already erased image. The halo size is a hypothesis, not a proof of the VAE's receptive field. Never feed the original native clean latent into face inference or its decode assembly; use the sanitized encoding there.</p>
<p><b>Precise invariant:</b> with mask, seed, reference, text and all pixels outside D fixed, changing only native pixels inside D must not change R, C, F or face-interior output. Rebuild sanitized context during this intervention. This establishes conditional invariance to erased pixels; pose boxes, hair and visible body outside D can still carry correlated identity information.</p>
''')

page('Inside one Flux 2 block',
    fig('flux2_block') + eq('''
u = (1 + s<sub>t</sub> + δs<sub>ID</sub>) LN(F) + b<sub>t</sub> + δb<sub>ID</sub><br>
A<sub>scene</sub> = Attn(Q<sub>native</sub>(u), K<sub>native</sub>(F,C,T), V<sub>native</sub>(F,C,T))<br>
A<sub>ID</sub> = Attn(Q<sub>b</sub>(u), K<sub>ID</sub>(R<sub>ID</sub>), V<sub>ID</sub>(R<sub>ID</sub>))<br>
A<sub>detail</sub> = Attn(Q<sub>b</sub>(u), K<sub>detail</sub>(R<sub>detail</sub>), V<sub>detail</sub>(R<sub>detail</sub>))<br>
F* = F + (g<sub>t</sub> + δg<sub>ID</sub>) W<sub>native</sub>A<sub>scene</sub><br>
&nbsp;&nbsp;&nbsp;&nbsp; + a<sub>ℓ,σ</sub> O<sub>b</sub>A<sub>ID</sub> + b<sub>ℓ,σ</sub> O<sub>b</sub>A<sub>detail</sub>
''') + '''
<p>Native QK normalization and RoPE are retained for the scene read, with native layer-specific modulated inputs for every group. The added reads use their own projections, head normalization and softmax; there is no text/exterior competition in the identity softmax. Their residuals enter outside the native attention gate.</p>
<p>In double blocks, follow F* with the native MLP using identity-adjusted MLP shift/scale/gate. In single blocks, retain the native joint attention/MLP projection and add the two new residuals afterward; identity adjusts that block's shared modulation triplet. The equations are schematic for the double-block case, not a replacement for the single-block implementation.</p>
<p class="small">Initialize modulation offsets and O<sub>b</sub> to zero, keep Q/K/V nonzero and initialize bounded read gates nonzero (proposed ID 1.0, detail 0.5). Zeroing both O and the gate would create a dead gradient path. This starts from the firewall baseline, not from native parity; only the explicit BA-off dispatch guarantees native equality.</p>
''')

page('Reference memory, positions and spatial authority', table(
    ['Component', 'Concrete initial specification'], [
    ['Identity memory', 'Frozen reference-face ArcFace embedding (512 dimensions), normalized; a trainable projector creates four identity tokens. Only the reference face supplies these inputs. No target embedding is ever conditioning.'],
    ['Detail memory', 'Frozen DINOv2 ViT-S/14 on a reference-only face crop, proposed 224px input. Retain the crop patch grid, pool to 8×8 = 64 tokens, then project. Suppress unrelated reference background.'],
    ['Two channels', 'Separate ID and detail softmaxes and K/V parameters; shared branch Q and output projection per block. Detail dropout (initial candidate 0.2) tests whether the model still obeys identity without copying expression/accessories.'],
    ['Modulation', 'One shared low-dimensional identity bottleneck, then per-block projections into native shift/scale/gate offsets. Applied only to F. This gives the reference access to MLP behavior as well as attention.'],
    ['Spatial positions', 'Keep native scene RoPE coordinates for F/C/T. Dedicated ID attention has no absolute scene/reference offset. Detail tokens carry learned normalized crop-grid position embeddings; do not pretend reference and target eyes share scene coordinates.'],
    ['Layer and time coverage', 'All 32 blocks in 9B; all 25 in 4B. Begin with time-constant strengths. Later test stronger early/mid identity and weaker late detail, with σ=1 meaning noise and σ=0 clean.'],
    ['Reference coverage', 'Original reference and selection stay fixed for the primary panel. A higher-resolution reference crop is a separately named preprocessing ablation, even if the output resolution is unchanged.']]) + '''
<p><b>Initialization preserves useful computation.</b> Flux 1A replaced a complete attention message with a reference-only read even at zero adapter weights, producing the poor step-0 score. Flux 2 retains within-face and context attention and adds identity residuals, reducing that particular disruption. The new directional topology is still a distribution shift and must be evaluated at step 0.</p>
<p><b>Geometry versus identity:</b> use the existing owner box for placement; avoid dense native facial landmarks or native low-frequency face pixels as compulsory conditioning, because these can fix the very facial proportions we want to change. A coarse head-pose condition is optional and must be tested as an explicit relaxation of isolation.</p>
<p><b>State ownership is the key change.</b> A second reference K/V bank alone would repeat Flux 1A. F must remain a distinct state with reference influence in its attention, modulation and residual trajectory across the complete depth. Frozen encoder features may be cached; trainable memory projectors must remain differentiable and be recomputed after optimizer updates.</p>
''')

page('Training: preserve flow, strengthen identity learning',
    fig('training_objectives') + eq('''
L = L<sub>flow,M</sub> + λ<sub>ID</sub> w(σ) [1 − cos(e<sub>generated</sub>, e<sub>target</sub>)] + λ<sub>ctr</sub> w(σ) L<sub>NCE</sub><br>
L<sub>NCE</sub> = −log { exp(sim(e<sub>g</sub>,e<sub>positive</sub>)/τ) /<br>
&nbsp;&nbsp;&nbsp;&nbsp; [exp(sim(e<sub>g</sub>,e<sub>positive</sub>)/τ) + Σ<sub>j</sub> exp(sim(e<sub>g</sub>,e<sub>negative,j</sub>)/τ)] }
''') + '''
<p><b>Keep the native flow objective.</b> Use fresh noise, the pinned native timestep sampler, velocity ε−z₀ and the same face-normalized MSE. Keep the same seeded training-row order for causal controls. Training-target masks and landmarks supervise training only; no target photo or target mask is loaded during validation.</p>
<p><b>Identity objective:</b> keep differentiable frozen VAE decoding and deterministic target-landmark alignment. Start with λID=0.05 as a control, then calibrate 0.1/0.2 on a training-only calibration set. Record weighted-ID/flow gradient norms and cosine on shared adapter parameters by σ bin. A proposed median ratio of 0.2–0.5 is a tuning target, not an optimum or a reason to amplify unstable gradients.</p>
<p><b>Contrastive objective:</b> propose τ=0.1, λctr=0.05 and 1,024 stored other-identity embeddings (about 2MiB FP32). Include the positive in the denominator. Exclude same-person aliases and near duplicates. Reliable cross-identity labels are a prerequisite; if absent, audit/build identity groups before enabling this term. Batch accumulation does not itself create negatives.</p>
<p class="small">Target-aligned supervision and extended identity negatives are motivated by WithAnyone [R2]. The exact formulas and starting coefficients here are our proposed adaptation. Microbatch-1 feasibility comes from the frozen embedding bank, not from pretending there are large in-batch negative sets.</p>
''')

page('Noise coverage, data and training/inference alignment', '''
<p><b>Noise coverage:</b> initially retain the Flux 1A σ≤0.5 gate while measuring effectiveness. Then test a smooth auxiliary window over σ∈[0.15,0.75], with reliability checks on decoded one-step estimates. Do not resample the flow timestep distribution in only one arm. If more auxiliary samples are needed, add a separately drawn identity forward after the same flow step and report the extra compute.</p>
<p>A later large-GPU experiment can differentiate through 2–4 additional denoising steps for identity, using checkpointing. This reduces reliance on a poor one-step estimate but raises memory and wall time. It is not required for the first 16GB version. Training a recognizer on corrupted inputs or introducing a distilled identity teacher would be separate research, not an assumed free replacement.</p>
''' + table(['Training issue', 'Proposed handling'], [
    ['Cross-view pairs', 'Preserve actual target/reference pairs initially; audit whether they are the same person and distinct images. Later replace a reference only with another verified image of that same identity, under a new data manifest.'],
    ['Target signal at low noise', 'Noised target-face tokens are necessary flow inputs, not a bank leak. Use reference swaps and high/mid-noise diagnostics to determine whether conditional identity is learned rather than merely reconstructed.'],
    ['Context sanitation', 'At training, erase the target owner face and halo before separately encoding C. F still uses the ordinary noisy target latent and the original masked flow target. This removes the known face from the context route while retaining correct supervised denoising.'],
    ['Real versus generated context', 'Training C comes from a photo; inference C from the native generation. First log the mismatch. Later train-only context dropout/color perturbation may improve robustness. Synthetic-context training needs its own provenance and matching audit.'],
    ['Invalid labels', 'Keep every flow row; skip only invalid identity terms. Report accepted reference and target label fractions separately. The prior 98.5% target-label acceptance says nothing by itself about pair identity correctness.'],
    ['Alignment drift', 'Fixed target landmarks avoid a detector on noisy outputs but can miss displaced faces. Inspect owner overlap and aligned crops; use deterministic bounded crop jitter only as a named loss ablation.'],
    ['Recognition overfitting', 'Never optimize the unchanged validation scores or mine negatives from validation. Evaluate a second frozen recognizer not used for conditioning/loss, and conduct blinded crop and full-image review.']]) + '''
<p><b>No native-identity repulsion loss.</b> A native face can already be correct. “Different from native” and “higher identity cosine” are not equivalent. Negatives must be verified different identities, not automatically the native image or a difficult same-ID view.</p>
''')

page('Large-GPU profiles and parameter budgets',
    fig('hardware_profiles') + table(['Profile', 'Core configuration', 'Approximate new parameter budget'], [
    ['Flux 2-9B', '768 target / reference cap 512; 32 sites; d=512; modulation bottleneck r=128; microbatch 1, accumulation 1.', '230.69M core + identity/detail projectors, biases and gates; budget ≤240M.'],
    ['Flux 2-4B-48G', 'Same output/reference protocol; 25 sites; d=256; modulation r=64; microbatch 1, accumulation 1.', '63.57M core + projectors; budget ≤70M.'],
    ['Flux 2-Lite-4B', 'Same architecture and 768 output; 25 sites; d=128; modulation r=32; local smoke initially accumulation 4.', '30.15M core + projectors; budget ≤35M.']]) + '''
<p><b>Transparent arithmetic:</b> each site has Q:D→d, O:d→D and separate identity/detail K,V:d→d, giving 2Dd+4d² weights. Modulation adds Dr(6n_double+3n_single). The core count excludes small projectors/biases/gates and frozen encoders. Exact tensor inventory is an implementation admission requirement.</p>
<p>At a conservative 16 bytes per trainable parameter for weight/gradient/optimizer/master state, the ≤240M and ≤35M budgets cost about 3.58GiB and 0.52GiB. These are parameter-state estimates, <b>not total VRAM predictions</b>. Activations, attention kernels, VAE backward, context K/V retention and allocator overhead can dominate.</p>
<p class="small">Start fresh adapters on frozen native weights; Flux 1A checkpoints do not directly load into this new architecture. A 48GB 9B variant is not the initial commitment. Use the 4B 48GB profile first if 9B admission fails.</p>
<p class="small">Use a model-specific native scene/mask bundle for each backbone. Keep fixed96 identities, order, prompts, references and seeds unchanged; do not silently reuse 9B scene latents as a 4B-native baseline. Cross-backbone results are a separate comparison.</p>
''')

page('A practical 16GB implementation path', '''
<p><b>What already worked:</b> the local Flux 1 one-ID run trained all eight Q/K/V/O adapters through the full frozen 4B denoiser at 768px, with microbatch 1/accumulation 4. Peak reserved memory reached <b>9.859GiB</b>. Its 19-pair training and fixed24 evaluation demonstrate local training feasibility, not multi-ID generalization. That measurement excludes Flux 2's new modules and identity decoder workload.</p>
''' + table(['Memory phase', 'Flux 2-Lite proposal'], [
    ['Conditioning', 'Stage Qwen text encoder, reference recognizer/detail encoder and VAE encoding separately. Retain small CPU features/latents in a bounded cache. Do not leave the full text encoder resident beside the denoiser.'],
    ['Backbone', 'Single shared BF16 4B weight set. Non-reentrant block checkpointing. Compute T/C without gradients; retain F autograd through every later native operation.'],
    ['Attention', 'Compute T, C and F query groups separately. Use block/sliced SDPA calls rather than materializing a dense custom attention mask that silently disables the efficient kernel. Preserve exact index mapping.'],
    ['Identity loss', 'Begin every fourth optimizer update on an eligible microstep, then calibrate effective contribution. Recompute the identical forward for a separate identity backward if needed; accumulate gradients before one optimizer step. Checkpoint VAE internals and record actual cost.'],
    ['Validation', 'Serialize denoising, decode, face metrics and CLIP in separate processes. Never run training and evaluation together on the local GPU.'],
    ['Admission target', 'Measured peak reserved memory below 90% of reported capacity: approximately 14.4GiB on a true 16GiB device. Test the longest text/context layout and a step with active identity backward.']]) + '''
<p><b>Fallback order:</b> recompute context K/V per block instead of retaining all layers; checkpoint more decoder internals; offload saved activations to CPU after verifying backward; reduce detail token count under a new profile; then use the ROI384 variant. Lower adapter rank alone will not solve a decoder/activation OOM.</p>
<p><b>Always-on identity availability:</b> feature precomputation is fine for frozen reference/target encoders, but generated-face recognition must remain differentiable. Do not replace it with a cached generated face embedding or an ONNX-only scoring path. A cropped VAE decode is not numerically equivalent to a full decode; validate its central crop against full decoding, or label it as an approximate objective.</p>
<p><b>Expected limitation:</b> all-layer adapters add backward work. A narrower face stream can reduce activations, but neither 16GB fit nor steps/second has been measured. The local recipe is a concrete candidate with an explicit lower-cost fallback, not a guarantee based on the old 9.859GiB number.</p>
''')

page('Flux 2-Lite-ROI384: more face detail, fewer tokens', '''
<p>This is the recommended secondary local experiment if full-scene identity backward exceeds memory, and also a useful quality ablation for the small faces in fixed96. It runs the <b>full 4B denoiser online</b> on a face-centered window; it is not an old cached velocity head or a one-step face refiner.</p>
''' + table(['Stage', 'Proposed crop protocol'], [
    ['Window', 'Derive a square crop centered on the frozen native owner box, with side twice its larger dimension; clip/pad deterministically. Resize to 384×384. Save the affine transform and original owner mask.'],
    ['Spatial capacity', 'The crop grid has at most 24×24=576 tokens, versus 2304 full-scene tokens at 768. Use a separate sanitized 256px scene context (16×16=256 tokens) plus the original text embeddings. These are token counts, not a 4× memory/speed claim.'],
    ['Training', 'Apply the same crop rule to training-target boxes and the corresponding reference-only preprocessing. Fresh noise/timesteps and masked native flow loss remain. Full native 4B weights are frozen, with Flux 2-Lite adapters at all 25 blocks.'],
    ['Coordinates', 'Map each crop token center back to original scene pixels/16 for the main spatial RoPE axes, retaining native temporal axes. Do not renumber selected tokens as contiguous scene locations. Dedicated reference attention still has its own crop-coordinate scheme.'],
    ['Local context', 'Erase the owner face + halo before both crop-context and global-context VAE encoding. Remove global context cells overlapping that exclusion. Reference detail/identity tokens remain independent.'],
    ['Decode / output', 'Decode the generated crop, map it back with the recorded transform and use the original output alpha to compose into the 768px native image. Exterior pixels remain exactly preserved.'],
    ['Evaluation name', 'Use fixed96_flux2_roi384 (same subjects/prompts/seeds/final canvas) and fixed24_flux2_roi384 for local smoke. Report beside the unchanged original protocol; never overwrite its baseline files.']]) + '''
<p><b>Why it can help:</b> the face occupies more model tokens and the differentiable decode covers 384px, reducing one costly training component. Most frozen model weights are still resident, so weights impose a memory floor. Generated details can improve identity even when the final small face is downsampled, but this is unmeasured.</p>
<p><b>Risks:</b> crop/full-scene distribution shift, lost body context, scale-dependent RoPE, edge/lighting artifacts and downsampling loss. Evaluate faces both at final resolution and in the full scene. Training/inference use the exact same crop transform rules and patched backend.</p>
<p>For quick wiring and memorization checks, reuse the separate 19-pair one-ID task and fixed24 checkpoints 0/500/1000/2000. Before a generalization claim, move to a small training-only multi-ID pilot and then the original identity-disjoint fixed96 evaluation.</p>
''')

page('Experiment sequence: make each improvement attributable', table(
    ['Arm / order', 'Change', 'Question / budget'], [
    ['0 · completed baselines', 'Native; Flux 1 2K; Flux 1A 4K and 6K. Preserve every artifact.', 'Existing evidence. No retraining needed to read scores.'],
    ['1 · causal probe of Flux 1A', 'Own bank, donor bank, BA-off second pass, zero/mean bank. Hold native conditioning, seed, mask and exterior fixed.', 'Does the trained bank control identity at 4K/6K? Named swap8 diagnostic; unperformed.'],
    ['2 · missing Flux 1A controls', 'Flux 1b flow-only; then Flux 1A with calibrated identity weight and unchanged architecture.', 'Separate objective benefit from architecture. Same data/order/sampler; initial 2K, extend to 4K only if useful.'],
    ['3 · Flux 2-Bridge', 'Recognition/detail memory + dedicated additive reads at the original 8 sites; original target topology retained.', 'Test memory/projection quality before stricter routing. This is a diagnostic, not the fully isolated Flux 2.'],
    ['4 · persistent conditioning', '8 sites → all blocks; then add face-only ID modulation. Retain the same loss for these comparisons.', 'Test depth/MLP authority. Include an equal-parameter 8-site control where budget allows.'],
    ['5 · Flux 2 core', 'Add RGB-sanitized C, all-layer directed connectivity and persistent face-state provenance.', 'Test whether removing scene/text relays improves donor control and actual ID without damaging cohesion.'],
    ['6 · Flux 2 + contrastive', 'Add verified other-ID negatives; preserve row order and flow distribution.', 'Test discrimination versus simply scaling cosine loss.'],
    ['7 · ROI / temporal / capacity', 'One change per arm: ROI384, scheduled strengths, short rollout, wider adapters.', 'Only after semantic reference control improves. These are not prerequisites for the first prototype.']]) + '''
<p><b>Practical prioritization:</b> implement the architecture in stages on 4B. Use the existing one-ID task only to establish training and save/resume. Then use a training-only multi-ID development split for tuning. Reserve the complete fixed96 for the required checkpoints and final paired comparison; repeated tuning against it weakens any generalization claim.</p>
<p><b>Matched science versus quick tests:</b> the large-GPU causal controls keep effective batch 1, original training order, resolution and native loss. The local accumulation-4 smoke is a distinct profile. If increasing effective batch, data diversity, crop scale or reference resolution, assign a new name and do not attribute the combined change solely to architecture.</p>
<p>At minimum, run arms 1, 3 and 5 with a common loss, plus the 16GB prototype. The complete list is an ordered research program, not a request to fund every run immediately.</p>
''')

page('How Flux 2 earns a claim of stronger reference use', table(
    ['Test', 'Controlled intervention', 'Expected evidence'], [
    ['Reference causality', 'Fix prompt, seed, owner mask and native scene. Swap only Flux 2 identity/detail memory to a different known identity.', 'Donor cosine rises and original-owner cosine falls; face still looks natural. Pixel change alone is insufficient.'],
    ['Channel separation', 'Swap only global ID, only detail, and both; then use another photo of the same identity.', 'Global ID controls person; detail does not overpower identity or force reference expression. Same-ID views yield stable person identity.'],
    ['Native-face exclusion', 'Edit only native RGB pixels inside fixed D, then redo sanitation/encoding while all other inputs are fixed.', 'R/C/F and face-interior result are equal within deterministic numerical tolerance. Do not compare feather-ring RGB, which intentionally uses native pixels.'],
    ['Hidden relay check', 'Perturb F in one block and compare next-layer T/C/R; repeat through depth.', 'T/C/R remain invariant; proves the mask is not bypassed through a shared residual, cached tensor or single-stream concatenation.'],
    ['Null control', 'Use BA-off native and a distinct firewall/no-ID baseline; zero or mean memory is only a diagnostic.', 'Separates architectural distribution shift from learned reference benefit. Zero memory is out of distribution unless dropout training explicitly supported it.'],
    ['Quality / ownership', 'Original fixed96 owner boxes, unchanged thresholds; blinded paired face/full-image review.', 'No new missing/ambiguous faces, expression freezing, copy-paste artifacts, seams or accessory loss.'],
    ['Independent identity audit', 'A different frozen recognizer, never used for conditioning or loss; supplementary held-out identities.', 'Improvement is not restricted to the optimized ArcFace system. Keep the official owner-ID unchanged.']]) + '''
<p><b>Proposed practical target, not a forecast:</b> +0.05 absolute owner-ID over native would mean ≥0.3242 on this panel and would exceed Flux 1A 4K as well as the historical Flux 1 2K result. Predeclare this ambition alongside improvement in at least 6/8 identity means, ≥60/96 image wins and no increase in ownership failures.</p>
<p>Report identity-cluster bootstrap intervals and all per-identity scores. For a strong claim, seek a positive interval on the gain, repeat a promising run with another training seed, and confirm on a fresh held-out panel. A target of 0.3242 is an engineering decision threshold, not a predicted score or universal face-recognition standard.</p>
<p>Retain CFG4 and 20 steps for the primary panel. Any identity-strength, CFG or sampling-step sweep has a separate name. Score every prescribed item, including failures; never drop difficult images or choose a different detected face to improve the result.</p>
''')

page('Implementation map, admission and failure handling', table(
    ['Location / work item', 'Required implementation'], [
    ['New branch module', 'Add a separate Flux 2 module for ID/detail projectors, all-layer reads and modulation offsets. Keep Flux 1A modules and checkpoint loaders intact.'],
    ['Native model patch', 'Explicit F/C/T partitions in both double and single blocks; directional reads, native positions and frozen shared weights. No in-place cross-lane writes.'],
    ['flux_runtime / config', 'New named dispatch with strict BA-off fallback, profile validation and source identity. Train and sample through the same patched functions. Do not reuse old manifests under a new topology.'],
    ['Conditioning / geometry', 'Separate sanitized context encoding; reference-only ID/detail features; affine and mask provenance; empty-mask behavior and owner assignment.'],
    ['Objective', 'Native masked flow; differentiable identity; optional verified negative bank; per-σ gradient telemetry. Labels and negatives come only from training.'],
    ['Checkpoints / metrics', 'Inventory every trainable parameter and optimizer tensor; preserve backbone hashes, reference-memory preprocessing hashes, RNG/cursor and scheduler. Keep official metric definitions and add diagnostic namespaces.']]) + '''
<p><b>Before a long run:</b> check exact token indices/RoPE after gather/scatter; native BA-off and empty-mask parity; independent R and directed T/C invariance; finite first gradients and actual updates after zero-output initialization; frozen-weight preservation; checkpoint reload; fresh-process resume with the same next update; one paired pretrained inference; and peak reserved memory below 90% of actual device capacity on the largest active-ID layout.</p>
<p>Native weights being frozen does not permit no_grad around F. Later frozen operations must propagate gradients to earlier identity adapters. Conversely, caching T/C/R is valid only if their input hashes, timestep, preprocessing and dropout state match; never reuse a σ-conditioned bank at another σ.</p>
<p><b>Initial run spec:</b> seed 142; LR 5e−5, warmup 100, AdamW with zero weight decay and clip 1.0; save every 500 updates; original fixed96 at 0 and every 2,000 updates. Start with a 2K pilot, then 4K/6K only after review. Log an immutable Comet key under rsrch_new, data/config/source/weight hashes and peak reserved memory. Keep validation serial.</p>
<p><b>Failure response:</b> a flow-only pass does not admit an identity-loss run. An OOM in decode triggers the explicit local fallback, not a silent metric/loss change. Failed invariance blocks the isolation claim. A weak donor response blocks rank scaling. A higher ArcFace score with worse independent identity or visible faces blocks promotion.</p>
<p class="small">No GPU host is selected or started by this report. The previously used Vast 53994096 is recorded as stopped. Any future rental follows the project's offer/cost review and explicit host authorization; existing jobs/checkouts remain untouched.</p>
''')

page('Evidence, reproducibility and remaining uncertainty', '''
<p><b>Measured evidence:</b> reports/261005_FLUX1a_results contains the completed results report, saved 0/2K/4K/6K per-image CSVs, routing masks and admission receipts. This report independently recomputes all displayed owner-ID/CLIP means, checks all 96 sample IDs/order and derives the face-size statistics from frozen boxes. It reuses two historical sample panels.</p>
<p><b>Source identity:</b> the deployed archive in scratch/FLUX1a_deploy was inspected and the relevant small source files copied into this report's evidence folder. The local masked-attention module, runtime and patch match that archive; the local isolated-reference and identity-objective files differ. Architectural descriptions above use the deployed code. The inspected local 6K manifest retains the same model identity and training-code digest as 2K; its hash and a copy are included.</p>
''' + table(['Pinned item', 'Recorded value'], [
    ['Upstream Toolkit commit', '<span class="hash">ecee894ed2b1f3716d9d7326693061ec1a3105bb</span>'],
    ['FLUX9B weight revision', '<span class="hash">32773329fbe7e81a90ef971740e8ba4b0364ecf3</span>'],
    ['Text weight revision', '<span class="hash">b968826d9c46dd6066d109eabc6255188de91218</span>'],
    ['VAE weight revision', '<span class="hash">3f679cf232e6d91d28396522d9502c31e8f7ccbe</span>'],
    ['Flux 1A Comet key', '<a href="https://www.comet.com/nikolay-2104/rsrch-new/2018ec7a730243bc98d922178e58aa5c">2018ec7a730243bc98d922178e58aa5c</a>'],
    ['GB10 memory', '60.668GiB maximum admission reservation; 59.994GiB recorded on resumed production. These are historical Flux 1A measurements.'],
    ['16GB historical memory', '9.859GiB reserved in the completed Flux 1 local 4B one-ID run. Not Flux 2 admission.']]) + '''
<p><b>Not measured here:</b> pretrained Flux 1A donor-reference causality; training-wide identity-gradient statistics; 1A native-to-trained face cosine over all 96 images; Flux 2 invariance, gradients, memory, speed or image quality. The earlier native-to-trained cosine 0.5708 belongs to Flux 1 at 8K, not Flux 1A, and is not reused as a 1A measurement.</p>
<p><b>Deliverables:</b> PDF; portable HTML; eight original chart/diagram sets in SVG/PDF/PNG; analysis.json with input hashes; copied per-image evidence and deployed source excerpts; proposed_profiles.json; and the two reproducible CPU build scripts. Architecture figures show proposals and contain no invented performance curves.</p>
<p class="small">Analysis limitations: a single completed training run, eight held-out identities, a repeatedly inspected validation panel, and a recognition metric also used for training. The proposed path improves the experiment's ability to test reference control; significant identity improvement remains to be demonstrated.</p>
''')

page('Primary research sources and reading guide', ''.join(
    f'<p><b>{key}. {html.escape(name)}</b><br>{html.escape(date)}.<br><a href="{url}">{url}</a></p>'
    for key, (name, date, url) in SOURCES.items()) + '''
<p class="small">Read R1 and R2 first for persistent identity modulation and contrastive supervision; R3/R10 for independent injection paths; R5/R6 for time/space tradeoffs. R7/R8 are the September 2026 additions. Paper mechanisms are inspiration; the Flux 2 information boundary is a project-specific proposal.</p>
''')

PROFILES = {
    'status': 'DESIGN_ONLY_NOT_ACCEPTED_BY_CURRENT_CONFIG_SCHEMA',
    'date': '2026-10-05',
    'shared': {
        'backbone': 'frozen pinned FLUX.2-klein Base',
        'reference_memory': {'identity_encoder': 'existing pinned ArcFace; reference only',
                             'identity_tokens': 4, 'detail_encoder': 'DINOv2 ViT-S/14; revision to lock',
                             'detail_crop_pixels': 224, 'detail_tokens': 64},
        'attention': {'T_reads': ['T'], 'C_reads': ['T', 'C'], 'F_reads': ['T', 'C', 'F'],
                      'F_extra_reads': ['R_identity', 'R_detail'], 'R_reads_target': False},
        'context': {'redact_before_vae': True, 'extra_halo_pixels_proposal': 32,
                    'reuse_original_native_latent_in_face_pass': False},
        'optimization': {'lr': 5e-5, 'warmup': 100, 'seed': 142, 'weight_decay': 0,
                         'gradient_clip': 1, 'checkpoint_every': 500, 'pilot_steps': 2000},
        'validation': {'output_size': [768, 768], 'steps': 20, 'guidance': 4,
                       'fixed96_at': [0, 2000, 4000, 6000], 'serial': True},
        'admission': {'max_reserved_fraction': .9, 'hardware_fit_measured': False,
                      'requires_pretrained_parity_and_invariance': True},
    },
    'profiles': [
        {'name': 'Flux2_9B_768_proposed', 'arch': 'flux2_klein_9b', 'target': [768, 768],
         'sites': 'all 8 double + 24 single', 'branch_width': 512, 'modulation_rank': 128,
         'core_parameters': 230686720, 'parameter_budget': 240000000,
         'microbatch': 1, 'grad_accum': 1, 'checkpointing': True,
         'hardware_target': '80GB or GB10-class, specified host required'},
        {'name': 'Flux2_4B_48G_proposed', 'arch': 'flux2_klein_4b', 'target': [768, 768],
         'sites': 'all 5 double + 20 single', 'branch_width': 256, 'modulation_rank': 64,
         'core_parameters': 63569920, 'parameter_budget': 70000000,
         'microbatch': 1, 'grad_accum': 1, 'checkpointing': True, 'hardware_target': '48GB'},
        {'name': 'Flux2_Lite_4B_16G_proposed', 'arch': 'flux2_klein_4b', 'target': [768, 768],
         'sites': 'all 5 double + 20 single', 'branch_width': 128, 'modulation_rank': 32,
         'core_parameters': 30146560, 'parameter_budget': 35000000,
         'microbatch': 1, 'grad_accum': 4, 'checkpointing': True, 'identity_every_updates': 4,
         'hardware_target': '16GB', 'first_smoke': 'existing 19-pair one-ID / fixed24'},
        {'name': 'Flux2_Lite_ROI384_16G_proposed', 'arch': 'flux2_klein_4b', 'target': [384, 384],
         'final_canvas': [768, 768], 'global_sanitized_context': [256, 256],
         'sites': 'all 5 double + 20 single', 'branch_width': 128, 'modulation_rank': 32,
         'crop_side': '2 * max(owner_box_width, owner_box_height)',
         'microbatch': 1, 'grad_accum': 4, 'checkpointing': True,
         'validation_protocol': 'fixed96_flux2_roi384; separate from full-scene protocol',
         'hardware_target': '16GB fallback; not measured'},
    ],
}
(OUT / 'proposed_profiles.json').write_text(json.dumps(PROFILES, indent=2) + '\n')
(OUT / 'research_sources.json').write_text(json.dumps({k: dict(title=v[0], date=v[1], url=v[2],
    accessed='2026-10-05', role='primary source; not a local reproduction') for k, v in SOURCES.items()}, indent=2) + '\n')

CSS = '''
body {font-family:sans-serif;font-size:10pt;line-height:1.28;color:#20323f}
h1 {font-size:21pt;color:#096c79;margin:0 0 11pt}
p {margin:7pt 0} b {color:#203b53}
.eyebrow {font-size:11pt;color:#637783}
.small,.caption {font-size:8.3pt;color:#556976;line-height:1.25}
.caption {margin:4pt 0 9pt}
.callout {background:#e9f4f5;padding:11pt;border-left:4pt solid #087f8c;margin:10pt 0}
.equation {font-family:serif;font-size:10.3pt;line-height:1.35;background:#f1f5f8;padding:9pt;margin:8pt 0}
table {width:100%;border-collapse:collapse;font-size:9pt;margin:9pt 0}
th {background:#096c79;color:white;padding:6pt;text-align:left}
td {vertical-align:top;padding:5pt;border-bottom:.5pt solid #d8e2e8}
tr:nth-child(even) {background:#f3f6f8}
a {color:#087f8c;text-decoration:none}
.hash {font-size:8pt;font-family:monospace}
img {max-width:100%}
'''

pdf = pymupdf.open()
sections = []
for title, body in PAGES:
    story = pymupdf.Story(html=body, user_css=CSS, archive=pymupdf.Archive(str(OUT)))
    doc = story.write_with_links(lambda _, __: (pymupdf.Rect(0, 0, 595, 842),
                                              pymupdf.Rect(42, 48, 553, 791), None))
    sections.append({'title': title, 'start_page': len(pdf)+1, 'pages': len(doc)})
    pdf.insert_pdf(doc)

toc = [[1, s['title'], s['start_page']] for s in sections]
pdf.set_toc(toc)
for i, p in enumerate(pdf):
    p.insert_text((42, 27), 'FLUX 2  |  Architecture proposal and Flux 1A evidence', fontsize=8, color=(.32, .40, .45))
    p.insert_text((42, 817), '5 October 2026  |  New architecture and hardware fit are unmeasured', fontsize=8, color=(.32, .40, .45))
    p.insert_text((515, 817), f'{i+1}/{len(pdf)}', fontsize=8, color=(.32, .40, .45))
pdf.set_metadata({'title': 'Flux 2 — Isolated face generation after Flux 1A',
                  'author': 'rsrch_new', 'subject': 'Architecture audit, research synthesis and 9B/4B/16GB experiment design'})
dest = OUT / 'Flux2_architecture_review_and_proposal.pdf'
pdf.save(dest, garbage=4, deflate=True)

body = ''.join(f'<section id="s{i}">{content}</section>' for i, (_, content) in enumerate(PAGES))
def embed(match):
    path = OUT / match.group(1)
    mime = 'image/jpeg' if path.suffix == '.jpg' else 'image/png'
    return 'src="data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode() + '"'
body = re.sub(r'src="([^"]+)"', embed, body)
nav = '<nav><b>Contents</b>' + ''.join(f'<a href="#s{i}">{i+1}. {html.escape(title)}</a>' for i, (title, _) in enumerate(PAGES)) + '</nav>'
screen_css = '''html{background:#e9edf1}body{margin:0 auto;max-width:930px;font-size:15px}
section,nav{background:white;padding:45px 54px;margin:20px 0;box-shadow:0 2px 14px #ccd3d9}
section img{display:block;width:100%;height:auto}nav a{display:block;padding:4px}table{font-size:13px}
@media print{html{background:white}section{page-break-before:always;box-shadow:none;margin:0;padding:0}nav{display:none}}
'''
(OUT / 'Flux2_architecture_review_and_proposal.html').write_text(
    '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    '<title>Flux 2 architecture proposal</title><style>' + CSS + screen_css + '</style></head><body>' + nav + body + '</body></html>')

violations = []
for i, p in enumerate(pdf):
    for b in p.get_text('dict')['blocks']:
        if b['type'] != 0:
            continue
        for line in b['lines']:
            for span in line['spans']:
                x0, y0, x1, y1 = span['bbox']
                if x0 < 0 or x1 > 595 or y0 < 0 or y1 > 842:
                    violations.append({'page': i+1, 'text': span['text'], 'bbox': span['bbox']})
audit = {'pages': len(pdf), 'sections': sections, 'bounds_violations': violations,
         'characters': [len(p.get_text()) for p in pdf],
         'links': sum(len(p.get_links()) for p in pdf),
         'pdf_sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
(OUT / 'document_validation.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))
