# Fast face-focused FLUX one-ID diagnostic

## Purpose and differences from the earlier run

The original one-ID experiment proved finite branch updates and visible output
changes. Its reference-face K/V delta was applied at eight sites and every target
query. This new, explicitly named experiment tests a smaller face-routed branch
at `single_blocks.19`, the last single-stream block of Base 4B. It leaves the
previous implementation, configurations and checkpoints intact.

The new branch computes one native-Q/native-K attention read of reference-face
values. A rank-16 output LoRA transforms that read. A scalar sigmoid router
predicts face membership from each target's native normalized/rotated query:

```
R = SDPA(Q_target, K_reference_face, V_reference_face)
p_face = sigmoid(W_gate * RMS_normalize(Q_target) + b_gate)
delta = p_face * LoRA_output(R)              # gamma = 1
attention = native_attention + scatter_target(delta)
```

The native last-block output projection, modulation gate, residual and final
prediction layer then run unchanged. Text and reference queries receive zero
direct correction. The same module and attention seam run during full inference
and cached training. Both classifier-free guidance lanes use the same BA
module. A first conditional-only trial is retained for comparison; it amplified
the shared branch correction by CFG=4 and reduced identity similarity.

Only `reference_branch.output_delta.{a,b}` and
`reference_branch.face_gate.{weight,bias}` are trainable: 101,377 parameters
(98,304 output-LoRA plus 3,073 router), compared with 1,572,864 previously.
There are no native-attention LoRAs, Q/K/V updates, encoder/VAE updates or
trainable backbone weights.

## Face targeting and initialization

Target boxes are used only to construct supervised training masks through the
exact target resize/crop and 16-pixel token grid. Fractional edge coverage is
retained. The objective sums mean face flow-MSE, mean background flow-MSE and
class-balanced router BCE. The face receives equal aggregate flow-loss weight
to the background despite covering a smaller area. This is a deliberate change
from native whole-image MSE, not a matched comparison to the previous experiment.

Inference takes reference images/boxes, prompts and seeded noise; it never takes
target boxes, generated-face detections or output masks. Its face gate is
predicted from internal features. It is a soft gate: background influence must
be measured, and exact face-only final-pixel changes are not guaranteed across
iterative denoising. The report must show router overlays and face/background
activity rather than merely label the mechanism face-specific.

Output-LoRA B starts at zero, so untrained predictions match native predictions.
The initial router prior is 0.1 everywhere; supervision teaches localization.
No unknown future face location is hardcoded into inference noise. Separate
face/background losses make the desired update explicit.

## Efficiency and experiment limits

The named pilot uses four cross-view training pairs of the same ID, three fixed
noise levels (0.2, 0.5, 0.8), and a second set of noise seeds reserved for probes.
It uses 512 px targets, batch four, 300 updates, and two original validation
prompts/seeds with 20 denoising steps. This is a separate diagnostic panel from
the earlier 768 px twelve-image run and the original 96-image evaluation.

The prefix is frozen and occurs before the only adapter. For these fixed inputs,
its features and native attention/MLP outputs can therefore be cached exactly.
Optimization loads only the BA, frozen last-block output projection and final
layer, and caches the small fit/probe set on GPU. The full backbone, text encoder
and VAE are absent from the optimization process. Keeping the original
concatenated output-projection call preserves native BF16 rounding.

This accelerates a finite-input overfit diagnostic. It is not a claim of equal
speed for arbitrary fresh diffusion noise, changing crops or earlier branch
sites: those require rebuilding the prefix features. Reused-noise training
improvement must be distinguished from the separate-noise probe and generated
validation images. The probe reuses the four images and is not an unseen-image
or unseen-identity generalization test.

## Run and checks

Use the existing FLUX environment and pinned checkout. On a fresh machine,
follow `docs/deployment.md` for `uv` environment/weights setup and import the
one-ID assets using `scripts/import_one_id.py` before running:

```bash
bash scripts/run_face_diagnostic.sh runs/flux4b_face_one_id_fast_NAME
```

The launcher stages conditioning precomputation, exact prefix capture, native
initial validation, BA optimization, full trained inference and VAE decoding in
separate processes, then creates `review.html`. Optimization deliberately exits
at update 50 and resumes in a fresh process to exercise checkpoint restoration.
Comet uses project `rsrch_new` and an immutable run key;
all scalar metrics are logged while the terminal keeps compact progress/ETA.
Generated gates are saved alongside validation latents for review.

Required checks are native/untrained parity, exact cached/full predictions,
training-only face geometry, BA-only optimizer inventory, finite/nonzero B
gradients, frozen-parameter isolation, checkpoint reload and CUDA peak below 90%.
Checkpoint/source identities and cache content hashes are recorded in the run.
The previous Vast instance was found stopped; it was not restarted. The first
trial is on the local 16 GB GPU.

## Measured results

The small synthetic check passed zero-output initialization, target-query scope,
nonzero B/router gradients, dependence on reference values and rejection of
external inference target masks. It is not pretrained-model validation.

The first cache attempt stopped before training: replaying only target rows
changed BF16 GEMM rounding (maximum prediction difference 0.03125). Preserving
the full native token layout in both final projections fixed this; all 24
pretrained cache cases then matched the full model exactly.

The first complete local trial, `runs/flux4b_face_one_id_fast_20261001_v2`, used
BA only on the positive CFG lane. Its 300 optimizer updates took 14.224 seconds
(0.0474 seconds/update), peak reserved training memory 3.803 GiB, and cache
capture took 13.695 seconds with 8.158 GiB peak reserved. These timings exclude
imports/model loading, precomputation, validation, decoding and Comet flushing.
All gradients were finite. Checkpoint reload was exact; an independent
uninterrupted-versus-resumed check reproduced all branch tensors exactly at
update 51. Trained cached/full predictions were also exactly equal.

Fit face-MSE fell 0.793452→0.706463 (10.96%); separate-noise probe face-MSE fell
0.792423→0.768510 (3.02%). Probe gate IoU reached 0.936452, with average gate
probability 0.927804 on faces versus 0.013043 off faces. Per-token velocity
correction RMS averaged 0.255982 inside faces and 0.003994 outside (64.1×).
The probe uses the same four target images with different noise, not new images.

Both generated images changed visibly (mean absolute pixel difference 23.32 and
8.33/255), and gate overlays highlighted faces with some hands/clothing leakage.
However, identity similarity declined 0.383189→0.290081; CLIP rose
27.241908→27.922511. This proves BA influence and localized finite-input
learning, not better identity reproduction. A shared-CFG comparison was run
with the same inputs/seeds and optimizer settings. Original sources, checkpoint,
images and metrics are preserved under the first trial, including
`source_snapshot/` for its exact implementation.

### Completed shared-CFG comparison

The retained implementation activates the same face BA in both CFG lanes.
Run: `runs/flux4b_face_one_id_fast_20261001_shared_cfg`; Comet:
[a49b042877154369a30080c2da5393f6](https://www.comet.com/nikolay-2104/rsrch-new/a49b042877154369a30080c2da5393f6).
All four final branch tensors and both native baseline PNGs are byte-identical
to the conditional-only trial, isolating the inference CFG change. The trained
full-model/cached-tail comparison remains exact. Both generated image pairs and
their actual predicted gates were visually reviewed.

| Two-prompt metric | Native baseline | Positive-only BA | Shared-CFG BA |
| --- | ---: | ---: | ---: |
| Identity similarity | 0.383189 | 0.290081 | **0.438182** |
| CLIP text similarity | 27.241908 | 27.922511 | 27.454217 |

Identity improves on both images: 0.491070→0.531499 and 0.275309→0.344866.
Both retain detected/owned faces; no missing or ambiguous masks were reported.
Mean absolute pixel differences versus native are 17.101 and 7.081/255.
The gates concentrate on faces, with some hands/clothing activation. Faces,
hands and clothing also change during denoising; this is not strict pixel-level
face isolation. The scoring masks are frozen from the native 512 px baseline
and used only for evaluation.

The fit/probe losses and routing measurements above are identical because the
trained weights are identical. All 300 recorded losses/gradient norms and all
checkpoint tensors are finite. The final output-LoRA B matrix is nonzero.
The run saved at update 50, exited, and resumed in a fresh process through 300.
The independent uninterrupted/resume check from the identical training
trajectory is preserved with the final audit.

Measured speed on the local RTX 4090 Laptop GPU: 300 updates in **14.698 seconds**
of optimizer time, batch 4, 0.0490 seconds/update; full-run maximum reserved
training memory **3.803 GiB**. Prefix capture took **13.425 seconds**, with
8.158 GiB peak reserved. Time from run creation through review generation was
**272 seconds** (about 4.5 minutes), including model/import startup, conditioning
preparation, baseline/final image generation, decoding, CPU ID/CLIP scoring and
Comet synchronization. This is a cached fixed-input experiment, not a measured
fresh-noise training throughput claim.

Artifacts: `review.html`, `paired_images_and_gates.png`, `learning_curves.png`,
`probes.json`, `final_audit.json`, `trained_cache_parity.json`, checkpoints every
50 updates, and the immutable Comet key. The report/curves/images and source
identity were uploaded to Comet. The original conditional-only trial and its
negative identity result remain available under experiment
`06b351cea6264fcfbffa25a12d15cfb1`.

Conclusion: the requested small experiment demonstrates finite, localized BA
learning and visible face changes, with an identity gain on the two generated
samples. Four training images and two validation prompts do not establish
generalization. No larger training run or Vast restart was initiated.
