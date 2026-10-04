# Prompted scenes: native background and BA-generated face

## Corrected task — 2026-10-01

The face-crop reconstruction experiment does not satisfy prompted validation.
The user explicitly requires new images from the ID reference and existing
prompts, native-backbone backgrounds, and face generation by BA. They also
explicitly authorize inference masks derived from native backbone generations.
That authorization is a named exception to the original reference-only mask
policy; no target photograph or target-photo mask is loaded during validation.

## CL14 precedent and deliberate adaptation

Inspected the previous project's
`diffusion_template/src/configs/CL14_cosmic_joint_shadow_sa128_softmask_24k.yaml`
and `src/model/photomaker_branched/attn_processor_cleanest.py`.
CL14 routes target queries spatially: background Q reads native target K/V,
face Q reads reference-face K/V, and the messages merge through the target
mask. CL14 specifically adds a two-latent-cell feather to training masks over
CL9; its inference pipeline supplies its own masks.

Here the simple FLUX proof routes **flow outputs**, not U-Net attention outputs.
This is an adaptation of the spatial split, not an exact CL14 port. Its reason
is explicit initialization: a zero BA flow leaves the selected face as noise,
whereas zeroing an attention residual would still let native FLUX draw a face.

1. Reuse 24 native FLUX generations: the existing 12 prompts, seeds 0 and 1,
   identity 51, 768 px, 20 steps, CFG 4. Validate exact image/latent hashes and
   the backbone/revision/panel mask signature.
2. Reuse each native face box, keeping weight one inside the box and a 16 px
   feather outside it. Review all 24 overlays. The pixel mask is averaged to
   the packed 48×48 token grid.
3. Start a new face latent at the original Gaussian noise. At each timestep,
   supply the frozen native scene outside the mask at the corresponding noise
   level. Frozen FLUX extracts prompt-conditioned target Q and reference-face
   K/V at the final single-stream attention block, using native normalization
   and RoPE. Only BA predicts face velocity.
4. After decoding, blend the face region into the original native PNG. This
   explicitly guarantees identical exterior pixels despite VAE spatial mixing.
   Save raw decoded images separately; pixel preservation is not a claim about
   attention alone.

The known background interpolation is `(1-sigma)*native_latent + sigma*noise`;
it is not a replay of the original native denoising trajectory. BA face CFG is
1 while native background CFG is 4, recorded as an experiment deviation.

```text
ID image + prompt + seed ── frozen native FLUX ── native scene
                                                 │
                                          detect/reuse face mask
                                                 │
new face noise + native background at sigma + prompt + ID
                       │
              frozen FLUX Q / reference K,V
                       │
                 trainable BA flow
                       │
              Euler face update × mask
                       │
           decode and preserve native exterior
```

## Trainable architecture and protocol

Reuse the small `FaceCropFlow` network as a full-image token head:

`v_face = W_out(Q_branch + Attention(Q_branch,K_branch,V_branch)) + W_noise(x_t)`

Random Q/K/V projections, zero W_out and W_noise, width 256, four heads,
2,408,448 parameters. All five tensors belong to BA. Backbone, VAE, text
encoder, native attention and output projection are frozen. No native velocity
is added inside the face. Pretrained query features already include native
reference conditioning: identity is not exclusively carried by the added read.

Train on all 19 full-image one-ID training pairs, at native target geometry,
using the fixed validation ID reference for reference conditioning. Cache six
noise levels × two fit seeds plus one separate probe seed. Sample 64 face
tokens per case using training-mask coverage weights; optimize native flow MSE
on those tokens, batch 8 with seeded uniform case sampling, AdamW lr .001 with 100 warmup steps, 2,000 updates.
These are cached-feature diagnostic updates, not fresh-noise end-to-end steps.

Validate all 24 **new prompted scenes** at 0/1k/2k with no target photographs.
Native backgrounds stay fixed across checkpoints. Check native BA-off parity,
finite gradients/parameter changes, memory below 90%, exact process resume,
cached/fresh prediction parity, checkpoint hashes, and zero exterior pixel
error. Keep existing ID/CLIP scoring and fixed native ownership boxes.

## Run and status

- Runner: `scripts/masked_face_flow.py`
- Routing: `ba_dit/nn/masked_face_flow.py`
- Launch: `bash scripts/run_masked_face_flow.sh runs/NEW_NAME`
- Retained run: `runs/flux4b_masked_face_flow_24_stable_20261001`
- [Comet](https://www.comet.com/nikolay-2104/rsrch-new/d59463ae28784748a9092746eafeee24)

Implementation and focused routing/initialization/gradient checks completed.
All 24 masks matched their source native PNGs and overlays were reviewed.
The retained run completed 2,000 updates and all 72 prompted outputs.
Crop reconstruction results are not presented as evidence for this task.

## Optimization pilot and correction

The initial full-scene pilot used the crop diagnostic's lr .002 and sequential
case batches. Its probe MSE improved to 1.522 at step 500, then regressed to
2.179 at step 1000 (zero-flow baseline 2.199). It was stopped at 1000 before
finishing the image sweep; both generated latent samples and all checkpoints,
losses and source snapshots remain in `runs/flux4b_masked_face_flow_24_20261001`.

Three cached 2000-step controls shared the same zero initialization, cases and
seeded uniform batch order. Probe MSE at 2000 was 1.071 (lr .001), 1.134
(lr .0003), and 1.287 (lr .0001). These are cached-loss controls, not full image
validation. The separate-noise probe was used to choose the learning rate, so
it is not an unbiased test set. The fresh retained run uses lr .001 with uniform seeded batches.
Its resumed step-1000 fit/probe scores exactly reproduce the chosen control:
0.9571/1.1197. Removing the reference read raises probe error to 1.2725.

The retry reuses cache files by verified hash and records donor provenance;
only optimization order/rate changed. Exact resume and the new full image
validation still run independently.

## Completed prompted-generation results

| Output | Mean ID similarity | CLIP text similarity |
| --- | ---: | ---: |
| Untouched native backbone | 0.33140 | 28.0729 |
| Zero BA / face noise | 0.02024 | 28.1677 |
| BA 1,000 | 0.15084 | 26.9011 |
| BA 2,000 | 0.19661 | 27.0128 |

At both trained checkpoints a face was assigned to the intended frozen face box
in all 24 outputs. The noise initialization can also trigger detections from
remaining context; detection alone is not proof of identity generation. The
images show learned faces in all the requested scenes, with substantial texture
and shape artifacts. BA has learned, but has not surpassed native identity or
image quality. The native comparison uses the same exact backgrounds/reference/
prompts/seeds; its original masks and metric definitions are unchanged.

Final fit/probe flow MSE is 0.83184/1.07052: reductions of 62.1%/51.3% from
zero BA. Removing the reference read raises probe MSE to 1.31533 (+22.9%).
All 2,000 losses and gradients are finite, and every BA tensor changed. Native
BA-off parity, fresh-process step-51 resume, and cached/live predictions at
1k/2k are exact. Peak reserved CUDA memory for inference is 9.422 GiB. All
72 final composites have exactly zero pixel error outside the blend support.
This preservation is enforced after decoding; raw decoder outputs remain
available and may have changes around/outside the mask.

Review artifacts in the retained run: `report.md`, `overview.png`,
`comparison_1.png` through `comparison_3.png`, `face_details_1.png` and
`face_details_2.png`, `mask_overlays.png`, `learning_curves.png`, and
`final_audit.json`. Raw decoded images use the `_raw.png` suffix. Images,
metrics, code snapshots and diagnostics are logged in the retained Comet run.
