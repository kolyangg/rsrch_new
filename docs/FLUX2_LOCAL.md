# FLUX2 local one-ID pilot

This implements the proposed **Flux2 Lite 4B** at full 768×768 resolution. It is
our experimental branch architecture on the pinned FLUX.2-klein Base backbone;
it is not a new upstream FLUX release. Improved identity quality remains an
experimental hypothesis until the fixed-panel metrics are available.

## Implemented architecture

The native backbone stays frozen. All 5 double and 20 single blocks have a
persistent face state, a rank-128 reference read and rank-32 identity modulation.
Text reads text; sanitized context reads text/context; face reads text/context/
face. Neither text nor context can read the face state. Original spatial RoPE
coordinates are retained after token gathering. Reference reads use separate
softmaxes and K/V projections for four ArcFace identity tokens and 64 DINOv2
ViT-S/14 detail tokens. A shared query/output projection adds the reads outside
the native modulation gate, with fixed identity/detail multipliers 1 and 0.5.
During checkpointed training, exact frozen text/context K/V buffers are held
on CPU and copied back for face replay; their queries are not retained.
The detail memory has learned crop-position embeddings; neither reference read
uses scene RoPE. Frozen reference features never consume a target or prompt.

The branch has 30,482,432 trainable parameters in 179 tensors. All branch parameters are FP32. Output and identity-modulation projections start
at zero; other projections are nonzero so output/modulation gradients are live
on the first update and upstream branch projections receive gradients after it.
The trainable identity/detail projectors remain in the autograd graph.

Before VAE encoding, context RGB is erased across face support plus feather and
a 32-pixel halo. Context tokens intersecting this region are discarded. During
inference, the binary face support starts from fresh noise, while the context
follows its noised sanitized latent. Decoder assembly also uses that sanitized
latent. The final native RGB exterior is restored with the established feather
compositor (generated RGB has full weight inside the detected face box).

BA-off and an empty mask use the original patched native forward exactly.
An active zero-initialized FLUX2 branch intentionally changes attention topology;
its step-0 output is not the native baseline. The existing upstream patch is
unchanged. The active forward is shared by training and validation.

## Data, objective and evaluation

- Existing 19-pair one-ID training manifest; unchanged shuffled sample stream.
- Existing fixed24 validation manifest: original prompts, reference images, seeds,
  768px output, 20 Euler steps and CFG 4. Validation has no target photos.
- Fresh native flow noise/timesteps and face-normalized velocity MSE.
- Frozen differentiable VAE/ArcFace identity auxiliary: weight 0.05, sigma ≤0.5,
  on every fourth update interval, starting at the first update. Training-photo alignment is used only for loss.
- Microbatch 1, accumulation 4, AdamW learning rate 5e-5, warmup 100, clipping 1.
- 2,000 updates; save and serial fixed24 inference/decode/scoring at
  **0, 500, 1000, 1500 and 2000**. This is a named one-ID diagnostic, not fixed96
  or evidence of generalization to unseen identities.

No fabricated negative identities, contrastive bank, or cached denoiser states
are used. The proposed larger-rank 48/80GB profiles and ROI384 fallback are not
qualified by this pilot.

## Reproduction

Configuration: `configs/FLUX2_local_4b_one_id.yaml`. DINO revision:
`ed25f3a31f01632728cabb09d1542f84ab7b0056`; ArcFace is the existing pinned
`buffalo_l/w600k_r50.onnx`. Preparation records source images, erased boxes,
feature hashes and reference encoder identity. Cache identity is included in
adapter checkpoint identity. Reference encoders run during preparation and are
not resident alongside the training backbone.

```bash
envs/metrics/bin/python -m scripts.prepare_flux1_identity \
  --config configs/FLUX2_local_4b_one_id.yaml --threads 2
envs/metrics/bin/python -m scripts.prepare_flux2 \
  --config configs/FLUX2_local_4b_one_id.yaml --references
envs/flux-toolkit/bin/python -m scripts.prepare_flux2 \
  --config configs/FLUX2_local_4b_one_id.yaml
PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 \
  envs/flux-toolkit/bin/python -m scripts.check_online_face_ba \
  --config configs/FLUX2_local_4b_one_id.yaml --output PATH_TO_NEW_ADMISSION
```

Initialization requires successful pretrained admission and exact fresh-process
optimizer/scheduler/RNG replay. Use the existing `scripts.online_face_ba init`
and `scripts.run_online_face_ba` controller only after those gates pass. The
controller retains immutable sources, checkpoints, masks and Comet identity.
Use the same allocator environment for admission and the production controller.

The local Linux filesystem was nearly full during implementation. New caches,
feature weights, admission artifacts and run outputs are stored beneath
`/mnt/c/Users/ogure/rsrch_flux2`, with ignored project symlinks for feature paths.
No historical runs were removed.

## Measured admission (2026-10-05)

All 32 focused tests passed. Pretrained native/BA-off and empty-mask equality,
frozen-weight equality, native-face/excluded-context invariance, reference
identity sensitivity and finite branch/identity gradients passed. Two updates
with accumulation 4 reserved at most **11.561 GiB (72.29%)**. Fresh-process
resume reproduced parameters, Adam state, scheduler, cursor and RNG exactly.
Every one of the 19 training pairs also passed a forced identity-loss backward.
ArcFace conversion differed from ONNX by at most 0.001491; the weighted
identity/flow gradient-norm ratio on the admission probe was 0.0523.

The initial accumulated-memory check without frozen K/V CPU offload failed at
14.764 GiB (92.32%); that configuration was not launched.

A 20-step CFG-4 generation/decode probe passed finite-output and exact-exterior
checks, but its nearly untrained face was patchy and poorly formed. This is a
functional admission, not a quality result. The production pilot starts fresh
from zero-initialized branch outputs, independently of admission checkpoints.

Run: `runs/FLUX2_local_4b_one_id_20261005`. Immutable Comet key:
`7c88a5c362164fdbad2df18c2f629153`, project `rsrch_new`.
The machine-readable evidence and source hashes are in `FLUX2_ADMISSION.json`.
The legacy `b_gradient_norm`/`updated_b_matrices` logs apply to older LoRA modules;
FLUX2 uses the overall gradient norm and its admission parameter inventory.
