# Local16GB online face BA — 2 October 2026

## Implemented experiment

`flux4b_oneid_online_face_qkvo_r128_768` implements the setting approved after
the [critical review](ONE_ID_CRITICAL_REVIEW_20261002.md). It runs the complete
pinned FLUX.2-klein Base4B denoiser for every training microbatch. Only the
branch-local Q/K/V/output LoRA parameters train. Native transformer parameters,
text encoder and VAE remain frozen.

- Eight sites: dual blocks `1,2,3,4`; single blocks `3,8,13,18` (zero based).
- Rank128, alpha128: **25,165,824 parameters /64 tensors**.
- Native Q/K/V are reused; zero-initialized LoRA deltas are applied before native
  Q/K normalization and original RoPE. K/V use only selected reference-face
  tokens, preserving original reference positions.
- Target face queries receive the reference attention message. Other queries
  retain their native attention message at that site. The native attention
  output projection, MLP, modulation gates, residuals and final velocity layer
  remain active. The output LoRA contributes after the native output projection,
  before its modulation gate.
- BA-off is native. BA-on at step0 is an untrained, native-initialized reference
  route; it is intentionally a separate baseline from native generation.

The implementation is in `ba_dit/nn/masked_face_attention.py`. Small additions to
the recorded native patch pass the face mask through the immutable checkpointed
block context and apply the output delta at the correct projection point.
Existing reference-delta BA follows its original path. The prior patch is
preserved at `runs/critical_review_20261002/runtime_before_online_qkvo.patch`;
older run artifacts/snapshots remain intact. Their strict live-source checks
will reject resuming against the newly extended runtime without restoring their
recorded source.

## Training and validation contract

Configuration: `configs/FLUX1_local_4b_one_id.yaml`.

| Setting | Value |
| --- | --- |
| Geometry | Target768px, reference512px |
| Data | Original19 distinct target/reference pairs, all identity51 |
| Reference handling | Original cross-view training pairs; validation retains its one fixed reference image |
| Noise/timesteps | Newly sampled each microbatch, pinned native flow scheduler |
| Objective | Face-mask-normalized native velocity MSE; no ArcFace auxiliary |
| Masks | Training-photo boxes for training; reviewed native-generated boxes for inference;16px outer feather |
| Input cache | Immutable text embeddings and VAE latents only |
| Precision | BF16 native transformer; FP32 LoRA parameters and Adam state |
| Optimizer | AdamW5e-5,100-update warmup, clip1, weight decay0 |
| Batch | Microbatch1, accumulation4, effective batch4 |
| Memory | Activation checkpointing; require peak reserved below90% |
| Validation | Named fixed24 panel,20 Euler steps, CFG4 in both patched lanes |
| Schedule | Generate/score at0,500,1000,2000; stop after2k final scoring |

The earlier cached heads forced one fixed reference into every training case.
This run restores the existing dataset's distinct cross-view pairs, matching
the native training loader. They are all the same person. This is an explicit
additional change in the new baseline, not an isolated cache-policy ablation.
The generic `lora` section remains an inactive schema default; only the64
branch tensors appear in the optimizer inventory.

Validation generates new images from prompt/reference and random noise. It
never loads target photographs. The exact same24 native images, masks and
latents are copied with checked hashes from the reviewed panel. The native
scene supplies noised exterior context during denoising, and final pixel
composition preserves its exterior exactly. This preservation is imposed by
the protocol, not a claim that BA learned background locality.

## Measured admission checks

Artifacts: `runs/online_qkvo_admission_20261002/`.

- Pretrained native BA-off parity before/after updates: exact.
- A zero face mask preserves the complete native prediction exactly.
- All64 branch tensors update. First B gradients and second-step A/B gradients
  are finite/nonzero at all eight sites.
- SHA256 over every frozen parameter is unchanged after two real updates.
- Largest training face support tested (`oneid_train_46`,48.12% coverage):
  9.574GiB peak reserved; trained prediction changes on fixed inputs.
- Production-shaped accumulation4 replay: uninterrupted two updates versus one
  update plus a fresh process gives bit-exact adapter tensors, Adam moments,
  scheduler, cursor and CPU/CUDA/Python RNG states.
- Accumulation4 peak reserved:9.682GiB maximum across replay workers, below90%
  of the15.992GiB device. Cold/warm updates took7.97–9.13s.
- Focused block tests confirm exact text/reference/background rows at each
  insertion site, including nonzero output LoRA and different batch masks.
  Existing FLUX reference-delta seam and background-composition checks pass.
  Tests were called directly with `runpy` because pytest is not installed.

These are implementation/hardware checks. Identity or quality improvement is
not yet established; the fixed24 generated images decide that.

## Run and operation

Completed run: `runs/flux4b_oneid_online_face_qkvo_r128_768_20261002`.
Comet key: **`c955799c57d94fd7a234dec331323e06`** in project`rsrch_new`:
[experiment](https://www.comet.com/nikolay-2104/rsrch-new/c955799c57d94fd7a234dec331323e06).
The serial controller first generates and scores step0, then trains to500,
validates, trains to1000, validates, trains to2000, and performs final scoring.
It preserves every checkpoint and records the best scored one. Training logs
retain all metrics in Comet/JSONL; console output is compact progress with ETA.

```bash
# Admission check in a new output directory:
envs/flux-toolkit/bin/python -m scripts.check_online_face_ba \
  --config configs/FLUX1_local_4b_one_id.yaml \
  --output runs/NEW_ADMISSION_NAME

# New experiment after admission passes:
bash scripts/run_online_face_ba.sh runs/NEW_RUN_NAME runs/NEW_ADMISSION_NAME
```

The controller holds `runs/face_flow_gpu.lock`, runs GPU stages serially and
stores PID/status/completed-command records. Each stage verifies immutable source,
config, mask and dataset hashes. Checkpoints retain optimizer/scheduler/RNG/cursor
state. Run source snapshots contain the imported branch, training, inference,
logging and native patch implementations. The shell launcher does not detach
itself; the current run was launched in a detached process group with a log at
`runs/flux4b_oneid_online_face_qkvo_r128_768_20261002.log`.

Previous head experiments and their best checkpoints are preserved. This run
does not automatically extend beyond2000 or assert convergence.

### Confirmed startup, 15:21 UTC

The controller completed step-zero generation, decoding, scoring and paired
face sheets, then started `train_500` from checkpoint0. Six optimizer updates
have completed, with finite gradients and all32 B matrices updated at step1.
The first two losses exactly match the admission replay (`0.988909826`,
`1.050353393`). Startup evidence is in `startup_confirmation.json` in the run.

- Mean across the first six updates:10.10s/update, effective batch4.
- Peak reserved9.641GiB (60.29% of device capacity); GPU utilization92% at
  the final startup sample. Device-wide memory use was11118MiB.
- Early estimate:5.6h of remaining training, plus three validation panels
  (step-zero generation/scoring took about15min). Throughput can change as
  the laptop heats up; the progress/ETA metrics update continuously.
- Step-zero ID_sim **0.146024**, CLIP text score27.702707, versus native
  ID_sim0.331399. All24 paired crops were inspected: the untrained full-face
  route produces substantial distortion and repeated facial features in many
  images. This is a weak baseline, not evidence of identity learning.
- All24 decoded images preserve exterior pixels exactly; mean difference
  within face support is31.57/255 relative to native. Inference peak reserved
  memory was8.525GiB.

Startup monitoring ended after these checks. The serial controller continues
through the documented2k schedule; trained image quality remains to be measured.

### Step1000 scoring and continuation verified, 2 October

The apparent stop at1000 was the scheduled validation stage. The original
controller and inference worker were alive with19/24 generations complete when
checked. No restart or duplicate GPU job was needed. Generation, decoding,
scoring and comparison sheets finished successfully in the same Comet run.

| Step | ID_sim | CLIP text score |
| --- | --- | --- |
| 0 | 0.146024 | 27.702707 |
| 500 | 0.365766 | 27.249032 |
| 1000 | 0.430254 | 26.967033 |

The Comet read API confirms all three metric points are stored remotely
(`comet_metrics_1000_verified.json`). Step1000 improves ID_sim on22/24 images
versus500 and24/24 versus0. All24 paired face crops were inspected: the initial
distortions are greatly reduced, although softness and some changes to facial
details/accessories remain. Exterior pixels stay exact. This is a one-ID
diagnostic, not evidence of generalization to other identities.

The checkpoint contains finite64 adapter tensors and Adam moments at update1000,
scheduler epoch1000, data cursor4000 and saved RNG states. Immutable sources,
configuration, manifests and masks match. The controller launched
`train --step 2000 --resume 1000`; updates1001–1005 were observed with finite
gradients, retained LR5e-5 and the expected sample cursor. Peak reserved memory
is9.859GiB (61.65%). `resume_1000_confirmation.json` records the actual process
and resume evidence. The controller continues to2000, then performs the complete
fixed24 validation/scoring before stopping. Startup monitoring ended again.

## Completed results, reviewed 3 October 2026

The controller finished all2000 updates and all four fixed24 validations at
23:40:58 UTC on2 October. No training or validation worker remains.

| Model/checkpoint | ID_sim | CLIP text score |
| --- | --- | --- |
| Native on the same panel | 0.331399 | 28.0729 |
| Untrained BA | 0.146024 | 27.702707 |
| BA500 | 0.365766 | 27.249032 |
| BA1000, best ID_sim | **0.430254** | 26.967033 |
| BA2000 | 0.419023 | 27.321863 |

All2000 loss/gradient records are finite; peak reserved memory was9.859GiB.
The final24 images have exact exterior preservation. All three final paired
face sheets were inspected: faces are substantially improved from step0, while
softness and accessory/lighting changes remain. At2000,22/24 cases improve
over0 and17/24 over500. Fourteen improve versus1000, but the mean falls0.011231;
the ten regressions outweigh those gains. Retain the best1000 checkpoint. This
single trajectory does not establish convergence or prove overfitting as the
cause of the small aggregate regression.

Best adapter SHA256:
`379408b6122082919783ccaecaf3d8d80d3ecac479d2216ce1e27d1519ec8890`.
Final2000 adapter SHA256:
`a1ae72e041fe8969f8b333bffb6d60e998145c028a0fc4530ad211318f459b6c`.
The committed [result record](ONLINE_FACE_BA_RESULTS.json) includes exact source
hashes, configuration, validation metrics and checkpoint provenance. All20 run
source hashes still match the implementation used for training.

### Artifact prerequisites for reproducing this exact run

Git contains the implementation and pinned dependency/weight manifests. It does
not contain model weights, photographs, embeddings, cached tensors or checkpoints.
The current specialized runner assumes a prepared project workspace:

- Set up the FLUX and metrics environments using `scripts/setup_machine.sh`;
  retain the pinned sources/patch and weight revisions in the configuration.
- Import the original one-ID data with `scripts/import_one_id.py`. The fixed24
  manifest is the12-prompt panel at seeds0 and1, with `_s1` appended to the
  second group of sample IDs. Retain the recorded manifest hash and order.
- Prepare native text/VAE conditioning caches for both splits and negative
  validation prompts, using the experiment configuration.
- Restore the reviewed native bundle at
  `runs/flux4b_deep_identity1024_det_20261001/`: `native/` (24 PNGs and24 latent
  files), `routing_masks.json`, `ownership_boxes.json`, `mask_overlays.png`.
  Initialization checks image/latent hashes, geometry, prompts/seeds, sampler,
  weight revisions and panel hash. This bundle supplies native scene context;
  the older trained face-head checkpoint is not loaded.
- Run the admission script in a fresh directory, then the launcher above with a
  fresh run name. Resume an existing completed segment with its own immutable
  run folder and full checkpoint, preserving optimizer/scheduler/RNG/cursor.

The one-ID controller deliberately asserts19 training pairs and24 validation
items and uses the above native bundle. A multi-ID run requires a generalized
controller and matching native validation bundle; editing only the manifest
path is insufficient. The masked Q/K/V/O installer currently supports4B only.

## Recommended next 48GB experiment — proposal, not launched

**Keep FLUX Base4B and test multi-ID generalization first.** The current result
shows that BA can learn this person. A larger model trained on the same person
would leave the central question of reference conditioning for unseen people
unanswered. Use a fresh branch initialization, with the frozen native4B weights;
do not use the one-ID adapter as the primary multi-ID starting point.

1. Use the adjusted Large Dataset first (the existing smaller download), with
   distinct target/reference photos of each identity. Audit canonical identity
   aliases, shared images and duplicates before splitting. Balance identities
   and rotate reference views. Start with a reproducible diverse4,096-pair pilot
   to validate this path, then a separately named full-data run using the same
   architecture and objectives. Cosmic is a later dataset comparison.
2. Keep target768/reference512, the eight rank128 Q/K/V/O sites, fresh noise and
   native timestep draws, masked flow loss, BF16 frozen backbone, FP32 adapters
   and LR5e-5 initially. Native weights, VAE and encoder remain frozen. A proposed
   first budget is10k optimizer updates with best-checkpoint selection; extend
   only if held-out identity/quality scores support it.
3. Benchmark actual microbatches2/4/8 on the48GB host, with compatible token
   layouts, and use effective batch8 across comparisons. Choose by samples/sec
   with peak reserved below90%; these batch sizes are proposals, not measured
   fits. The current training loop uses microbatch1, so batching/bucketing needs
   implementation. Cache text/VAE inputs only; keep the full denoiser in the
   gradient path. No larger rank is required simply to occupy spare VRAM.
4. Use the original fixed96 order/prompts/seeds and metric definitions at0 and
   every2000 updates. It contains only8 identities: all must be excluded from
   training by identity and image content. New photographs of training people
   do not establish unseen-identity generalization. For a stronger final test,
   add a separate locked panel with more entirely unseen identities and reference
   photographs; do not replace or tune on that test panel.
5. Preserve the current two-pass spatial protocol. Generate a new frozen native
   scene/mask bundle for the fixed96 panel at the declared20-step/CFG4 sampling
   setting. These20-step outputs are separate from the existing50-step native96
   baseline; never reuse masks across schedules or backbones. Target-photo masks
   remain training supervision only.
6. Compare native BA-off, untrained routed BA and trained BA under the same
   geometry/sampling/prompt protocol. Add fixed-input reference-swap controls to
   check reference conditioning, and a subset with shuffled branch reference
   K/V while preserving native conditioning to isolate the added branch. Report
   per-identity/paired ID_sim, CLIP, face-detection coverage, reviewed face crops
   and face-quality metrics. Report variation across identities rather than
   treating all96 images as independent people. Exact exterior pixels are an
   imposed composition rule, not a learned-quality metric.

After the4B multi-ID baseline is useful, compare Base9B on the same split,
resolution, effective batch, training exposure and evaluation. Both Base models
are undistilled models intended for fine-tuning according to the official
[4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) and
[9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) cards.
The9B card's approximately29GB figure describes model usage, not measured BA
training memory. A48GB adapter run is plausible with staged encoders/VAE and
checkpointing, but requires an actual forward/backward/resume/validation memory
check. It also requires the9B masked-branch integration, its own text caches and
native masks. Width-specific4B adapters cannot be loaded into9B. A larger model
is the subsequent capacity experiment, after generalization is demonstrated.
