# FLUX1 — masked branched attention inside FLUX

**FLUX1** is the experiment previously called `online_masked_qkvo_v1`: rank-128
branch-local **Q/K/V/output LoRA at eight attention sites**, trained through the
complete frozen FLUX denoiser with fresh noise and timesteps. Only BA parameters
train. The backbone, text encoder and VAE stay frozen; generic backbone LoRA is
disabled. Naming the experiment FLUX1 does **not** change its attention equations,
initialization, masks, loss, data, sampling or evaluation settings.

The native attention path stays intact when BA is off. With BA on, the branch
routes reference-face attention into the target face region. Training target
masks are supervision; inference uses frozen masks from native backbone
**generations**, never target photographs. Native exterior pixels are preserved
explicitly. Training encodes examples online by default; only the fixed
validation inputs are cached. The earlier cached face-head experiments and
rank-16 reference-read-delta profiles below are separate historical experiments.

### Architecture charts

The existing architecture report is now titled **FLUX1**. Its measured results
and model dimensions describe the completed **4B one-ID variant**, not the 9B
Vast run. The same BA mechanism is used by the Vast and cluster variants; the
backbone width, attention-site indices, precision and training panels differ.

- [FLUX1 whole-model chart](reports/261003_online_face_ba/FLUX1_whole_model.svg)
- [FLUX1 branched-attention chart](reports/261003_online_face_ba/FLUX1_branched_attention.svg)
- [FLUX1 inference/mask protocol](reports/261003_online_face_ba/FLUX1_inference_protocol.svg)
- [Report source and rebuild instructions](reports/261003_online_face_ba/rebuild.txt)

The rebuild writes `reports/261003_online_face_ba/FLUX1_architecture_and_one_id_results.pdf`.
It uses the original immutable one-ID evidence; private images/checkpoints stay
outside Git. The report's equations and numerical results have not changed.

## FLUX1 deployment variants

| Variant | Config | Actual setup | Validation |
|---|---|---|---|
| **Vast — current 9B** | [FLUX1_vast_9b.yaml](configs/FLUX1_vast_9b.yaml) | GB10, BF16, microbatch 1 / accumulation 1, 768 px / reference 512; 2k pilot continued to8k | Fixed12 during training; native96 and trained8k full96 separately |
| **HSE cluster — current 4B** | [FLUX1_cluster_4b.yaml](configs/clust/FLUX1_cluster_4b.yaml) | Two V100s: GPU 0 denoiser, GPU 1 frozen text encoder; one training worker; selective FP16 with FP32 BA/residual safeguards; batch 1 | Original fixed96 at 0 / every 2k; configured 20k target |
| **Local — 4B one-ID** | [FLUX1_local_4b_one_id.yaml](configs/FLUX1_local_4b_one_id.yaml) | 16GB, full-denoiser BA-only, accumulation 4, 2k | Named fixed24 one-ID panel |
| **Vast — 4B alternative** | [FLUX1_vast_4b.yaml](configs/FLUX1_vast_4b.yaml) | 48GB proposal, online conditioning, effective batch 8 | Fixed96 at 0 / every 2k |

These are variants of one attention mechanism, **not matched hardware or quality
benchmarks**. On the Vast fixed12 panel, ID_sim was 0.4341 at 2k and 0.4165 at 8k;
2k remains best. Cluster progress, interruption/resume and exact deployment
versions are recorded in [implementation notes](plans/260930/IMPLEMENTATION_NOTES.md).

### FLUX1 on Vast

Environment/weights: `scripts/setup_machine.sh` (uv), using `flux80` for 9B or
`flux48` for 4B. Data download/import: `scripts/prepare_dataset.py`,
`scripts/download_dataset.py`; workstation transfer fallback:
`scripts/sync_large_dataset_vast.py`. See [deployment/data instructions](docs/deployment.md).

```bash
# New 9B pilot, after setup and data import; creates a fresh run.
envs/flux-toolkit/bin/python -m scripts.run_flux9b_training_first \
  --config configs/FLUX1_vast_9b.yaml --run runs/FLUX1_vast_9b_2k

# Full-state continuation of that completed 2k run, with validations at 4k/6k/8k.
envs/flux-toolkit/bin/python -m scripts.continue_flux9b \
  --parent runs/FLUX1_vast_9b_2k --run runs/FLUX1_vast_9b_8k --target 8000

# 4B alternative: preflight/admission, native masks, then training.
bash scripts/run_flux4b_multi_id.sh --config configs/FLUX1_vast_4b.yaml \
  --run runs/FLUX1_vast_4b --images-root /workspace/datasets/large_dataset
```

`run_flux9b_training_first.py` is the measured fast pilot protocol: it saves
checkpoint 0, trains to 1k, then scores 0/1k and finally 2k. `continue_flux9b.py`
restores optimizer/RNG/data cursor; only total steps and validation cadence can
change. `finalize_flux9b_remote.py` and `finalize_flux9b_local.py` implement the
**existing instance 53994096 completion/download/Comet/stop workflow**; they are
bound to that run and must not be used as generic new-machine launchers.

### FLUX1 on the HSE cluster

Environment: `scripts/setup_clust_v100.sh` and `scripts/activate_clust_env.sh`.
Dataset locations/import: `configs/clust/training_sources.yaml` and
`scripts/import_training_dataset.py`. The cluster launcher is separate from
Vast's training-first pilot:

```bash
# On the cluster after environment/data setup; scheduler allocation is required.
mkdir -p logs/clust
sbatch jobs/flux4b_clust_2v100_amp.sbatch
```

- `jobs/flux4b_clust_2v100_amp.sbatch` selects **FLUX1_cluster_4b.yaml** and runs
  `scripts/run_clust_v100.sh` → `scripts/run_multi_id_face_ba.py`.
- `jobs/clust_comet.sbatch` / `scripts/upload_clust_comet.py` publish job status.
  `scripts/sync_clust_comet.py` relays metrics/images from the workstation when
  compute-node uploads fail. All publishers reuse the saved Comet key.
- `scripts/check_clust_v100.py` and `scripts/check_online_face_ba.py` verify the
  environment and pretrained training invariants. Benchmark/proposal configs
  ending `_fp32` or `_bf16_proposed` are not the active selective-FP16 setup.
- Two V100s do **not** mean two data-parallel workers in this configuration:
  the second GPU holds the frozen text encoder.

### Config rename and existing checkpoints

FLUX1 configs are the current names for fresh launches. Saved run directories,
Comet keys, source snapshots and checkpoint configs keep their original names
and hashes. Use the exact frozen runtime for an existing checkpoint; renaming
its saved config in place would invalidate resume checks. The source versions
used on each machine are preserved in Git as **`FLUX1-vast-runtime`** and
**`FLUX1-cluster-runtime`**; [deployment provenance](docs/FLUX1_DEPLOYMENTS.md)
records their source hashes and the old-to-new config map. `main` contains the
latest maintained code and FLUX1 naming.

## Earlier project profiles and experiments

The sections below describe the original rank-16 reference-read-delta setup,
Qwen profiles and historical diagnostics. They are retained for reproducibility.

### Original branched-attention setup for FLUX.2 and Qwen-Image-2.1

A reference-read correction for native FLUX.2-klein Base and Qwen-Image-2.1. The branch adds `gamma * (adapted_reference_read - native_reference_read)` at eight attention sites. The original attention, conditioning, key normalization and RoPE remain in place. Zero-initialized adapters preserve native outputs.

## Profiles

| Launcher profile | Backbone | Target pixels | Reference area budget | Intended GPU |
| --- | --- | --- | --- | --- |
| `flux48` | FLUX.2-klein **Base 4B** | 768 × 768 | 512² | 48 GB |
| `flux80` | FLUX.2-klein **Base 9B** | 1024 × 1024 | 768² | 80 GB |
| `qwen48` | Qwen-Image-2.1 **7B** | 768 × 768 | 512² | 48 GB |
| `qwen80` | The same Qwen **7B** | 1024 × 1024 | 768² | 80 GB |

`flux80-matched` and `qwen80-matched` retain the 768/512 geometry for comparisons. Reference images keep the native aspect-ratio rules; the area budget is not a forced square crop.

The six research YAMLs are executable `ba_dit` configs. They use BF16 backbones, FP32 adapters, rank-16 branches, optional rank-8 attention LoRA, gradient checkpointing, batch size 1, accumulation 8, and a 90% reserved-VRAM training gate. Modes are `native`, `branch_only`, `lora_only`, and `lora_plus_branch`.

## Implementation

- [ba_dit/nn/reference_read_delta.py](ba_dit/nn/reference_read_delta.py): shared reference attention arithmetic, identical key support in both reads, target-only corrections.
- [ba_dit/backends/attention.py](ba_dit/backends/attention.py): native projection/normalization/RoPE seams. Small recorded source patches import these helpers.
- [ba_dit/adapters.py](ba_dit/adapters.py): explicit sites and optimizer inventory. FLUX fused LoRA slices exclude MLP weights.
- [ba_dit/data](ba_dit/data): paired manifests, independent target/reference geometry, content-addressed disk caches.
- [ba_dit/training.py](ba_dit/training.py) and [ba_dit/inference.py](ba_dit/inference.py): the same patched backbone for training and generation.
- [ba_dit/validation_masks.py](ba_dit/validation_masks.py): separate output face masks for each backbone, revision, resolution and validation schedule.

Text encoding, VAE encoding, transformer training/denoising, and VAE decoding run in separate processes. Training loads one cached pair at a time. Checkpoints contain adapters, optimizer/scheduler state, CPU/CUDA/Python RNG, sample cursor, data fingerprints, source/adapter fingerprints and a relocatable resume config.

## New Linux / Vast machine

See [docs/deployment.md](docs/deployment.md) for source/data transfer and archive import. For an existing checkout:

```bash
scripts/setup_machine.sh flux48 --download-weights
scripts/setup_machine.sh qwen80 --download-weights
```

Choose the required profile. The installer uses `uv` to create pinned Python 3.11.13 environments, applies the pinned source patches, checks the adapter invariants, installs separate CPU environments for identity/CLIP and face-quality metrics, and optionally downloads only locked model files. FLUX uses Torch 2.13 and Qwen uses Torch 2.10; setup selects the matching CUDA 12.6 wheel on driver 565.77 or the original CUDA 13.0/12.8 wheel on newer drivers. The host needs `git`, `curl`, a C compiler and a compatible NVIDIA driver.

Set `COMET_API_KEY` in untracked `.env` or the environment. Comet project `rsrch_new` appears as `rsrch-new` in URLs. Hugging Face downloads use `HF_TOKEN` or the normal local HF login cache.

## Validation and backbone-specific face masks

The original fixed **96 items**, prompts, image order and seed 0 are preserved. Reference pixels are hash-checked against the manifest.

```bash
# Full native baseline; generates this backbone's output masks and scores with them.
scripts/run_profile.sh flux48 infer --mode native --quality-metrics \
  --output-dir runs/flux48_native_full96

# Untrained branch and exact native comparison; --limit is an explicit smoke subset.
scripts/run_profile.sh qwen48 infer --mode branch_only --limit 1 \
  --compare-native --quality-metrics --output-dir runs/qwen48_branch_initial

# Full validation after training; frozen native-baseline masks are reused.
scripts/run_profile.sh flux48 infer --mode branch_only \
  --checkpoint runs/flux48_branch/checkpoint-002000 --quality-metrics \
  --output-dir runs/flux48_branch_step2000
```

Two mask sets have separate purposes:

1. **Reference-token masks** are projected through each backbone's native reference preprocessing. They select the branch's reference keys during training and validation. `scripts/build_face_masks.py` exports both pixel and token-grid masks using the runtime geometry code.
2. **Generated-face masks** come from each backbone's own untrained/native baseline images. They live under `data/validation/output_masks/<architecture>/<geometry-and-schedule>/`, with per-image provenance, pixel masks and token masks. Validation scoring consumes these PNG masks. A different backbone, resolution, reference budget, step count or guidance setting uses a different set. No-face and ambiguous detections get empty masks and explicit diagnostics; they can be reviewed with `build_validation_output_masks.py --overrides boxes.json`.

Generated-face masks are evaluation labels, following the attached plan's reference-only generation protocol. The initial branch applies its correction to all target queries. The historical PhotoMaker output boxes are retained as source metadata and are not used for these runs.

Quality scoring is enabled by default (`--no-quality-metrics` explicitly disables it). It runs the previous project's `id_sim` subject metric (IoU .05, ambiguity margin .02), `id_sim_legacy_best`, face diagnostics, and CLIP `ViT-L/14@336px` logits. The seven original `face_quality/` curves use PyIQA 0.1.15: TOPIQ-Face, TOPIQ, MUSIQ and MANIQA-PIPAL all receive the same largest-face crop, with 25% padding per side and a 512-pixel Lanczos resize. Missing faces and alignment failures have explicit coverage diagnostics. PyIQA has its own environment to preserve the legacy CLIP package.

Original identity embeddings are included in the private data bundle. Images, configs, per-image reports and metrics go to Comet; the immutable experiment key is saved at startup. Every output folder has JSON and CSV results. Resume a partial validation with the same command plus `--resume-validation`. `--skip-output-masks` is reserved for tensor-parity diagnostics and also disables scoring unless explicitly contradicted.

## Training

Both **Cosmic Large** and **adjusted Large Dataset** are supported on every GPU profile. [Dataset preparation](docs/deployment.md#2-download-and-import-the-training-data) uses the original metadata and separate manifests. The two original Cosmic Drive links are saved in [locks/datasets.json](locks/datasets.json); Large accepts its archive link through `--url` / `LARGE_DATASET_URL` or an existing image folder. Its original link has not yet been recovered. The later BigCelebs release is separate from this Large preset. The included 38-pair sample is a plumbing smoke set.

```bash
# After transferring/extracting the private training-metadata bundle:
envs/flux-toolkit/bin/python scripts/prepare_dataset.py cosmic
envs/flux-toolkit/bin/python scripts/prepare_dataset.py large --url "$LARGE_DATASET_URL"

# Choose train_pairs_cosmic.jsonl or train_pairs_large.jsonl for any profile.
scripts/run_profile.sh flux48 preflight --split train \
  --train-manifest data/train_pairs_cosmic.jsonl
scripts/run_profile.sh flux48 precompute --split train \
  --train-manifest data/train_pairs_cosmic.jsonl
scripts/run_profile.sh flux48 train --mode branch_only --quality-metrics \
  --train-manifest data/train_pairs_cosmic.jsonl --run-name flux48_cosmic_branch

# Same state, noise stream, pairing position and Comet experiment:
scripts/run_profile.sh flux48 train --mode branch_only --quality-metrics \
  --resume runs/flux48_cosmic_branch/checkpoint-000500
```

Normal training validates all 96 images at step 0 and every 2,000 optimizer updates, after the training process exits. Checkpoints are written every 500 updates and at each segment boundary. Deterministic CUDA training is enabled for reproducible resume; registration is followed by a fresh training RNG seed so adapter modes get the same noise/timestep stream.

Use `--init-adapter PATH` to start a new curriculum with compatible adapters and a fresh optimizer, for example moving Qwen to the expanded 80 GB geometry. `--resume` restores the exact saved experiment; it does not silently adopt changed hyperparameters. FLUX 4B adapters cannot be loaded into 9B.

```bash
# Explicit local 16 GB smoke; reduced geometry is recorded in the run config.
scripts/run_profile.sh qwen48 train --mode branch_only --smoke-steps 2 \
  --train-manifest data/train_pairs_smoke.jsonl --limit 2 --grad-accum 1 \
  --target-size 256 256 --allow-small-gpu --no-comet \
  --output-dir runs/qwen_local_smoke
```

## Fast face-focused one-ID diagnostic (FLUX 4B)

```bash
bash scripts/run_face_diagnostic.sh runs/flux4b_face_one_id_fast_NAME
```

This separate experiment uses one final-block reference-face attention read,
rank-16 output LoRA and a learned face gate (101,377 trainable parameters).
All native model weights stay frozen. Target boxes supervise training only;
inference predicts face locations from model features. Four same-ID image pairs
and fixed noise/timestep inputs allow exact frozen-prefix caching for a fast
overfit check. A separate set of noise seeds tests the update beyond fitted
noise. Both CFG lanes use BA.

The 512 px/two-prompt validation is a named diagnostic, separate from the
original panels. The launcher writes paired images, inference gate overlays,
learning curves, checkpoint/source audits and a `review.html` in the run folder.
See [face BA diagnostic](plans/260930/FACE_BA_DIAGNOSTIC.md) for measured speed,
parity checks, mask roles and limitations. Use the original profiles above for
the original multi-site experiment.

### Stronger local one-ID experiment

```bash
bash scripts/run_face_suffix.sh runs/flux4b_face_one_id_strong_NAME
```

This separate 768 px diagnostic trains eight late BA sites at rank 256 on all
19 one-ID pairs for 2,000 updates. Only BA output LoRAs and their face routers
train. The frozen prefix is cached; the complete eight-block suffix remains
differentiable. The launcher measures batch throughput and memory, chooses a
safe configuration for the local GPU, checks process resume, and runs serial
four-prompt validation at 0, 1,000 and 2,000. Caches use a finite bank of noise
inputs; this is an overfit diagnostic. See
[strong face BA experiment](plans/260930/FACE_BA_STRONG.md).

### BA-only faces from noise

```bash
bash scripts/run_face_crop_flow.sh runs/flux4b_ba_only_face_crop_NAME
```

This controlled 256 px face-crop experiment replaces the entire native velocity
prediction with a small BA network. Its output projections start at zero, so
step-zero samples remain noise. Only 2.41 million parameters in the added
attention/query/noisy-latent paths train; FLUX is a frozen feature extractor.
The launcher runs 2,000 cached-input updates, checks process resume and exact
cached/full predictions, and generates four seeds at 0/1,000/2,000 for Comet.
The completed trial changed noise into detected faces on all four seeds.
This demonstrates reconstruction on one ID, with rough image quality, and does
not establish full-scene face replacement or generalization. See
[BA-only face experiment](plans/260930/FACE_CROP_BA_ONLY.md) for the equations,
failed controls, reference-read ablation and measured results.

## Verification and scope

Both pretrained backbones available locally passed branch initialization parity, two real optimizer updates, finite/nonzero gradients at all 16 branch B matrices, and exact save/resume checks. Full-step initialized-branch checks generated one image each for FLUX 4B at 768 pixels and Qwen 7B at 768 and 1024 pixels.

The complete native panels at 768 pixels contain 96 images and 96 usable output masks per backbone. All 96 corresponding FLUX/Qwen pixel masks differ, and both mask sets were used in identity scoring. Qwen's expanded 1024-pixel profile has the initial one-item mask; its full panel remains for the larger host. The FLUX 9B site layout passes structural/gradient checks, but its pretrained weights exceed this 16 GB GPU, so its image and training admission tests also need the larger host.

Measured results, Comet links, full-panel mask coverage and outstanding hardware checks are recorded in [implementation notes](plans/260930/IMPLEMENTATION_NOTES.md). These are engineering checks, not evidence of a trained identity-quality gain. The original [implementation plan](plans/260930/CL39_Qwen_FLUX_48GB_80GB_Implementation_Plan.md) remains unchanged.

## Native background with BA-generated faces

The user-requested CL14-like diagnostic generates faces in the existing 24
prompted native scenes, using masks detected from those exact backbone images.
It trains only a small BA flow head; zero output starts the face region as noise.
Native exterior pixels are preserved explicitly after decoding. This is a
separately named two-pass generation protocol; target photographs are never
validation inputs. See [protocol and measured status](plans/260930/MASKED_FACE_FLOW.md).

```bash
bash scripts/run_masked_face_flow.sh runs/NEW_NAME
```

### Stronger reference refiner and convergence run

This historical local experiment freezes the best trained BA core and adds a
1024-wide, 16-head reference refiner (14.05M trainable parameters). It uses a
larger verified noise/sigma cache and measured batch128 throughput. The serial
controller selects the best checkpoint using the 24-image prompted ID score
and stops after two small gains. See
[architecture, benchmarks and measured results](plans/260930/REFERENCE_REFINER.md)
for the actual module, launch commands, stopping rule and Comet run.

### Deeper BA with direct identity supervision

The refiner above regressed after step 500. The next named experiment starts
from that best checkpoint, adds two reference-attention reads (30.83M trainable
BA parameters), and combines flow MSE with a differentiable face identity loss.
The backbone, trained 512-wide core, VAE and ArcFace remain frozen. Batch256
was measured below the local 16GB memory limit. Generated-image ID scores,
paired crops and the unchanged 24-image panel decide whether it improves.
See [identity-flow architecture, admission and launch](plans/260930/IDENTITY_FLOW.md).

## FLUX1 local one-ID results: BA inside the full denoiser

The approved replacement for the cached face heads trains rank128 branch-local
Q/K/V/output LoRA at eight FLUX Base4B sites, with the complete frozen denoiser
in the gradient path and fresh noise/timesteps every microbatch. Only25.17M BA
parameters train. At768 px / reference 512, accumulation 4 uses under10GiB in the bounded
local16GB admission checks. The fixed24 prompted panel runs serially at
0/500/1000/2000, using reviewed native-generated masks and exact exterior
composition. It stops after2000 and final scoring.

```bash
bash scripts/run_online_face_ba.sh runs/NEW_NAME runs/PASSED_ADMISSION_NAME
```

The completed run scored ID_sim 0.1460 / 0.3658 / 0.4303 / 0.4190 at
0 / 500 / 1000 / 2000 updates; the native baseline was 0.3314. The best scored
checkpoint is step1000. This is a same-identity diagnostic; performance on
unseen people has not been measured.

The launcher requires the prepared one-ID data, conditioning caches and frozen
native image/mask bundle. These private/generated artifacts are not in Git.
See [architecture, artifact prerequisites, results and proposed 48GB experiment](plans/260930/ONLINE_FACE_BA.md).
