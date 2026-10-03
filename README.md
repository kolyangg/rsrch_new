# Branched attention for FLUX.2 and Qwen-Image-2.1

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

## Verification and scope

Both pretrained backbones available locally passed branch initialization parity, two real optimizer updates, finite/nonzero gradients at all 16 branch B matrices, and exact save/resume checks. Full-step initialized-branch checks generated one image each for FLUX 4B at 768 pixels and Qwen 7B at 768 and 1024 pixels.

The complete native panels at 768 pixels contain 96 images and 96 usable output masks per backbone. All 96 corresponding FLUX/Qwen pixel masks differ, and both mask sets were used in identity scoring. Qwen's expanded 1024-pixel profile has the initial one-item mask; its full panel remains for the larger host. The FLUX 9B site layout passes structural/gradient checks, but its pretrained weights exceed this 16 GB GPU, so its image and training admission tests also need the larger host.

Measured results, Comet links, full-panel mask coverage and outstanding hardware checks are recorded in [implementation notes](plans/260930/IMPLEMENTATION_NOTES.md). These are engineering checks, not evidence of a trained identity-quality gain. The original [implementation plan](plans/260930/CL39_Qwen_FLUX_48GB_80GB_Implementation_Plan.md) remains unchanged.

## Current one-ID setting: BA inside the full denoiser

The approved replacement for the cached face heads trains rank128 branch-local
Q/K/V/output LoRA at eight FLUX Base4B sites, with the complete frozen denoiser
in the gradient path and fresh noise/timesteps every microbatch. Only25.17M BA
parameters train. At768px/ref512, accumulation4 uses under10GiB in the bounded
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
