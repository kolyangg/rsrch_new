# Implementation notes — 30 September 2026

The attached [implementation plan](CL39_Qwen_FLUX_48GB_80GB_Implementation_Plan.md) is the research specification. This file records what has actually run. The `ba_dit` branch seams are preliminary; there is **no working branched-attention trainer, cache precompute, full-panel evaluator, or checkpoint/resume path yet**. The four research YAMLs in `configs/` describe proposed runs and cannot be passed to the upstream trainers.

## Reproducible foundation

- Public repository: `https://github.com/kolyangg/rsrch_new`. Local checkout: `/home/kolyangg/rsrch_new`.
- Pinned Diffusers Qwen source: `fef717ffb01f407d2637584ed936c16db908587a`; pinned AI Toolkit FLUX source: `ecee894ed2b1f3716d9d7326693061ec1a3105bb`. Local patches in `patches/` reproduce the current modified source trees. They apply/reverse-check cleanly against the pinned sources.
- Weight locks under `locks/` capture selected filenames, resolved revisions, and byte counts. Local model files and separate Qwen/FLUX Python environments are ignored by Git.
- Imported the PhotoMaker validation source images, original 12 prompts, classes, and reference face boxes. `data/validation/manual_val_96.jsonl` fixes the first 8 sorted references × 12 prompts, seed 0, in 96 rows (SHA-256 `d769406704ddbcdf0bb09d408ad4929b302aaa84a12e295872618e3df9f09a65`). Twelve reference images stay local.
- `scripts/build_face_masks.py` produces face masks at reference-size settings 512 and 768 for each backend (48 metadata rows, 96 image/token mask files). FLUX uses its capped and center-cropped reference preprocessing; Qwen uses its 32-pixel-rounded resize. The recorded token grids are backend-specific. Generated-image boxes are never fed into inference.
- Imported a disjoint 3-identity, 38-pair training smoke sample from the previous repository. Paired exports exist for both upstream training systems. This is plumbing data, not a substitute for the final training set.

## Local native-model feasibility

Machine: NVIDIA GeForce RTX 4090 Laptop GPU, 16,376 MiB physical memory. Both measurements used one fixed panel item at 512 × 512, one reference image, BF16, CPU placement/offload where noted. They are **native, untrained baseline outputs**, not branch validation and not the full 96-item panel.

| Backbone | Run | Peak CUDA reserved | Time per sample | Comet |
| --- | --- | ---: | ---: | --- |
| FLUX.2-klein Base 4B | 50 steps, Toolkit native pipeline, CPU model offload | 7.836 GiB | 51.0 s | [experiment 724bdd15](https://www.comet.com/nikolay-2104/rsrch-new/724bdd15295e47a7a918943203a92d2e) |
| Qwen-Image-2.1 7B | 40 steps, Diffusers pipeline, encoder on CPU and transformer/VAE on GPU, KV reuse disabled | 15.998 GiB | 264.13 s | [experiment 1e939c0b](https://www.comet.com/nikolay-2104/rsrch-new/1e939c0b03d94f26a2f16031485cbc42) |

Outputs and exact telemetry are in ignored `runs/native_flux4b_local_50step/` and `runs/native_qwen21_local_40step/`. Each has `validation.json`, `00.png`, and the immutable Comet key record. Both patched source trees also passed a two-step native inference regression after branch insertion, at the same respective 7.836 and 15.998 GiB reserved peaks. The Qwen run leaves only about 0.0 GiB of *reserved-memory* headroom on this device; allocation and fragmentation can still change. A 16 GB Qwen training run or larger-resolution full-panel run is therefore not a credible plan. FLUX 4B inference fits, but that does not establish training memory fit. Neither FLUX 9B nor a 48/80 GB host was tested locally.

The prompts describe distinct scenes, but these first native image-editing outputs stayed close to the reference appearance; no improvement claim follows from this single example. The pair also has different step counts and pipeline implementations, so its timing is not a controlled speed comparison.

## Branch integration and checks

- `ReferenceReadDelta` registers FP32 low-rank key/value corrections with zero-initialized B matrices. It computes `R1−R0` over identical selected reference keys, applies the correction only to target query rows, and keeps the native projection path.
- The Qwen patch threads explicit branch context through its transformer and attention processor while preserving normal forward calls. A small synthetic Qwen processor check gave exact initialized forward/input-gradient parity and nonzero first B gradients. This is not a pretrained model branch validation.
- The FLUX patch threads target/reference indices through checkpointed double and single blocks and applies the correction before their native output projections. `tests/test_flux_branch_seam.py` passed exact initialized output parity for both block types and nonzero first B gradients. The shared reference-read unit checks passed. Native two-step inference passed after both backend patches.
- The first pretrained-panel item also completed a two-step **initialized branch-on** run with its 512-reference face mask on each patched backend. The FLUX output PNG was byte-identical to the corresponding native two-step output (peak 7.875 GiB reserved); Qwen was likewise byte-identical (15.986 GiB reserved). This verifies that both first-sample mask layouts reach the model and zero initialization preserves the generated pixels. Comet: [FLUX branch smoke](https://www.comet.com/nikolay-2104/rsrch-new/31c7186a06f643448cb47709f3b03fee), [Qwen branch smoke](https://www.comet.com/nikolay-2104/rsrch-new/219d489dd2a54f63a3a5e41a8ee78eb6).
- `ba_dit.checkpoint` saves only registered branch tensors with an exact backbone revision identity; a tiny adapter save/reload round trip and wrong-revision rejection passed. This does not provide optimizer or full training resume state.
- No **trained** pretrained-model branch generation, identity metric, reference-only/control experiment, or optimizer update has run. The masks have only been checked against the first live reference layout; all validation references and target grids still need full-panel verification. These are required before interpreting research results.

## 48 GB and 80 GB handoff

`scripts/run_native_validation_profile.sh 48|80 flux|qwen [limit]` is a GPU-gated native full-panel validation launcher; it logs to Comet after completion. `scripts/train_flux_native_lora_smoke.sh 48|80` and `scripts/train_qwen_native_lora_smoke.sh 48|80` are GPU-gated **upstream paired-LoRA plumbing smokes**. The FLUX scripts select Base 4B or 9B; the Qwen scripts use the same 7B checkpoint at both GPU classes. They have passed shell syntax checks but have not run on those GPU classes. The FLUX 9B weights have not been downloaded. The research configs (`flux4b_48`, `flux9b_80`, `qwen7b_48`, `qwen7b_80`, and matched arms) are design inputs, not runnable training commands.

Before a branch or matched LoRA experiment can run, implement and verify: encoder/VAE precompute with exact native transforms; disk-backed cache keys and live/cache equivalence; transformer-only train loaders with reference masks and the original native loss; branch/LoRA optimizer ownership; checkpoint save/resume; full-panel inference and metrics on the **patched** backend; Comet run metadata and validation at step 0/every 2,000 updates; and memory telemetry below the plan's 90% gate. Run the upstream LoRA smoke first on the chosen 48/80 GB host, then the staged research workflow in §11 of the plan.

## Local commands already exercised

```bash
cd /home/kolyangg/rsrch_new
scripts/clone_sources.sh
envs/flux-toolkit/bin/python scripts/validate_native_flux.py --arch 4b --steps 50 --output-dir runs/native_flux4b_local_50step
envs/qwen21/bin/python scripts/validate_native_qwen21.py --steps 40 --output-dir runs/native_qwen21_local_40step
```

The two native and two initialized-branch smoke runs were uploaded with `scripts/log_validation_comet.py` from `envs/qwen21`, which contains Comet ML. Source cloning, weight audit/download, mask building, and pair export scripts are kept separate so their outputs can be inspected before expensive runs.
