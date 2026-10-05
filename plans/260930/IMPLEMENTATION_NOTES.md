# Implementation notes — 30 September 2026

The [attached plan](CL39_Qwen_FLUX_48GB_80GB_Implementation_Plan.md) is preserved unchanged. The branch trainer, staged conditioning caches, inference, optimizer resume, native/LoRA/branch controls, Comet logging and deployment scripts are now implemented. The six research YAMLs are executable through `python -m ba_dit.cli` / `scripts/run_profile.sh`.

## Sources, environments and data

- Public repository: `https://github.com/kolyangg/rsrch_new`; local checkout `/home/kolyangg/rsrch_new`. Current implementation changes remain in the working tree, per `AGENTS.md`; a source-packaging script includes uncommitted code for deployment.
- Qwen Diffusers: `fef717ffb01f407d2637584ed936c16db908587a`. FLUX Toolkit: `ecee894ed2b1f3716d9d7326693061ec1a3105bb`. Small patches import the branch helpers; runtime checks both Git HEAD and the exact recorded diff. FLUX's ancillary Diffusers library is pinned at `c943837899b16cbae2f619b8dd4f7bb6f07dd81a`.
- Fresh deployment environments were actually created and checked in `envs/deployment_check/{flux-toolkit,qwen21}` using `scripts/setup_machine.sh`. Both passed dependency checks and architectural tests. Python 3.11.13 includes the headers needed by Torch compilation. Original locally bootstrapped environments used Python 3.11.11; `runtime.py` can find the managed headers for them.
- FLUX: Torch 2.13.0+cu130 / Transformers 5.5.3. Qwen: Torch 2.10.0+cu128 / Transformers 5.17.0. Separate CPU metrics environment uses Torch 2.2, InsightFace .7.3, ONNX Runtime 1.21 and the previous project's pinned OpenAI CLIP commit. Installed dependency versions are recorded in `locks/*-constraints.txt`.
- HF file/revision/byte locks: `weights-flux48.json`, `weights-flux80.json`, `weights-qwen.json`. FLUX 9B metadata was audited; its 18.157 GB transformer plus 16.397 GB encoder were not downloaded onto the 16 GB GPU machine. Its three selected components total 34.891 GB.
- Fixed validation: 8 sorted held-out references × 12 original prompts, seed 0. `manual_val_96.jsonl` SHA256 remains `d769406704ddbcdf0bb09d408ad4929b302aaa84a12e295872618e3df9f09a65`. Reference bytes are checked against the stored hashes. Twelve original reference photos and both original identity-embedding lookup files stay outside Git.
- Training smoke: 38 distinct-file pairs from the old three-identity sample. Exact file overlap is checked; different identity naming systems require explicit aliases/exclusions for semantic holdout separation. Both Cosmic Large and adjusted Large Dataset are selectable; see the dataset update below.

## Branch and control implementation

`ba_dit/nn/reference_read_delta.py` owns all added attention arithmetic. Eight sites per backbone register independent FP32 K/V low-rank adapters, rank/alpha 16, gamma .1, random A and zero B. Native queries retain their original RoPE. Both reads use exactly the same selected reference-face keys, capped at 512. The difference is added to target queries before the native output projection. Empty support returns zero.

The source patches only pass immutable per-sample contexts and call `ba_dit/backends/attention.py`. FLUX retains its target/reference joint stream; Qwen retains causal conditioning and zero-timestep prefix modulation, with KV reuse disabled. Reference and target sizes are independent.

Optional rank-8 native attention LoRA is registered at the same sites. FLUX Q/K/V slices and attention output columns are explicit, leaving fused MLP slices untouched. All trainable parameters are registered before optimizer construction; inventory and separate branch/LoRA optimizer groups are saved.

Training uses native flow-matching conventions: FLUX Toolkit sigmoid timestep draws and MSE velocity target; Qwen shipped unshifted training sigmas with uniform sampling/no weighting. Qwen's paired trainer normalizes VAE latents using FP32 statistics before casting to BF16, while validation uses the native pipeline's BF16 normalization. Those caches have distinct identities. Qwen training timestep division occurs before the native BF16 cast; FLUX's Toolkit path casts before division. A source comparison caught and corrected that FLUX rounding order. Target posterior samples are fixed per image/content hash for the cache; this is a recorded cache policy.

## Local measured results

Hardware: RTX 4090 Laptop, **16,376 MiB physical VRAM**, approximately 30 GiB host RAM. Validation stages are separate processes; transformer timing below excludes encoder/VAE stages. It includes both branch-on and branch-off denoising when parity is enabled.

| Initialized branch validation | Target / reference area | Steps | Peak transformer reserved | Denoise + native comparison | Result / Comet |
| --- | --- | ---: | ---: | ---: | --- |
| FLUX Base 4B | 768² / 512² | 50 | 8.248 GiB | 88.67 s | Exact latent/pixel parity; [675ef93f](https://www.comet.com/nikolay-2104/rsrch-new/675ef93ff6a4402b919ba74df7ed7de3) |
| Qwen 7B, 48 profile | 768² / 512² | 40 | 13.904 GiB | 79.52 s | Exact parity; [e191dfdd](https://www.comet.com/nikolay-2104/rsrch-new/e191dfdd844a4baf8f9900c661d3c328) |
| Qwen 7B, 80 profile geometry | 1024² / 768² | 40 | 14.457 GiB | 201.68 s | Exact parity; [47ded9cf](https://www.comet.com/nikolay-2104/rsrch-new/47ded9cf018146899a6eab7780b7eaad) |
| FLUX Base 9B | 1024² / 768² | — | — | — | Requires larger host; BF16 transformer file alone exceeds local physical VRAM |

These three full-step branch runs generated panel item `00`, with outputs under `runs/{flux4b_48,qwen7b_48,qwen7b_80}_untrained_*step`. They do not constitute 96-item branch comparisons or trained-quality improvements.

`check_cached_pipeline.py` separately compared the original native sampling pipeline and live reference VAE encoding against cached execution at 512 pixels/two steps. Both FLUX and Qwen matched reference latents and final latents **exactly**. This uses the same component placement for the comparison. Earlier all-in-one/offloaded baseline PNGs differ slightly from staged GPU-VAE decoding; those are not used as an exact cached parity reference. FLUX decoding now explicitly converts VAE pixels to FP32 before the native image processor.

### Real backward and resume

Two updates per pretrained backbone used two actual cross-view samples, target 256², reference area 512², accumulation 1, gradient checkpointing and no encoder/VAE in the training process.

| Backbone | Branch parameters | Updated B matrices | Deterministic peak reserved | Resume result |
| --- | ---: | ---: | ---: | --- |
| FLUX 4B | 1,572,864 | 16 / 16 | 8.223 GiB | All 32 adapter tensors exactly equal after continuous 2 updates versus 1 + process restart + 1 |
| Qwen 7B | 2,097,152 | 16 / 16 | 14.150 GiB | All 32 adapter tensors exactly equal; max absolute difference 0 |

Evidence: `runs/flux4b_resume_native_rounding/` and `runs/qwen7b_resume_deterministic/`, including `resume_parity.json` and per-update metrics. The final FLUX recheck includes the corrected native timestep rounding order; all 32 tensors again match exactly with maximum difference 0. Both backbones remained below 90% of local VRAM at this reduced geometry. This does not admit the full training profiles on a 48/80 GB host.

A first FLUX training attempt caught a FP32 noisy-input/BF16 projection mismatch and was corrected to the native input dtype. A first fast-kernel resume comparison exposed nondeterministic attention backward differences; deterministic algorithms plus an explicit cuBLAS workspace are now configured and the real resumed trajectories match exactly. CPU/CUDA/Python RNG and sample position are restored, and initialization draws are separated from the training noise stream across modes.

## Face masks used by validation

- Reference masks: 48 backend/size records and 96 PNGs, rebuilt using the shared runtime geometry. Boxes use exclusive XYXY endpoints. Metadata SHA256: `0e52b50c09b925cfe14f520ef8cc0fa187555a67e4b208dcb10678037a81220e`.
- Output masks: new native/initialized output detections, separately keyed by backbone, weight revision, target/reference geometry, steps, guidance and fixed panel hash. FLUX 4B's first face box is `[283,55,454,307]`; Qwen 768 uses `[290,77,492,397]`; Qwen 1024 uses `[385,73,655,480]`. They are different masks derived from their own outputs.
- `validation_masks.py` freezes the largest detected face, flags near-equal face-area ties and no-face cases, and records source image and PNG hashes. Selection does not maximize identity similarity. `metrics.py` reads those PNG masks and uses their actual support for subject ownership scoring. Historical PhotoMaker output boxes are not reused.
- Output-mask generation is part of ordinary native/untrained validation. Trained validation requires the matching baseline masks. Per the attached plan, output masks are evaluation labels; inference uses reference masks and an all-target branch gate.
- The FLUX 48-profile native baseline has completed all 96 images and prepared **96/96 usable output masks**, with no missing or ambiguous detections. Both 48-image inspection plots were reviewed: the selected boxes cover the foreground subject even in crowd scenes, and no manual corrections were needed. Panel item `00` has exactly the same PNG hash as the earlier initialized-branch mask source. Total transformer denoising was 137.12 minutes, with 8.482 GiB peak reserved memory on the thermally limited laptop. The run is [d3a00a02](https://www.comet.com/nikolay-2104/rsrch-new/d3a00a02d8e04c5cbf4997984d80ad3b).
- FLUX's full-panel masked identity mean is `0.371019`, legacy-best mean `0.310177`, and CLIP logits mean `28.285446`. Mean mask IoU is `1.0`; missing, unowned and ambiguous fractions are all zero. These are untrained/native baseline measurements.
- Qwen's 48-profile native baseline completed all 96 images and **96/96 usable masks**, with no missing or ambiguous detections. Both inspection grids were reviewed; no manual corrections were needed. All 96 pixel masks differ from the corresponding FLUX masks. Item `00` matches the earlier initialized-branch latent file and PNG hash exactly. Transformer denoising took 109.38 minutes and peaked at 14.117 GiB reserved. The run, mask grids and frozen manifest are in [3a6cb4d7](https://www.comet.com/nikolay-2104/rsrch-new/3a6cb4d7176c4dcfa14f7bf0c46fbcdc).
- Qwen's full-panel masked identity mean is `0.752387`, legacy-best mean `0.648705`, and CLIP logits mean `25.490243`. Mean mask IoU is `1.0`; missing, unowned and ambiguous fractions are all zero. Both backbones' scoring consumed their own new mask PNGs.
- Expanded-profile coverage remains explicit: Qwen 1024 has a verified first-item mask; FLUX 9B has no local pretrained output masks. Their full sets are generated by the same deployment command on the larger host.

Metrics preserve the old subject-v2 `id_sim`, legacy-best score, no-face/count/ownership diagnostics, and CLIP logits. The old legacy embedding file gives very different values from the corrected subject-v2 file; they retain distinct metric names. Masked first-item results are uploaded to the same Comet experiments, not replacement runs. Single-item identity scores are not a research comparison.

The seven original `face_quality/` curves are also implemented with PyIQA 0.1.15 in a separate pinned CPU environment. TOPIQ-Face, TOPIQ, MUSIQ and MANIQA-PIPAL use the same largest-face detection and 25%-per-side padded square 512-pixel crop as `clean_full`'s scorer. The crop implementation was compared directly with the original function, including image-edge cases. All four models successfully scored the initial FLUX image and both Qwen resolutions, uploading to each existing Comet run. Scoring is enabled by default; PyIQA's separate CLIP dependency cannot replace the original identity/CLIP scorer's pinned package. The setup script installs the headless OpenCV wheel last so scoring also works on minimal GPU images without desktop OpenGL libraries.

The full FLUX panel also completed all four face-quality models: 96/96 detections and TOPIQ-Face coverage, TOPIQ-Face mean `0.671817` (p10 `0.467516`), TOPIQ mean `0.487473`, MUSIQ mean `61.376513`, and MANIQA mean `0.593544`. Its CPU pass took 89 minutes with two threads while Qwen used the GPU; MANIQA dominated. The default scorer now uses up to eight CPU threads, supports `--threads`, and records per-model progress/timings and the actual thread count. Model settings and the crop policy are preserved. Both FLUX mask review grids and the frozen mask manifest were uploaded to its existing Comet run.

Qwen's full panel completed all four models with 96/96 detections and TOPIQ-Face coverage: TOPIQ-Face mean `0.714250` (p10 `0.569287`), TOPIQ mean `0.547539`, MUSIQ mean `70.717471`, and MANIQA mean `0.648878`. Eight CPU threads were used; the four models took 24.74 minutes in total, including 23.34 minutes for MANIQA. The complete JSON/CSV reports and seven metric curves were uploaded to the existing Qwen Comet experiment. Both full validation pipelines exited successfully.

## Focused verification and deployment

`check_invariants.py` passes seven checks per environment: zero-effect/first-gradient behavior, key support, adapter round trip/revision rejection, fused-LoRA MLP exclusion, optimizer/RNG resume, backend seams, and full-depth layouts with two simultaneous sample contexts under checkpoint recomputation. The FLUX layout test covers both 4B and 9B depth/site maps; Qwen also checks that trained branch corrections leave the native reference prefix unchanged.

`setup_machine.sh` and `setup_metrics.sh` were run to create clean environments. Source patch checks, config validation, Python compilation and shell syntax checks passed. Dataset bundle extraction was exercised. The dataset update below also checks both importers using real local images.

The private transfer archive `data/bundles/validation-and-smoke-with-metrics.tar.gz` is ready (17,600,535 bytes; SHA256 `6ac6d580a7713e9fdf7324620ef52b7c875b629d7881047154b0a547ecf42e78`). It contains the references, both original identity-embedding files, reference masks, the two complete 96-item output-mask sets plus the initial Qwen 1024 mask, and the 38-pair training smoke data. All 386 output-mask PNG hashes were checked inside the archive; it contains no symlinks or credentials.

`data/bundles/rsrch_new-code.tar.gz` snapshots the current working source, including uncommitted files. `data/bundles/SHA256SUMS` records the source, validation and training-metadata archive checksums. The archives exclude credentials, installed environments and model weights. No commit or push was made for this implementation.

Primary commands are documented in [README](../../README.md) and [deployment guide](../../docs/deployment.md). The old native validators/upstream LoRA smokes remain historical reference utilities; the new `run_profile.sh` uses the shared branch-capable backend. No 48/80 GB host was supplied and no remote job was submitted.

Still requiring that host/data: FLUX 9B pretrained images/backward; full-resolution 48/80 training memory admission; preparation of the selected full training dataset on that host; and research-duration trained adapter quality/control comparisons. Native full-panel masks and all smoke outputs are local ignored artifacts, with metadata and scripts suitable for transfer.

## Both original dataset options — 30 September update

- `locks/datasets.json` and `scripts/prepare_dataset.py` provide `cosmic` and `large` options. Each gets separate downloads/extractions, a training manifest, and an audit record; any 48/80 GB profile selects it with `--train-manifest`. Metadata SHA256 is checked before download/import.
- The old project history supplied the original Cosmic Drive IDs: `16upgN9HNXRKdASiWsYvo1pw1-ndpC3uO` for targets and `1YV1sGZIFE4qQ1Ju-5Yqo9NvcKt2q16zO` for references. Both share pages returned HTTP 200 and the expected names (`cosmic_dataset_images.tar`, `cosmic_large.tar.gz`). Cosmic needs both archives; the importer now supports separate target/reference roots.
- Large uses `large_dataset_adj.tar` and `filtered_ids3_adj.json`. The old repository, its download-script history and saved project setup conversations did not contain its actual Drive ID: the relevant conversation only supplied an example ID, then reported a manual download. The original link has been requested. `--url`, `LARGE_DATASET_URL`, and reuse of the old local extracted folder are already supported. BigCelebs is a separate release, not an alias for these image archives.
- The private `data/bundles/training-metadata.tar.gz` contains both original JSON files, verified against the old dataset audit: Cosmic `8ba369ef2fdc0496a0d3d55afb5c7923c1aa299343a676ac6bc0d94f3a3a0196`; Large `0056f9647c6ca69079c3b7ae479ea5cdf9e642f076460249b160000eecb3ee50`. Use `scripts/package_training_metadata.py` to rebuild it. This keeps the new host independent of the old checkout.
- Verification: the new Large preparation command imported four real pairs with the complete pinned metadata. The Cosmic importer imported four real pairs using an unchanged subset of the original metadata and separate existing target/reference roots. All eight pairs passed distinct-image/validation-overlap checks, target preprocessing and nonempty reference-token masks for `flux48`, `flux80`, `qwen48` and `qwen80`. These are data/preprocessing checks, not additional model inference runs. Evidence is under `scratch/dataset_options/`.
- The metadata transfer bundle was extracted by the deployment downloader in the Qwen environment, and both extracted JSON hashes match their pins. Python compilation passed for all four changed/new data scripts. Metadata archive SHA256: `f5603007ac5d0389d2b42c178fe90972b1d8a2e4ef91c70d598ba37c6a80d709`.
- Full image archives were not downloaded again onto the laptop (approximately 45 GB free). Dataset-choice and source changes do not alter the fixed validation panel or either backbone's generated-face masks.

## Initial 48 GB Vast run

Instance `53574065` is an already rented RTX 6000 Ada with 49,140 MiB GPU memory, 200 GB disk allocation, 93 GiB system RAM and NVIDIA driver 565.77. The first profile is `flux48` (FLUX.2-klein Base 4B). Driver 565.77 cannot run the original CUDA 13.0 wheel (NVIDIA's CUDA 13.0 GA minimum is 580.65.06). `setup_machine.sh` selected Torch 2.13.0+cu126; this is a deployment deviation from the CUDA 13.0 pin. The original adjusted Large Dataset images are 17 GB locally; `scripts/sync_large_dataset_vast.py` resumes their transfer along with the private validation and metadata bundles. To leave room for weights, separate `uv` environments, caches and checkpoints, the first run selects 4,096 pairs by stable hash across the full Large metadata. The Large preset maps seven IMDb IDs present in the metadata to the fixed panel's held-out names; the matching photo subjects were identified from the panel references. This is a named pilot; record its own results and never treat it as a full-population comparison. Use the normal 96-item validation at step 0 and every 2,000 updates.

- Remote setup completed with `uv`, separate FLUX, metrics and face-quality environments, pinned source checkouts and the three locked FLUX 4B weight aliases. CUDA availability and `uv pip check` passed. The exact remote preflight imported the pinned FLUX source and resolved the fixed 96-item validation manifest; GPU capacity was 47.398 GiB and met the profile.
- A two-update, 768 px `flux48` branch-only hardware smoke on the real RTX 6000 Ada passed with the unchanged accumulation of eight. All 16 B matrices changed after update 1, gradients and losses were finite, a checkpoint was saved at update 2, and peak CUDA reserved was 9.525 GiB (20.10% of device memory). A separate one-update checkpoint resume produced the same step-2 loss and byte-identical adapter tensor file as the uninterrupted run. This used two smoke pairs; it is hardware and training-path verification, not pretrained-model validation or a measurement of the full 4,096-pair cache footprint.
- The complete original Large metadata and local images yielded 4,096 stable-hash pilot pairs after excluding 110 pairs by held-out identity and six pairs whose distinct filenames contain identical bytes. Exact validation-image overlap was zero. The importer now refills after rejecting duplicate-content pairs. Pinned metadata SHA256 was `0056f9647c6ca69079c3b7ae479ea5cdf9e642f076460249b160000eecb3ee50`.
- The same 4,096-pair selection imported on Vast with 110 identity exclusions, six duplicate-content exclusions and no exact validation-image overlaps. Its exact profile preflight passed. A second two-update smoke using this manifest and the production 768 px geometry/accumulation passed: finite losses and gradients, all 16 B matrices updated, checkpoint saved, and peak CUDA reserved 9.523 GiB (20.09%). The original full 17 GB image transfer continues in the background; all images used by this pilot are already transferred.
- Production pilot command `scripts/run_profile.sh flux48 train --mode branch_only --quality-metrics --train-manifest data/train_pairs_large_4096.jsonl --run-name flux48_large4096_branch` started on Vast from commit `742bfba`, PID 2660. Comet project `rsrch_new`, immutable experiment key `fe87f8bbd2b743a49db187fb1b2fd447`. Remote log: `/workspace/flux48_large4096_train.log`; run directory: `/workspace/rsrch_new/runs/flux48_large4096_branch`. At the first check, encoder conditioning preparation was progressing (70 of 4,094 uncached pilot rows); no optimizer update had occurred yet. The run is configured for the fixed 96-item step-zero and 2,000-update validations. Update this entry with actual validation, optimizer and memory results after they occur.
- The resumable transfer completed with 47,500 image files and 17 GB on both source and Vast; remote disk use was 76 GB of 200 GB after all training conditioning caches were prepared. The production step-zero validation began and its first two 768 px samples completed denoising in about 32 seconds each, with 8.248 GiB peak CUDA reserved. Image decoding, optimizer updates and the 96-image metrics are still pending at this observation.
- A requested FLUX batch-eight validation check used 15.006 GiB peak CUDA reserved. Its untrained branch output matched the native output exactly within the batch for all eight samples. With native comparison enabled, elapsed time was 63.612 seconds per sample, or about 31.806 seconds per sample for one denoising pass, versus roughly 32 seconds for serial production validation. Batched latents differed from serial latents (sample 00 max absolute difference 1.902). The serial job was interrupted after 11 completed latents to test batching; batch eight is retained as an explicitly named experiment option, while the production profile remains serial. The 11 completed latents and original Comet key are retained for validation resume.
- `--resume-run` restarted the same production run from its saved config and 11 completed latents on commit `fecd523`, PID 3673. A one-sample serial check under the new code reproduced the original latent byte for byte. The resumed validation progressed to sample 12 (13/96) and retained Comet key `fe87f8bbd2b743a49db187fb1b2fd447`. Resume log: `/workspace/flux48_large4096_resume.log`. Optimizer updates remain pending until the full fixed panel and scoring finish.

## Named 12-item wiring pilot

At user request, the 96-item run was stopped before its first update to start training sooner. `data/validation/manual_val_12_pilot.jsonl` is an exact ordered subset of the fixed panel at original row indices `[0, 6, 12, 24, 30, 36, 48, 54, 60, 72, 78, 84]`, covering all eight identities. Source prompts, seeds, references and metric definitions are unchanged. Panel SHA256 is `db92d25e630a17b7da4ae053538bfbc92ae176497e52b1d4b41594f52b8bc803`. `configs/flux4b_48_pilot12.yaml` keeps the same model, geometry, branch, optimizer, 2,000 steps and 2,000-update validation interval while selecting this 12-item manifest and a separate profile name. Its baseline output masks are stored under a panel-specific directory so the original 96-item backbone masks remain frozen. The pilot's metrics and image changes must be compared only between its step-zero and step-2,000 validations, using `scripts/compare_validation_steps.py`.

The named pilot started on Vast instance `53574065` from commit `3c343ec`, process ID `4092`, log `/workspace/flux48_large4096_pilot12.log`, run directory `/workspace/rsrch_new/runs/flux48_large4096_branch_pilot12`. Its Comet experiment key is `77a3271d63794483a988f654b5e7e6ef` in project `rsrch_new`. The 12 step-zero images and both legacy and face-quality metric summaries were produced. At the final requested check, six optimizer updates had completed and the process remained running. Update one had finite loss `0.66147`, B-gradient norm `0.00025674`, and all 16 B matrices changed. Update two had finite loss `0.75944`, B-gradient norm `0.00027258`, peak reserved CUDA memory `9.52344` GiB (20.09% of the 47.398 GiB device). Monitoring stopped at the user's request; step-2,000 validation and image/metric changes are not yet measured.

At 2026-09-30 23:58:41 UTC the pilot had completed 1,144/2,000 optimizer updates (9,152 training images). The last 100 updates averaged 10.5909 seconds each, projecting 9,066 seconds (2 h 31 min 6 s) to update 2,000, approximately 2026-10-01 02:29:47 UTC / 03:29:47 BST; final 12-item validation and scoring add time after that. The effective batch is eight images per update (microbatch one, gradient accumulation eight). Three one-second GPU samples reported 99–100% compute utilization, 31–55% memory-controller utilization, and 10,346 MiB of 49,140 MiB VRAM occupied. These are observations and a projection, not a measured completion time.

The progress logger now records percentage, remaining steps, ETA seconds, and updates/hour alongside every original scalar in `metrics.jsonl` and Comet, while printing one compact progress-bar line every 25 updates. The already-running Python worker cannot reload that code, so `scripts/live_training_progress.py` adds the same progress scalars and compact output to its existing Comet experiment without restarting or changing the optimizer. Its original verbose stdout log is retained separately.

The running pilot pulled commit `ce55a36` without interrupting PID 4092. The live logger started at step 1,150 and connected to the same Comet key. Its compact output is `/workspace/flux48_large4096_pilot12.log`; the optimizer's original per-step JSON output continues in `/workspace/flux48_large4096_pilot12_raw.log`, and sidecar errors are in `/workspace/flux48_large4096_progress.err`. The first compact line projected 2 h 30 min to step 2,000. No later training or validation outcome has been checked yet.

## Queued one-ID diagnostic

At the user's request, a fresh FLUX one-ID experiment is queued behind completion of this pilot's 2,000 steps and all validation/scoring. It uses the original 19-image `nm0005092` data, `51.jpg` reference, 12 prompts and identity embedding, with an explicitly named same-identity diagnostic split and new panel-specific FLUX output masks. Validation is every 500 updates through 2,000. See [ONE_ID_DIAGNOSTIC.md](ONE_ID_DIAGNOSTIC.md) for source hashes, deliberate pairing differences, remote queue state, and the scheduled conditional wiring investigation. At deployment the old run remained active at step 1,215, queue PID 5913 was waiting, and no one-ID training had started. No branch wiring fault has yet been established from pretrained validation evidence.

At 2026-10-01 02:39 UTC the pilot had completed all 2,000 updates and final validation/scoring. All 12 images changed visibly; paired settings matched, and the final report used the saved branch checkpoint. Identity similarity improved 0.484306→0.535192 and TOPIQ-Face 0.822842→0.841349; CLIP text similarity declined 24.394645→23.951896. All 16 branch B matrices were nonzero (combined L2 norm 5.520810); last-window losses/gradients were finite, and peak reserved memory was 9.5293 GiB. The queue started one-ID parent PID 6851 under Comet key `25604bb4dc58429d9378efea58b43ed2`; step-zero generation was in progress. Detailed comparison and artifact locations are in the diagnostic notes. No wiring fix is justified by these pilot results; the requested one-ID test continues.

At 04:39 UTC the one-ID step-500 validation/scoring was complete and training had resumed to step 572. All twelve outputs changed visibly relative to baseline under equal validation configs and the recorded step-500 checkpoint. All 16 B matrices were nonzero (combined L2 norm 5.788778), and recent gradients remained finite. Identity similarity declined 0.310866→0.301947 and CLIP text similarity 28.510215→28.073405; this shows branch influence without an early identity-quality gain. No wiring change was made; later checkpoint reviews remain scheduled. Full results and paired-image artifact locations are in `ONE_ID_DIAGNOSTIC.md`.

At 06:09 UTC one-ID checkpoint 1,000 and validation/scoring were complete, and training had resumed to step 1,028. All twelve images again showed visible changes relative to baseline with the correct checkpoint and equal settings. Branch B-matrix L2 norm increased to 8.454246, with L2 parameter change 4.409597 since step 500. Identity similarity was 0.302849 (baseline 0.310866), CLIP 28.299676 (28.510215), and TOPIQ-Face 0.667383 (0.687094). Gradients remained finite. No branch activation/loading fault was indicated; the 2,000-step diagnostic continues.

At 08:09 UTC one-ID checkpoint 1,500 and scoring were complete, with training at step 1,654. All twelve images changed visibly under matched settings and the correct checkpoint. Identity similarity improved over baseline to 0.338955 (0.310866), CLIP to 28.673897 (28.510215), and TOPIQ-Face to 0.713561 (0.687094); MUSIQ, MANIQA and TOPIQ means also improved. All sixteen B matrices remained nonzero (combined L2 norm 10.430811; change since step 1,000 is 3.877858), and gradients were finite. No wiring repair is indicated; the final 2,000-step validation remains pending.

## Validation mask visualization — 2026-10-01

`scripts/visualize_validation_masks.py` overlays actual cached reference masks on the backbone-preprocessed references, showing the source face box, selected reference keys and enlarged 16×16 pixel token cells. Local report: `runs/mask_review_20261001/index.html`. The one-ID reference and all eight references in each fixed 96-item FLUX/Qwen panel were inspected: all 17 cached masks match runtime geometry and token counts exactly and align with the intended faces. All nine FLUX reference images also match native preprocessing pixel-for-pixel. This verifies mask placement, not additional pretrained inference.

The one-ID face selects 88 of 1,024 reference tokens. Qwen's Eddie reference has 567 face cells; the configured 512-key cap selects spatially spread cells inside it, explaining gaps in the overlay. Cell boundaries can extend beyond the original face rectangle. The report also shows all twelve frozen one-ID generated-face scoring masks on step-zero outputs; those visually align with the faces.

Mask roles are explicit: both attention backends read masked reference keys/values but apply the branch correction to **all generated-image tokens**. Neither passes the optional `target_gate` to `reference_branch`. Generated-image masks are evaluation-only and are never supplied to inference. No training process or mask was changed for this visualization.

Reproduce the current diagnostic report with:

```bash
envs/flux-toolkit/bin/python -m scripts.visualize_validation_masks \
  --config configs/flux4b_48_one_id.yaml \
  --output runs/mask_review_20261001/one_id_flux \
  --validation-dir runs/flux48_one_id_diagnostic/validation-000000
```

For the full reference panels, use `configs/flux4b_48.yaml` or `configs/qwen7b_48.yaml`, choose a separate output directory and omit `--validation-dir`. Existing conditioning caches and private validation assets are required; visualization itself performs no GPU inference.

## Current FLUX architecture report — 2026-10-01

`reports/261001_flux4b_branched_attention/flux4b_branched_attention_architecture.pdf` is a 16-page illustrated audit of the active one-ID FLUX Base 4B run, following the earlier E13/CL39 report style. It includes whole-model and BA diagrams, exact insertion sites, token/mask geometry, equations, source excerpts with line numbers, the original CL39 comparison, and measurements through the 08:41 UTC snapshot. The same directory contains an HTML viewer, per-page SVG/PNG exports, standalone `whole_model.pdf` and `branched_attention.pdf`, the generator and a source-hash audit.

Before generation, the local adapter identity, training-code digest and config digest were verified against the copied step-1,500 checkpoint; all match. Its inventory contains 32 trainable tensors / 1,572,864 parameters. The comparison uses the actual `rsrch_clean_new` `hardcase_attn_processor.py` cited by the old project's report, plus the user-supplied PDF. It explicitly distinguishes the current pre-projection `R1-R0` reference-read delta from CL39's projected `R-N` spatial/frequency/confidence route. The current run has no target-face gate, entropy confidence, frequency split or surface-ownership objective. All 16 report pages and both standalone diagrams were rendered and checked; no model or training code was changed.

At 09:12 UTC the one-ID run had completed all 2,000 updates and saved its final checkpoint. All tensors and recent gradients/losses are finite; all sixteen B matrices are nonzero (L2 norm 11.996713; change since 1,500 is 3.454949). Peak reserved training memory was 9.52930 GiB. Serial final validation started from checkpoint 2,000 and had recorded its first sample. Decoding/scoring and the final comparison remain pending; the follow-up stays active. The architecture report above retains its explicitly dated earlier evidence snapshot.

## Completed one-ID diagnostic — 2026-10-01 09:41 UTC

All 2,000 updates and final validation/scoring completed; the queue records `one_id_complete` and all training/validation/queue processes have exited. All twelve final images change visibly versus step zero under identical settings, prompts, seeds and geometry. The final validation uses the verified step-2,000 branch checkpoint. All 2,000 logged losses/gradient norms and all final adapter tensors are finite; all sixteen B matrices updated. Peak reserved memory was 9.52930 GiB.

Final identity similarity is 0.331836 (baseline 0.310866), CLIP 28.009225 (28.510215), and TOPIQ-Face 0.672174 (0.687094). Step 1,500 is stronger on all three: 0.338955, 28.673897 and 0.713561. Branch influence is established, but quality improvement is not monotonic. No wiring fix or retry is indicated. Full metrics, checkpoint checks and artifact paths are in `ONE_ID_DIAGNOSTIC.md`; the local visual review is `runs/review_one_id/final_review.html`. Complete checkpoints 1,500 and 2,000 are retained locally.

The follow-up automation was paused through the app and its saved `PAUSED` state verified. Vast instance `53574065` remains running with an idle GPU; it was not stopped or terminated. No model or training code was changed during this final review.

## Training source verification and artifact delivery — 2026-10-01

Before the requested Git push, local `adapter_identity`, `training_code_digest`
and `config_digest` were recomputed and exactly matched the copied final
step-2,000 checkpoint manifest. The architecture audit's model/training source
hashes also matched local files and committed HEAD `56fffea`. Fetching GitHub
confirmed that HEAD was already on `origin/main`. A live Vast source comparison
could not run because SSH refused the connection; checkpoint provenance supplies
the verification here. No machine state was changed.

The newer mask-visualization script, report generator/dependency pins/source
audit, completed-run notes and Dropbox helper are included in the subsequent
source commit. Generated report exports, run artifacts, weights and credentials
remain untracked. The 16-page PDF was uploaded with verified Dropbox content
hash to `Apps/temp/rsrch_new/2026-10-01/flux4b_branched_attention_architecture.pdf`.
The Dropbox helper was copied from EMDR and configured for API root
`/rsrch_new` inside `Apps/temp`; credentials are in ignored `.env` with mode 0600.

## Fast face-focused BA diagnostic — 2026-10-01

Implemented a separate FLUX Base 4B path with one reference-face attention read
at the final single-stream block, rank-16 output LoRA and a learned scalar face
gate. Only the 101,377 BA/router parameters train. Target boxes supervise gate
BCE and face-balanced flow loss during training; inference uses model queries
to predict the gate and receives no target/generated-face masks. The existing
multi-site FLUX/Qwen paths and checkpoints are unchanged.

The new `scripts/run_face_diagnostic.sh` stages exact frozen-prefix caching,
training, save/process-resume, serial validation, ID/CLIP scoring and a visual
report. An initial cache attempt caught a BF16 rounding difference (0.03125)
when target rows alone were replayed. Retaining the complete native token
layout fixed it: all 24 cached cases and the trained full-model comparison are
exact. Tests also cover native initialization, reference-value dependence,
target-only scatter, finite/nonzero gradients and rejection of external
inference target gates.

Completed on the local 16 GB GPU: four one-ID images, three fixed noise levels,
separate probe noise seeds, batch four and 300 updates; validation is a separate
512 px/two-prompt/20-step diagnostic. Optimization took 14.698 seconds at
3.803 GiB peak reserved; prefix capture took 13.425 seconds at 8.158 GiB.
The complete run through review/scoring took 272 seconds. These are fixed-input
cache timings, not general fresh-noise training throughput.

Face-MSE fell 10.96% on fitted noise and 3.02% on separate probe noise. Probe gate
IoU reached 0.936, with per-token correction RMS 64.1× larger inside faces than
outside. Both generated images changed visibly. A conditional-only CFG trial
reduced identity similarity; using BA in both CFG lanes with identical trained
weights improved identity 0.383189→0.438182 and CLIP 27.241908→27.454217.
Some gate activation and final image changes extend to hands/clothing. These
small-panel results demonstrate learning and branch influence, not strict
pixel isolation or generalization.

Details, failed-trial evidence and artifact locations are in
[FACE_BA_DIAGNOSTIC.md](FACE_BA_DIAGNOSTIC.md). Final review:
`runs/flux4b_face_one_id_fast_20261001_shared_cfg/review.html`;
[Comet experiment](https://www.comet.com/nikolay-2104/rsrch-new/a49b042877154369a30080c2da5393f6).
Vast instance 53574065 was observed stopped and was not restarted.

## Stronger local face BA — 2026-10-01

Added a separate differentiable eight-block suffix (FLUX single blocks 12–19)
with rank-256 face BA: 12,607,496 trainable parameters, exclusively output LoRA
and query-predicted face routers. Native weights remain frozen, including the
blocks through which branch gradients propagate. Exact frozen-prefix caching
uses 228 fitted and 114 separate-noise cases from all 19 one-ID pairs at 768 px.
The first/last cached full-model predictions and native initialization match
exactly; all 19 transformed training masks were visually checked.

Six local throughput/memory trials selected batch 1 without activation
checkpointing: 0.600 s/update and 7.033 GiB reserved in the short benchmark.
Batch 2 fits at 11.641 GiB but is slower per sample; batch 4 without checkpointing
exceeded the memory gate (21.027 GiB reserved) and was rejected. Checkpointed
batches 1/2/4 fit but are slower. The actual training rate and final results are
recorded separately from these short benchmarks.

Run `runs/flux4b_face_one_id_strong_20261001` has started its requested 2,000
updates. All eight BA sites updated during the first 25 steps, native gradients
remained absent, and checkpoint reload plus fresh-process update-26 resume
matched exactly. Initial training peak was 7.051 GiB. Serial four-prompt
validation runs at 0/1,000/2,000 using the same patched model in both CFG lanes.
Baseline ID similarity is 0.306791 and CLIP is 28.454863; final gains are not yet
established at this note's creation.

Final launch check at 11:31 UTC: 374/2,000 updates, all recorded losses/gradient
norms finite, exact resume at update 26, peak training reserved 7.145 GiB.
Sustained speed was 1.156 s/update; NVIDIA reported active software thermal
slowdown at 88°C and roughly 98% GPU utilization. About 31 minutes of optimizer
time remained, plus serial validations. The launcher continues through 2,000
and final scoring/report generation; active monitoring stopped per the user's
earlier request. This is a running experiment, not a completed quality result.

Details and benchmark table: [FACE_BA_STRONG.md](FACE_BA_STRONG.md).
[Comet](https://www.comet.com/nikolay-2104/rsrch-new/7beb18408145451896ccdeaa9a6722bb).
Sources/configs are snapshotted in the ignored run folder. Vast was not used;
new experimental code remains uncommitted pending an explicit push request.

## Strong face BA: 24-image one-ID validation — 2026-10-01

The completed local 2,000-update FLUX 4B run was revalidated on a named
24-image panel at initialization, step 1,000 and step 2,000, using the existing
branch checkpoints and patched backend. The original one-ID panel has twelve
prompts, so this panel uses those twelve at seeds 0 and 1. The source four-image
outputs were reused byte for byte; all other images were regenerated with the
same reference and inference settings. All 72 PNGs and saved gate records are
complete. Peak CUDA reserved memory was 9.611 GiB; sustained inference slowed
under local GPU thermal throttling. No Vast machine was used.

Mean ID similarity was 0.331399 → 0.339098 → 0.339220, while CLIP was
28.0729 → 27.6710 → 26.8572. TOPIQ-Face declined 0.689587 → 0.673590 →
0.636220. At step 2,000 only 10 of 24 ID scores improved; a 12-prompt clustered
bootstrap interval for mean ID change spans zero. Mean saved gate activation at
step 2,000 is 0.9168 within the frozen baseline face boxes and 0.0848 outside,
and all 24 images changed. This confirms a face-focused active branch, but the
larger panel does not establish an identity or quality gain.

All 24 scoring masks passed automatic coverage and visual overlay review;
the cached BA reference mask selects the intended face. Masks from generated
images are for scoring only and were never input to inference. Full tables,
bootstrap intervals, overlays, source/checkpoint provenance and the dedicated
[Comet experiment](https://www.comet.com/nikolay-2104/rsrch-new/65297da706af4cd6a0308b78fc18f9ca)
are recorded in [FACE_BA_24_VALIDATION.md](FACE_BA_24_VALIDATION.md) and the
ignored run `runs/flux4b_face_one_id_24_val_20261001`.

## BA-only face reconstruction from noise — 2026-10-01

The user clarified that faces should start near random and depend on BA. Added
a separate 256 px face-crop flow head whose zero output projections initially
leave Gaussian latents untouched. It uses one four-head attention layer,
learned query/noisy-latent paths, 2,408,448 BA parameters, and a frozen FLUX
feature extractor. The native FLUX velocity is disabled. No target mask or
generated face box enters inference. Only native flow MSE is optimized.

Pure-reference and query-only residual trials reduced losses but left speckled
undetected faces. A timestep-scaled noisy projection was unstable at the initial
learning rate; all failures are retained. The final direct-velocity variant
completed 2,000 updates and produced detected faces on all four generation seeds
at both 1,000 and 2,000, versus none at initialization. Mean ID similarity is
0→0.30705→0.30599; image quality remains rough. Separate-noise flow MSE fell
58.7%. Disabling the reference read raises probe error 31.8% and lowers seed-zero
ID similarity 0.39975→0.27812, although the query path still produces a coarse face.

All trainable tensors updated with finite gradients; fresh-process resume and
cached/full predictions were exact. Peak CUDA reservation was 7.527 GiB with
the backbone and 0.461 GiB for cached BA training. The 2,000 optimizer steps took
11.10 seconds, and the complete primary run about three minutes. This is a
one-ID reconstruction proof using fixed cached inputs, not full-scene editing
or evidence of identity generalization. See [FACE_CROP_BA_ONLY.md](FACE_CROP_BA_ONLY.md),
`runs/flux4b_ba_only_face_crop_direct_20261001/report.md`, and
[Comet](https://www.comet.com/nikolay-2104/rsrch-new/29b7730044c94c2c97b736d27ae442e0).

## Corrected prompted validation with spatial routing — 2026-10-01

The user clarified that the task is new prompted generation, not face-crop
reconstruction. Added `scripts/masked_face_flow.py` and
`ba_dit/nn/masked_face_flow.py` for a named CL14-inspired spatial split: reuse
native generated scenes and their verified face masks; BA alone predicts face
flow from noise, with native background context and explicit exterior pixel
preservation after decoding. Target photographs are never validation inputs.
The user's instruction explicitly authorizes native-output masks for inference
in this experiment; the general reference-only protocol is unchanged.

All 24 mask/image hashes matched and overlays were reviewed. Pretrained BA-off
prediction is bit-exact after branch installation. The complete step-zero panel
contains noisy faces and zero pixel error outside the blend support. A fresh
full-scene one-ID training run and 0/1k/2k prompted validation are in progress:
`runs/flux4b_masked_face_flow_24_20261001`,
[Comet](https://www.comet.com/nikolay-2104/rsrch-new/1ef76d1341fa4c52989af372a5b65ce2).
See [MASKED_FACE_FLOW.md](MASKED_FACE_FLOW.md) for the precise difference from
CL14, conditioning, trainable parameters, cache protocol and limitations.

The first full-scene pilot was stopped at 1k after probe loss regressed with
lr .002/sequential cases. Its evidence is retained. The completed fresh run
`flux4b_masked_face_flow_24_stable_20261001` uses lr .001 and seeded shuffled
batches, with the verified same feature cache. All 72 prompted images at
0/1k/2k are generated and scored. Mean ID similarity is
**0.02024 → 0.15084 → 0.19661**; untouched native scores **0.33140**. This
demonstrates learning for prompted generation, with rough faces that remain
below native quality. Probe flow MSE falls 51.3%; disabling reference attention
raises it 22.9%. Native-off parity, exact resume and cached/live checks pass,
all BA tensors change with finite gradients, and all 72 final composites
preserve the exterior exactly. Peak inference CUDA reservation is 9.422 GiB.
The final [Comet run](https://www.comet.com/nikolay-2104/rsrch-new/d59463ae28784748a9092746eafeee24)
and [protocol/results](MASKED_FACE_FLOW.md) include the masks, raw decoder
outputs, face close-ups, source snapshots and limitations.

## Requested continuation to 10k — 2026-10-01

Started `flux4b_masked_face_flow_24_10k_20261001` as a new Comet experiment,
continuing both BA weights and Adam state from update 2000. Training/data/
architecture and the prompted 24-image panel are unchanged. New image metrics
are collected at 4k/6k/8k/10k. Extended batch-order prefix and process-resumed
update 2001 match exactly. See [MASKED_FACE_FLOW_10K.md](MASKED_FACE_FLOW_10K.md)
for lineage, scope and measured results.

## User extends target to 100k, validates every 20k — 2026-10-01

The 10k controller was superseded at 6k. Its completed 6k panel scores ID
0.21402 / CLIP 27.1623 (4k: 0.21165 / 26.7754). A new detached continuation
`flux4b_masked_face_flow_24_100k_r1_20261001` retains the 6k weights and Adam
state, and targets 100k total updates with 24-image validation at
20k/40k/60k/80k/100k. Same BA architecture, one-ID feature cache, lr .001,
batch8 and inference protocol. Cached-loss probes run every1k; all updates
are recorded locally, with live metrics sampled every100.

An initial 100k setup failed strict config validation before training because
cadence was placed in core config fields. Cadence now belongs to experiment
identity metadata; config validation runs before Comet creation. See
[MASKED_FACE_FLOW_100K.md](MASKED_FACE_FLOW_100K.md) and the
[new Comet run](https://www.comet.com/nikolay-2104/rsrch-new/fa43a8dd82274d1397cbf9befbbda9bf).
The cached-feature optimization speed must not be presented as end-to-end
FLUX throughput or fresh-noise training.

## Investigating ID regression and BA capacity — 2026-10-01

The completed 60k panel declined to ID 0.15408 (6k: 0.21402). Stopped the
superseded continuation and retained its checkpoints. Fixed-input diagnostics
show attention saturation and growing cancellation between the reference and
query/noise output paths, alongside fixed-cache overfitting. Added a separate
512-wide, time-conditioned head with bounded cosine attention, plus a plain
width512 capacity control and a fresh noise/sigma cache builder. See
[FACE_FLOW_CAPACITY.md](FACE_FLOW_CAPACITY.md) for evidence, controls and results.
A strict restart check exposed a 1.86e-9 FP32 difference, not a lost optimizer
state; the new check records an explicit numerical tolerance. No improved
identity claim is made before actual prompted-image scoring.

The completed wider/stabilized head on the original cache did **not** improve
ID: .18814 at 2k → .17927 at 6k. It is retained as a rejected control. A fresh
separate run combines the same conditioned width 512 head (5,327,360 BA-only
parameters) with 684 fit cases, continuous fresh sigmas/noise, lr .0003 and
weight decay .01. On the unchanged 24 prompted images, ID rises
**.02024 → .16421 → .22660 at 0/2k/10k**; 21/24 images improve from 2k to 10k.
This exceeds the prior best .21402, while native remains .33140 and CLIP text
similarity declines 26.8878 → 26.1877. Faces still show artifacts. Fit/probe MSE
at 10k is .51707/.80847, with finite gradients and every BA tensor updated.
Native-off and cached/live parity pass; restart differs by at most 7.45e-9.
All 72 composites preserve exterior pixels exactly and checkpoint hashes match.
Peak inference reservation is 9.396 GiB; cached training peaks at 1.334 GiB.
The run is complete at 10k, with no further GPU job queued. See
[FACE_FLOW_CAPACITY.md](FACE_FLOW_CAPACITY.md) and
[Comet](https://www.comet.com/nikolay-2104/rsrch-new/9faec049f9a544948c4faeb2e2fc3a7b).

## Stronger reference refiner and measured batch scaling — 2026-10-01

The user requested stronger BA, training until ID convergence, and higher GPU
utilization. Added a separate1024-wide/16-head reference refiner with bounded
cosine attention, pooled-reference modulation and a zero-initialized correction.
Its14,048,385 parameters train; the best previous512-wide BA core and all FLUX
weights stay frozen. All24 initial images exactly reproduce the previous best
ID .226596 checkpoint. The original batch8 admission was superseded after50
updates and preserved; no quality rejection is implied.

Cached-core outputs, stacked CUDA batch gathering and consolidated finite
gradient checks improve throughput. Benchmarked batches8/32/64/128/256/512;
128 reaches3536 cases/s and97–98% post-startup GPU utilization. Larger batches
add under1% throughput. New run `flux4b_reference_refiner1024_b128_20261001`
has passed500 real updates: finite gradients, all17 tensors changed, frozen
core exact, process resume exact, actual peak training reservation5.193 GiB.
Probe MSE improves .80847→.71048; prompted-image scores are the acceptance gate.

The serial controller validates all24 images at0/500/2k/5k and every5k thereafter,
stopping after two gains below .003 while retaining the highest-ID checkpoint.
See [REFERENCE_REFINER.md](REFERENCE_REFINER.md) for architecture, stopping rule,
batch-size evidence, protocol limitations and reproduction. Active
[Comet](https://www.comet.com/nikolay-2104/rsrch-new/debb81df08cf4f5591730dffa86f704c).

The first completed refiner panel at500 scores ID **.22971** versus .22660 at
initialization (+.00312;13/24 images improve). CLIP is nearly unchanged,
26.1877→26.1648. This is a small gain; the controller continues toward the
later checks. A5-second sample during actual batch128 training measured
96% mean GPU utilization and6393 MiB total device memory. All48 decoded
images at0/500 preserve the exterior exactly; checkpoint hashes and actual
cached/live prediction audits pass. A16-page architecture/results presentation
is generated in `reports/261001_reference_refiner/`.

### October 1 — refiner regression and direct identity rerun

The completed 1024 refiner failed to sustain its early ID gain: fixed24 ID_sim
.22660 → .22971 (500, best) → .20559 (2k) → .19275 (5k). It stopped after
the 5k panel/scoring, preserving the best checkpoint. Flow loss improvement
did not predict identity improvement; paired crops show degradation.

The new `flux4b_deep_identity1024_20261001` initializes from the best500 weights,
adds two independent reference-attention/MLP reads (30.83M trainable BA params),
and trains flow MSE plus direct differentiable ArcFace identity loss through
the frozen FLUX2 VAE. Nineteen training photos and their landmarks supply the
auxiliary data; validation remains generated images on the same fixed24 panel.
The native backbone, 512-wide core, VAE and recognition model are all frozen.

Measured admission: exact parent predictions, ArcFace/ONNX relative RMS1.19e-6,
finite identity gradients, batch256 .350s/update with the largest identity
crop every second update, peak reserved11.06GiB/69.2%. The cuDNN TF32 setting
was disabled after its less precise path failed the tight ArcFace parity gate.
Identity weight.5 every2 updates; LR5e-5; optimizer reset. No claim of improved
generated ID scores until the new panels finish. Save/resume and full patched
inference audits run in the serial controller before longer continuation.

[Protocol, admission and launch](IDENTITY_FLOW.md).
[New Comet run](https://www.comet.com/nikolay-2104/rsrch-new/c76d2a9a84814005bcd86287d95c2771).

The fresh24-image step-zero panel exactly reproduces the best parent images
and ID_sim .2297139211. Native BA-off and full-model cache/live checks pass
exactly. Fifty initial updates changed all35 trainable tensors while preserving
all11 frozen core tensors; live peak reserved11.693GiB and mean.276s/update.
Optimizer resume at51/52, including an identity update, passed its numerical
tolerance (max parameter error3.27e-7). Training is continuing past100 updates;
new trained image scores are still pending. The existing follow-up now tracks
this run using `scripts.review_identity_flow`.

### October 1, 20:30–20:41 UTC — first ID gain and reproducibility repair

The first trained deep-BA panel improves ID_sim **.229714 to .251904** at500
(20/24 images improve). CLIP26.1648 to26.2695; all faces owned/unambiguous and
all48 baseline/trained exteriors exact. Paired crops were inspected: artifacts
remain, and native ID.331399 still exceeds the BA score. This is an early gain,
not convergence or proof of independent perceptual quality.

The old controller failed on the resumed502 identity update because a4.37e-7
parameter discrepancy exceeded its near-zero tolerance;501 flow-only resumed
exactly. CUDA grid_sample backward is nondeterministic. The trainer did not
enforce the config's deterministic flag. The recovery uses equivalent bilinear
gathers and enforced deterministic algorithms/cuDNN/cuBLAS settings. Forward
and CPU gradient parity passed; two separate full training replays at501/502
now match bit-for-bit. Deterministic on/off prediction equality also passes on
an actual cached input. No old sources/manifests or failure evidence were changed.

Active continuation: `runs/flux4b_deep_identity1024_det_20261001`,
[Comet](https://www.comet.com/nikolay-2104/rsrch-new/75e5d8961bdf40efab63ffd2770711ca).
It imports unchanged0/500 panels with provenance and preserves the500 weights
**and Adam state**. Only the uncheckpointed501 log entry is excluded from the
new history; its original remains preserved. New controller
`scripts.run_deterministic_identity_flow` resumes toward2k and keeps the same
image-ID convergence rule. Historical initial optimizer-reset metadata refers
to the original experiment; `continuation.optimizer_reset` is false.

### October 1, 21:04 UTC — 2k evaluation

Deterministic continuation2k scores ID_sim **.241371**, down from best500
**.251904** (step0:.229714). CLIP26.0794 versus26.2695 at500. Only9/24 images
improve over500; paired face crops retain artifacts. All72 reviewed exteriors
are exact; face ownership succeeds for all24. Cached flow and identity losses
improve while generated ID dips, so those losses are not acceptance evidence.

The2k checkpoint/source hashes, finite gradients, all35 parameter updates,
all11 frozen core tensors, native BA-off and cache/live checks pass. Fresh
resume at2001/2002 is bit-exact, including identity loss. Peak training reserved
12.424GiB, inference9.400GiB. Best500 is preserved. The controller continues
to5k because this is the first of two allowed insufficient-gain checks; the
follow-up remains active. See IDENTITY_FLOW.md and the continuation Comet run.

### October 1, 21:44 UTC — deep identity run stopped after final scoring

Completed all24 images at5k: ownership-matched ID_sim **.225265**, CLIP26.2110.
Best remains **.251904 at500**, compared with .229714 initially and native
.331399. The two-check rule stopped the controller after2k/5k regression;
no training or validation process remains. This is a plateau of the best
development score, not statistical convergence. Fifteen of24 images score worse
than best500. Paired-face sheets were inspected; eye/mouth/texture artifacts
persist and the final checkpoint does not establish better visual quality.

All5,000 loss/gradient records finite, all35 BA tensors updated, all11 frozen
core tensors exact. Source hashes, checkpoint/sample provenance, native BA-off,
cache/live prediction and exact resume including ID updates pass. All96 reviewed
composites preserve exterior pixels exactly. Peak training reserved12.424GiB
(77.7%); inference9.400GiB. Best500 weights and Adam state are hash-verified and
retained. Original failed/regressed runs remain preserved.

Fit/probe flow MSE ends at.206468/.675383; auxiliary ID loss falls to.042233 over
the final full76-case cycle despite generated ID regression. Final Comet curves,
paired images and audits are logged with the correct deep-identity factory.
See [IDENTITY_FLOW.md](IDENTITY_FLOW.md) for hashes and detailed results.
The new architecture report describes three BA reads, the frozen core, decoded
identity objective and deterministic recovery; final export is complete below.

### Final report delivered and follow-up paused

Rendered and inspected the23-page final PDF, including architecture diagrams,
code, metric curves, masks and all24 paired faces:
`reports/261001_deep_identity/flux4b_deep_identity1024_architecture.pdf`.
Uploaded to `Apps/temp/rsrch_new/2026-10-01/flux4b_deep_identity1024_architecture.pdf`
with verified Dropbox content integrity (5,059,984 bytes).
SHA256 `0ac45675d1ec943ef2327602028cfcb95c2d29094fb75f9e9e8ae1fd22c79136`.
The PDF and final audits are also logged to continuation Comet. Best500 weights
and Adam state remain preserved. The local BA convergence/report follow-up is
**PAUSED**; see the run's `dropbox_report_upload.json` and
`followup_completion.json`. No training/validation process remains.

## 2 October 2026 — Critical one-ID review and local full-denoiser benchmark

The user requested a critical comparison with CL14 and a16GB/48GB recommendation.
The current final-feature head learns its cached objective but regresses in
prompt-generated identity/quality; widening it further is not the recommended
next experiment. The proposed replacement keeps native denoising layers and
trains branch-local attention projections with fresh noise/timesteps. Full
analysis, historical CL14 trainable ownership and the two proposed settings are
in [ONE_ID_CRITICAL_REVIEW_20261002.md](ONE_ID_CRITICAL_REVIEW_20261002.md).

A bounded eight-update local test of the existing full-transformer K/V BA path
at768px/ref512, eight sites, rank128, microbatch1 and gradient checkpointing
passed:12,582,912 trainable parameters;9.449GiB peak reserved (59.09%);2.088s mean
per update after two warmups. All32 trainable tensors changed, gradients stayed
finite and initial/BA-off native predictions were exact. Artifacts and source:
`runs/critical_review_20261002/`. This establishes bounded feasibility on the
RTX4090 Laptop16GB, not quality or long-run thermal performance. The proposed
masked Q/K/V/output variants (approximately25.2M/50.3M at ranks128/256) remain
unimplemented and unmeasured. No long training run was started; GPU work ended
after the probe. Prior checkpoints and paused automations are preserved.

## 2 October 2026 — Approved online masked Q/K/V/O implemented and launched

Implemented the16GB setting requested after the critical review: native-initialized
face attention inside eight FLUX4B blocks, rank128 Q/K/V/output LoRA only
(25,165,824 parameters), fresh native noise/timesteps, full denoiser autograd,
masked native flow MSE, microbatch1/accumulation4 and LR5e-5. The original19
distinct same-ID target/reference pairs are used. Fixed24 validation uses the
reviewed native-generated masks,20 steps and CFG4 at0/500/1000/2000.

Pretrained checks passed: exact native BA-off/zero-mask parity, all64 trainable
tensors updated with finite gradients, every frozen tensor hash unchanged, and
bit-exact fresh-process replay including Adam/scheduler/RNG/cursor. The production
accumulation4 checks peaked at9.682GiB reserved and took7.97–9.13s/update. Focused
mask locality, original FLUX branch seam and composition tests pass. These are
admission results, not identity-quality improvements.

New run: `runs/flux4b_oneid_online_face_qkvo_r128_768_20261002`, Comet key
`c955799c57d94fd7a234dec331323e06`. The controller was launched and entered
step-zero generation before training. Details and commands are in
[ONLINE_FACE_BA.md](ONLINE_FACE_BA.md). Old runs/checkpoints remain preserved;
the recorded source patch has a new optional masked-branch path, so old runs
must use their source snapshots if replayed. No remote GPU operation was made.

At15:21 UTC, step-zero validation/scoring was complete and the main training
worker had completed six optimizer updates from checkpoint0. All32 B matrices
updated at step1; finite gradients,9.641GiB peak reserved,92% sampled GPU use,
10.10s/update mean. Early remaining-training ETA5.6h excludes three validation
panels. The24-image baseline scored ID_sim0.146024 versus native0.331399;
paired crops show substantial untrained face distortion. Exterior pixels are
exactly preserved. No trained quality improvement is claimed yet. Evidence is
in the run's `startup_confirmation.json`, validation0 metrics and paired sheets.
Startup monitoring ended; the serial controller continues to2k and final scoring.

### 2 October — Step1000 validation and continuation confirmed

The user reported an apparent stop at1k. Inspection found the original controller
and inference worker still active; validation generation was19/24 complete.
No restart was necessary. All24 images then completed generation, decoding and
scoring, followed by the automatic `--step 2000 --resume 1000` training worker.
The full checkpoint has finite adapter/Adam tensors, optimizer/scheduler step1000,
cursor4000 and RNG states; immutable code/config/data/mask checks passed.
Updates1001–1005 were observed with finite gradients and the correct continued
data cursor/LR. Peak reserved9.859GiB remains below90%.

Fixed24 ID_sim is0.146024 /0.365766 /0.430254 at0/500/1000; CLIP is27.702707 /
27.249032 /26.967033. Comet's read API independently confirms these points in
the same experiment. Step1000 improves ID_sim on22/24 cases versus500. All24
paired face crops were inspected, with clear removal of much of the initial
distortion and residual quality limitations. Pixel exterior remains exact.
Evidence: the run's `comet_metrics_1000_verified.json` and
`resume_1000_confirmation.json`. The existing serial controller continues to2k
and final scoring; no model or training code was changed for this check.

### 3 October — Completed online one-ID run and source preservation

All2000 updates and0/500/1000/2000 fixed24 generation/decoding/scoring completed;
the controller finished at2026-10-02 23:40:58 UTC. ID_sim is0.146024 /0.365766 /
0.430254 /0.419023; best1000 remains preserved. CLIP at2000 is27.321863 versus
26.967033 at1000. All2000 logged losses/gradients are finite, peak reserved is
9.859GiB and the final24 exteriors are pixel-exact. Final paired faces were
reviewed: clear gains from untrained BA, with residual softness/accessory and
lighting changes. These are one-ID fitting results, not unseen-ID evidence.

All20 immutable run sources match the setup being committed at the user's
request. [ONLINE_FACE_BA.md](ONLINE_FACE_BA.md) records completed results,
artifact prerequisites and the recommended next48GB experiment: fresh4B BA on
multi-ID data with identity-disjoint fixed96 validation before a9B comparison.
The one-ID runner's19/24 assumptions and private native bundle are documented;
the multi-ID/9B settings are proposals, not launch-ready claims. No new GPU run
or machine operation was requested or started during this commit/review.

### 3 October — New-machine FLUX4B multi-ID launcher (prepared only)

Added `scripts/run_flux4b_multi_id.sh`, its serial controller and
`configs/flux4b_48_multi_id_large.yaml`: full pinned adjusted Large dataset,
validation identity aliases excluded, fresh rank128 Q/K/V/O BA only, target768,
reference512, microbatch1/accumulation8, LR5e-5,10k updates, save500 and original
fixed96 validation every2k including step0/final. Pair sampling remains shuffled
deterministic next-view pairs; no identity-balanced sampler or true batch8 is
claimed. Trainable architecture/native attention/training loss are unchanged.

Generalized the online runner's initialization and validation to accept an
explicit native96 bundle, full-cache header verification, dynamic panel counts
and a measured native ID_sim baseline. The new controller imports already
downloaded images on the destination, checks provenance/disjointness and disk
space, runs GPU admission including exact optimizer replay, generates native96
masks, then runs training/decoding/ID+CLIP/face-quality scoring serially. Resume
uses complete checkpoints, preserves uncheckpointed metric tails, retries
unfinished stages and retains the Comet key. Missing native face boxes require
review; they are not fabricated or silently dropped. Setup and worker source
hashes are frozen; earlier completed one-ID source is preserved at3714e7d.

Verification here is limited to static syntax/configuration, CLI dry-run and
focused resume-selection checks. Per the user's narrowed request, **no dataset
was prepared and no training/GPU job was launched locally or remotely**.
48GB memory/throughput remain unmeasured; admission executes on the destination.
The full-data eager-cache estimate is roughly400–500GiB: about47k unique
captions alone imply ~347GiB of BF16 text embeddings, before image latents.
This does not apply to the earlier4,096-pair pilot, which used76GB total disk
after caching on a200GB host. It is a limit of this full-data launcher, not a
general GPU-training requirement.
See `docs/deployment.md` for launch/resume commands. No commit/push this turn.

### 3 October — Vast GB10 inventory check

The user's manually rented Vast instance `53994096` was visible as running in
Denmark at an API-reported $0.49829/hour. It has one GB10, 20 ARM64 CPU cores,
124,544 MB API-reported RAM, 212.3 GB allocated disk, and marketplace-reported
3,555.6/2,241.8 Mb/s internet down/up. SSH key authentication succeeded.
On-host `free -h` reported 121 GiB total shared system memory; `nvidia-smi`
reported `NVIDIA GB10`, driver 580.178.04, compute capability 12.1, and GPU
memory `N/A`. Vast's dedicated `gpu_ram` field is zero for this instance. The
default `python3` and `/opt/miniforge3/bin/python` could not import `torch`;
no CUDA allocator capacity, model load, training memory, or throughput was
measured. This 212.3 GB allocation can support a pilot like the earlier
4,096-pair run (76 GB total disk after caching); the prepared full-data
eager-cache launcher exceeds it. ARM64 dependency installation and training
remain untested. This GB10 inspection is separate from the proposed 48 GB
discrete-GPU experiment.

### 3 October — GB10 deployment and on-demand conditioning

User confirmed Vast53994096 (superseding53574065), then rejected the full
training-cache default and requested both options. Default4B multi-ID config
now uses `data.conditioning: online`: frozen native encoder/VAE produce each
training pair on demand with RNG isolation and no training cache writes.
`--conditioning cached` retains the previous full-precompute option under a
separate run name; the mode is immutable on resume. Only fixed96 validation
inputs are cached by default. The earlier400–500GiB number described storing
all ~47k padded text embeddings (~7.5MiB each) plus latents, not a model or
training requirement. It no longer applies to the default setup.

Host SSH/CUDA verified: ARM64, GB10 compute12.1,121GiB unified RAM,212GB disk,
driver580.178.04. PyTorch2.13+cu130 installs and a CUDA matmul passes. NVIDIA's
pinned cuSPARSELt0.8.1 has legacy`manylinux2014_sbsa` WHEEL metadata; its actual
ELF machine183 is AArch64. `check_environment.py` accepts only this specific
metadata warning after verifying the binary, while preserving all other uv
dependency checks. Environment setup/data transfer are in progress; no new
training result is claimed here. Native parity and exact resume admission
will execute on the destination before the long run.

ARM64 scoring setup also required removing the x86-only `+cpu` local-version
suffix from the existing torch2.2.0/torchvision0.17.0 constraints; release
versions, metric code and model definitions stay unchanged. Both scoring
environments now pass uv dependency checks. All FLUX/text/VAE locked weights
are downloaded and verified. The fixed96 native job is running while the raw
17GB dataset transfers in16 disjoint resumable streams. No training cache is
being built.

Both training modes are exposed through `--conditioning online|cached`; the
provided config defaults to online. CUDA device placement now explicitly
matches cache loading for CPU-created token IDs and masks. Online admission
compares an independently precomputed validation pair against live encoding
and checks unchanged CPU/CUDA RNG before the existing gradient/resume tests.
The fixed96-only cache measures191MB on the GB10.

`rsrch_training` is supervised and queued behind completed native generation
and a verified dataset-transfer receipt; optimizer updates have not yet begun.
Local transfer watcher/retry paths are in `MACHINES.md`. A few of16 simultaneous
SSH handshakes were reset; the transfer helper now defaults to8 streams and
retries failures while preserving partial files. The running completion watcher
will retry any failed initial chunks before releasing training.

### 3 October — Cosmic switch and short GB10 hardware proof

User requested Cosmic instead of Large, no long image validation before
training, and measured download speed. Cosmic metadata pairs each scene with
its associated reference face bank (2–10 candidates); these are target-specific
pseudo-identities, not curated multi-photo person IDs. The importer retains
the first sorted eligible distinct reference. The original report's22,140
effective count included a192px face filter; this importer does not impose it.

Stopped `rsrch_native`, `rsrch_training`, the Large transfer and its local
completion watcher; partial artifacts remain intact. Both pinned Drive links
failed: the target returned an explicit quota-exceeded HTML page, and the
reference endpoint initially advertised an8,560,574,742-byte binary but later
returned an HTML error too. No archive download completed; observed transfer
speed is0MB/s while blocked. Downloader now tries the official public download
endpoint on a gdown parsing failure, rejects HTML errors, and verifies length
and Range before accepting a resumed file. This does not bypass Drive quotas.

To get a real hardware check without waiting on downloads, imported all15
complete original first-reference pairs present locally into a separately
named manifest, preserving prompts/boxes and original selected metadata.
No image bytes overlap the original validation panel. The1,914,880-byte
bundle plus launcher/config transferred in2.98s including three SSH setups
(not a Google Drive throughput measurement). The manifest subset SHA256 is
2756c5c8f1d5d119a7c4251af17685cfa6b84b7c3e0f1ae5513742192b310bb7.

`rsrch_cosmic_smoke` runs `scripts.run_cosmic_smoke`, with immutable source
copies, config, manifest hash, command receipts and Comet key
898fb16369334b9e950bfc9605308d7d in project rsrch_new. Run:
`runs/flux4b_cosmic_hardware15_qkvo_r128`. Full4B native gradient path,
25,165,824 BA parameters, Q/K/V/O rank128, eight sites,768px targets,
microbatch1/accum8, fresh noise/timesteps,100-update budget. Conditioning is
online with no training cache. Only one existing validation input is used
for live/cache parity; pretrained native/off and two-update finite-gradient
checks run before the actual optimizer. Main run checkpoints at2 updates and
resumes in a fresh worker, then checkpoints every25. Generated-image panels
and scoring are omitted for this hardware-only smoke by explicit user request.
Full-data training remains blocked by unavailable full reference images.
Do not interpret smoke loss or15 pairs as held-out identity improvement.

Measured GB10 admission passed: exact native/BA-off and zero-mask predictions,
exact live/cache conditioning and RNG preservation, finite two-update
gradients with all64 BA tensors changed, and frozen model hash unchanged.
Branch-on/off prediction mean absolute difference0.212628; trained-versus-
initial difference0.009181. Peak CUDA reserved16.684GiB (13.72% of the
GB10-reported unified capacity). These are real pretrained tensor checks,
not image-quality measurements. Comet URL:
https://www.comet.com/nikolay-2104/rsrch-new/898fb16369334b9e950bfc9605308d7d

The actual Cosmic smoke optimizer completed steps1/2:22.95/21.88 seconds,
loss0.98681/0.99949, finite gradient norms0.09960/0.08088, effective batch8,
all32 B matrices updated on step1, peak reserved17.006GiB. Saved
checkpoint-000002 and launched a fresh worker to restore it and continue
to100. Startup/admission optimizer updates are separate from these two
recorded training steps. No loss trend is inferred from two random batches.

Latest user instruction: continue transfer using whichever dataset is faster.
Resumed the original Large dataset with8 resumable streams and no queue
watcher: local source17.127GB, already14.271GB on remote at11:00:54UTC.
This preserves the fastest route to a complete dataset while Cosmic15 trains.
The old96-image native job and old Large training queue remain stopped.
Local transfer PID/log: scratch/deploy_53994096_large_resume.{pid,log}.

Measured resumed Large transfer:14,271,018,566 to14,422,539,965 bytes in43s,
3.52MB/s aggregate (du apparent bytes, includes small directory overhead).
About2.705GB remained, ~13minutes at that instantaneous rate.

Fresh-process resume succeeded: steps3/4 were recorded after loading
checkpoint-000002 (Adam/scheduler/RNG/cursor restore passed compatibility
checks). Step4: finite loss1.01674, gradient norm0.07585,21.93s/update,
peak reserved16.898GiB; main optimizer has now seen32 samples. This confirms
save/reload and continued updates, not a bit-exact replay comparison.
Both background processes remain active: Cosmic hardware smoke and Large
raw-image rsync. No long validation or full-data optimizer was started.

### 3 October — GB10 larger-backbone feasibility review (proposal, not a new run)

User asked whether the high-memory machine supports a larger base and how to
utilize it well. Running4B job was left unchanged; no second GPU job was run.
Observed host:121GiB unified CPU/GPU RAM,175GiB free disk. At step26, last10
updates averaged22.273s (effective batch8:0.3592 training images/s), peak
reserved17.076GiB. Ten subsequent one-second GPU-utilization samples were
[96,96,77,96,95,95,96,94,96,96], mean93.7%. This is busy time, not achieved
FLOP throughput; there is spare memory but no demonstrated idle GPU capacity.
NVIDIA documents128GB LPDDR5x shared memory and273GB/s bandwidth; the advertised
1PFLOP is sparse FP4 and is not BF16 training throughput:
https://docs.nvidia.com/dgx/dgx-spark/hardware.html

Recommended next model: FLUX.2-klein-base-9B, frozen BF16 base with BA-only
training. The official card identifies it as undistilled and intended for
fine-tuning/research: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B
Pinned locks/weights-flux80.json has16.910GiB generator,15.271GiB Qwen3-8B
encoder and0.313GiB VAE (32.495GiB total file bytes, approximately resident
BF16 weights). A CPU meta-device construction, with no weights or CUDA
allocation, confirmed9,078,581,248 generator parameters, width4096/32heads,
8dual+24single blocks. Current4B:3,875,544,576, width3072/24heads,5+20blocks.
Eight rank128 Q/K/V/O BA sites would contain33,554,432 trainable parameters
(about0.5GiB for FP32 parameters/gradients/two Adam moments); rank256 doubles
that to67,108,864 and about1GiB. Rank alone does not consume the available
memory or guarantee better ID scores.

Planning budget for9B at768px/ref512, microbatch1, online conditioning and
activation checkpointing: approximately40–60GiB including activations and
workspaces. This is an estimate, not measured9B training; exact throughput
and peak memory require a serial benchmark after the active run. Existing
disk capacity comfortably covers the additional ~35GB pinned files.

Implementation gaps: config.py and masked_face_attention.install restrict
masked_face_qkvo to4B and hardcode its site map. Older9B configs use the
different reference_delta branch, not the present face-ownership mechanism.
Port the current path to width4096 and9B sites dual[2,4,6,7],
single[4,10,16,22], derive inventory counts instead of4B constants, and
initialize new adapters/optimizer. Re-encode text with Qwen3-8B and generate
new9B native images/masks for inference;4B adapter tensors and native masks
are not reusable. Current training is truly microbatch1 with accumulation8.
Real batches2/4 require collation/bucketing, per-example reference-key masks
and independent noise/timesteps; changing grad_accum alone is not batching.

Proposed tuning order: short9B parity/finite-gradient/resume proof at matched
768/512/r128; benchmark checkpointing on/off to exchange available memory
for reduced recomputation; implement/benchmark true batches1,2,4 while
retaining effective batch8 (accum8,4,2). Choose by images/sec, preserve CPU
memory headroom (~20GiB), then consider rank256 or1024px as separate quality
experiments. Keep online conditioning and avoid a full dataset cache. Use
full transferred Large data for substantive results; the15-pair hardware
smoke cannot establish generalization. Short explicitly named held-out image
checks can precede longer evaluation, matching the user's latest request.

Alternatives: Qwen-Image-2.1 has a7B visual generator but is a separate
architecture, not a larger FLUX sibling:
https://huggingface.co/Qwen/Qwen-Image-2.1
FLUX.2-dev32B plus its24B Mistral encoder imply roughly104GiB BF16 weights
before training activations/OS, so the current all-resident online layout
is too tight to recommend as the next step. It would need a separate port
and memory strategy, with unmeasured speed:
https://huggingface.co/black-forest-labs/FLUX.2-dev
https://github.com/black-forest-labs/flux2

Access prerequisite: pinned9B weight HEAD returned401/GatedRepo. No HF token
was available in the local project environment or Hugging Face token cache.
A token authorized for the model's gated repository is needed before download.
No model download, training configuration mutation, commit or push was made
as part of this review.

### 3 October — Existing Hugging Face credential found; 9B access still gated

At the user's request, searched environment files in other local projects
under /home/kolyangg without printing credential values. Found one genuine
distinct token in voice_bot/VoiceAssistant/.env (HUGGING_FACE_TOKEN). A broader
non-template environment-file search found no additional distinct tokens.
Hugging Face whoami-v2 returned200; the token is valid and fine-grained.
Authenticated access to the pinned FLUX.2-klein-base-9B weight returned403
GatedRepo: the account/token is not on the authorized access list. Thus the
previous missing-token finding is superseded; model authorization remains
unresolved. Reused this credential as HF_TOKEN in rsrch_new/.env only after
confirming that file is ignored/untracked, and set permissions0600. Source
credential file unchanged; no credentials were printed, tracked, or sent to
the rented host in this review.

Recommendation remains9B Base, BA-only rank128 initially,768/ref512 with
actual microbatch2 or4 after implementing correct batching, effective8.
Account owner should open the official9B model page using the account behind
this token, review/accept or request gated access, and ensure the fine-grained
token permits reading that repository. Recheck the exact pinned weight URL
before downloading; do not substitute unofficial mirrors to avoid the gate.
Meanwhile the4B job/data transfer can continue. Next engineering work can
benchmark checkpointing on/off and true batches on4B, then port the current
face-specific BA path to9B, use new adapters and9B-generated routing masks,
and run a short pretrained parity/gradient/resume check before a larger
full-Large training pilot. Suggested first quality pilot:2000 updates with a
separately named fixed12 held-out panel at0/1000/2000, preserving the original
prompts/order/seeds/reference images and metric definitions. No claim of9B
training throughput or memory measurement is made yet.

### 3 October — Prepared HSE two-V100 FP16/DDP experiment (not submitted)

Prepared configs/clust/flux4b_2v100.yaml and jobs/flux4b_clust_2v100.sbatch for
clust:/home/nasilaev/rsrch_new, account proj_1892, rocky, one node, two V10032GB,
16 CPUs and24 hours. FLUX.2-klein Base4B, eight masked Q/K/V/output branch sites,
rank128,768px targets/ref512, fresh adapters, LR5e-5, microbatch1 per rank and
accumulation4 give globalbatch8. Initial2000 updates, checkpoints500 and unchanged
fixed96 at0/2000; training workers exit before serial patched validation.

This is a separately named FP16 experiment, not a BF16-equivalent result. Denoiser
FP16 uses dynamic loss scaling; adapters/optimizer/loss are FP32. Online frozen
text encoding is FP32 on CPU per rank; the VAE is FP32 on GPU. Precision participates
in adapter/cache/native-mask identity. New native images/masks are required.
Distributed checkpoints preserve the global sample cursor, per-rank RNG and scaler.
Pretrained native/off equality, frozen equality, branch gradients/updates, exact
fresh-process two-rank resume, and reserved memory below90% per GPU are enforced
before long training. Runtime fit, throughput and FP16 pretrained stability remain
unmeasured. No V100 allocation or training was started.

Software verification: isolated Python3.11.13/PyTorch2.7.1+cu126/torchvision0.22.1
with pinned Transformer/Diffusers dependencies installed and passed dependency
validation. Compiled architecture flags include sm_70. On the selected source
snapshot, seven existing FLUX invariant checks, masked attention/background checks,
and a real two-process CPU/Gloo test passed. The latter compares averaged DDP
gradients against a serial global batch and exact adapters/Adam/scheduler/scaler/
per-rank RNG after a fresh-process restart. CPU tests are not pretrained admission.
The initial local dependency smoke omitted torchvision; installing the same
explicit torchvision pin already present in setup_clust_v100.sh resolved the
import failure. No GPU memory measurements were made. Remote bash syntax and
sbatch --test-only accepted the exact resource request; squeue remained empty.

The existing cluster validation references matched all eight original SHA256s;
current prompts, boxes and metric embeddings were synchronized. Historical Cosmic
paths were missing/inaccessible. User-authorized background rsync is transferring
47,500 local adjusted-Large images (17,116,845,489 bytes) into nasilaev/datasets.
The verified metadata/import yields47,341 eligible cross-view pairs,110 excluded
pairs,49 duplicate-content pairs and zero validation-image overlap. Cluster-path
manifest SHA256:fea6c505016c73c995d25a65ca02202d27a2b622e9a463ac3096f8b1586c6302.
A transfer publisher writes data/clust_large_transfer.json only after all four
checksum dry runs pass; the launcher requires this receipt. Several first-pass
SSH streams stalled near92%; they were restarted resumably with120s rsync timeout
and30s SSH keepalives. Dataset transfer logs/receipts remain ignored.

The user authorized committing/pushing the cluster implementation but requested
approval before training. Concurrent 9B/true-microbatch/Vast edits are excluded
from this commit. Cluster dependency/weight provisioning and private Comet setup
remain prerequisites. See docs/clust_deployment.md for exact commands and limits.

### 3 October — Gated 9B access granted; deployment and throughput admission

The authorized account now returns HTTP200 for the pinned 9B weights. Reused
the ignored HF_TOKEN on Vast53994096 through encrypted SSH stdin; credentials
remain untracked with mode0600. Downloaded pinned FLUX.2-klein-base-9B
32773329fbe7e81a90ef971740e8ba4b0364ecf3 (18.157GB), Qwen3-8B
b968826d9c46dd6066d109eabc6255188de91218 (16.397GB), and shared VAE.
The previous authorization blocker is resolved.

Preserved completed 4B Cosmic hardware smoke at100 updates. Created isolated
/workspace/rsrch_9b for the new 9B code; weights/environments are shared, runs
are separate, and both checkouts share the same GPU lock. The current 9B
masked Q/K/V/O branch uses width4096 at dual2/4/6/7 and single4/10/16/22;
rank128 gives33,554,432 trainable parameters (64 tensors), with the native
backbone frozen. True batches preserve each example's reference key mask,
independent noise/timestep and face-normalized loss. No native token padding.
The small BA read executes per example while native attention executes batched.
A focused CPU check matched separate BA outputs and all parameter gradients.

Serial real9B two-update benchmarks use8 distinct Large pairs only for speed,
768/ref512, effective batch8 and online conditioning. First measured profile
(microbatch1/accum8, checkpointing enabled):51.158sec/update after warmup,
35.678GiB peak CUDA reserved, finite gradient norms .04352/.03510. This does
not establish quality or full-data training. Other profiles and exact native/
resume admission are pending. Full Large transfer continues; no training
latent/text cache is planned. The next quality run is a separately named
fixed12 held-out 2000-update pilot with validation0/1000/2000, new9B-native
images/masks and the original ID_sim/CLIP definitions.

Transfer completion: all four streams returned0 from checksum verification with
zero file differences. The280 reported differences were exclusively shared
directory modification times changed by concurrent streams; they do not affect
file-content validation. The47,500-image dataset,47,341-pair manifest/audit and
verified completion receipt were published to clust. Training-source readiness
is now true for Large; paired Cosmic remains unavailable. No transfer remains
active and no GPU job has been submitted.

Throughput follow-up: checkpointing OFF, batch1/accum8 measured40.969sec/update
and51.789GiB, with the same first two gradient norms as checkpointing ON.
Batch2/accum4 measured41.909sec/update and69.891GiB: larger batches were
not faster in this short measurement. Batch4 is being tested with an80%
allocator cap to retain unified-system memory headroom. Its uncapped startup
was stopped before any update, with the incomplete directory preserved.

Removed unnecessary startup work before full-data admission: masks are built
only for training rows actually used, and initialization records target byte
hashes/source boxes instead of evaluating47k dense masks. Mask geometry can
be obtained without decoding/resizing a photo;20 original images matched the
old transform exactly. Separable coordinate broadcasting in face_alpha matches
the former full-grid result bit for bit (including fractional/border boxes).
Measured local mask construction fell from24.9ms to2.1ms per sample. Dataset
import also avoids RGB decoding when the original adjusted image already
needs no crop. These preserve mask values and model mathematics.

Large transfer was resumed with16 disjoint file-level streams after the final
two directory streams slowed. Preserve logs in scratch/deploy_53994096_large_*;
the unrelated cluster transfer was not touched. SSH TCP retransmissions are
visible on this link, so transfer ETA is variable.

9B benchmark selection: batch4 reached the97.30GiB (80%) allocator cap and
raised CUDA OOM before completing an update; preserved its worker traceback.
No host OOM or driver changes. Selected microbatch1/accum8 with checkpointing
OFF (40.969sec/update;51.789GiB); batch2 offered no speed improvement.
The fresh pilot command is `scripts.run_multi_id_face_ba --run
runs/flux9b_large_qkvo_r128_pilot12 --config
runs/flux9b_gb10_benchmark/selected_config.yaml --images-root
/workspace/datasets/large_dataset --pilot-panel --id-clip-only` in the9B
checkout, supervised as rsrch_9b_pilot. Configuration dry-run confirms
2000 updates and panels0/1000/2000. Training-only estimate from benchmark
is22.8hours, not yet a full-data run measurement. Startup admission is still
required. Original source inventory contains47,500 files,17,116,845,489bytes;
remote data import waits for every file to match size.

Full transfer completed and every source file passed the size inventory check.
Pinned importer produced47,341 pairs, excluding110 held-out pairs and49
duplicate-content pairs, with0 validation-image-overlap pairs. Remote manifest
SHA256:3deb922e295c300d9a0e72578d0e0b11eb48ccf1550037f3ab68a16c55d61d22.
Supervisor rsrch_9b_pilot started at approximately12:05UTC on3October;
controllerPID29311. First observed stage:preflight. It owns the shared GPU
lock and will run cache12, admission, native12/masks/ID+CLIP, then serial
training/validation. No long96 panel or full training cache is requested.
Deployment bundles retained locally for exact source reconstruction:
- `scratch/flux9b_code.tar.gz` SHA256 `a101709cbbddb8781085b3f533ec9d93c7c8952429fe7c1f8baf955f55c4d946`
- `scratch/flux9b_startup_updates.tar.gz` SHA256 `57b35a65a4b07fc1c45cc775408ccf5e501d0c83bd5d807e77b654086faf8776`

Pretrained9B admission native checks passed around12:14UTC on the largest
training face support: native/off and zero-mask exact; all64 BA tensors finite
and updated; full frozen parameter hash unchanged; live/cached conditioning
exact and training RNG preserved. Branch-on/off mean absolute prediction
difference .374550; two-update prediction change .008759. Peak CUDA reserved
52.1328GiB (42.86%). Exact fresh-worker save/resume check is running next.

### 3 October — Authorized20k HSE run and one-V100 immediate probe

The user approved20,000 optimizer updates on two V100s, then asked for one
available V100 to start some training sooner. Kept the approved4B/r128,
768/ref512, effectivebatch8 scientific setup and fixed96 validation at0 and
every2000 updates. Increased the Slurm envelope to7 days, retaining500-update
atomic checkpoints. Runtime is unmeasured; this limit is not a completion ETA.

Submitted training4372978 on rocky (two typed V100s,16 CPUs), initially gated
on CPU setup4372977. CPU setup did not start: real scheduling estimates were
hours later despite an immediate dry-run estimate. Cancelled4372977 while it
was still pending. Submitted one-V100/8CPU30-minute test job4372995, with
bootstrap plus a separately named two-update training probe on eight eligible
pairs. At15:19MSK it was RUNNING on cn-025, in setup with no optimizer result yet.
Training4372978 now has afterok:4372995 dependency. The probe checks native/off,
actual branch gradients/updates, frozen weights, conditioning parity and GPU
memory. It does not replace the full-data two-rank resume/admission gate or
claim image quality. Each experiment uses its own Comet identity.

Data receipt and47,341-pair manifest remain verified. Installed only the
necessary Comet credential into a private remote0600 .env via SSH stdin;
no credentials entered tracked files or logs. Exact submissions and source
hashes are recorded remotely at runs/clust_20k_submission and mirrored under
local ignored scratch/clust-start-20k/submission. The source is the committed
cluster snapshot6156c31 plus explicitly recorded20k/smoke-launch changes;
concurrent Vast/9B workspace edits were not synchronized to clust.

### 3 October — User-directed immediate training; validation deferred

The user explicitly requested immediate training after objecting to startup
delay. Stopped the supervised pre-training controller and its children,
preserving all admission evidence. Native/off, zero-mask, finite gradients,
all64 branch updates, frozen equality, conditioning/RNG and memory had passed;
two uninterrupted full-data updates completed. Fresh-process replay was
interrupted before final comparison, so exact replay is NOT claimed for this
9B run. No remaining GPU diagnostic process was present before launch.

Started `scripts.run_flux9b_training_first` under the same rsrch_9b_pilot
supervisor, run `runs/flux9b_large_qkvo_r128_pilot12`. It initializes fresh
adapters/Adam, saves checkpoint0 before the first update within the training
worker, then trains immediately to1000. Only then does it generate native12
images/masks and evaluate preserved checkpoints0 and1000; next it continues
to2000 and evaluates2000. Thus step0 is evaluated retrospectively from the
actual untrained checkpoint, with unchanged prompts/seeds/held-out panel.
The validation schedule deviation is recorded in execution_plan.json and
Comet. No cache or baseline generation blocks the first training segment.

Comet key:7b7169cd61b64a2cace2fe1e52861afc
https://www.comet.com/nikolay-2104/rsrch-new/7b7169cd61b64a2cace2fe1e52861afc
Observed main controllerPID30088, training workerPID30097, status train_1000.
Its source/config/manifest identity is frozen in training_identity.json and
source_snapshot. A separate inference identity is written only once native
masks have been generated and frozen. Existing training save/resume checks
remain in force; no checkpoint configuration hashes were bypassed or edited.

### 3 October — HSE short-job bootstrap failure and repair

The first one-V100 test allocation, job4372995, started immediately on cn-025
at15:18:57MSK but failed at15:22:08MSK before Python environment preparation or
any optimizer updates. The uv installer reported curl write failure and a
missing site temporary directory /tmp/job-4372995. No GPU admission or
training result is claimed. Both setup launchers now create a private
project scratch directory per Slurm job and set TMPDIR there. The pinned
uv0.11.12 executable is being transferred directly to avoid another installer
download in the GPU allocation. This is one diagnosed retry; the20k job
remains dependent on successful prerequisite completion.

Main training confirmed at12:29UTC: actual updates1 and2 completed in41.914
and40.749sec, respectively (16 training examples). Finite losses .84734 and
.95390 and gradient norms .04714/.03169; all32 B matrices updated on step1.
Peak reserved51.627GiB (42.45% of device capacity); sampled GPU utilization
96%. Actual checkpoint0 contains manifest, adapters and optimizer/RNG state.
Current training ETA to2000 is22.94hours plus deferred validation. Main run
is active with Comet7b7169cd61b64a2cace2fe1e52861afc. Per the user's earlier
instruction, stopped active monitoring after these two verified updates.

### 3 October — Five-hour total budget: separate batch1 pilot

The user requires2000 updates plus validation within5hours. The prior41sec
update contained8 complete9B training examples (online encoder/VAE and full
denoiser forward/backward); sampled utilization96% showed no idle-GPU issue.
Stopped that run after9 logged updates, preserving its checkpoint0, metrics
and source snapshot. Updates1–9 were not a saved resumable checkpoint.

Started fresh `runs/flux9b_large_qkvo_r128_fast5h_b1`, config
`configs/flux9b_gb10_large_qkvo_r128_fast5h.yaml`: same9B/768/ref512/r128,
full47,341-pair sampling pool, online conditioning, checkpointingOFF, but
microbatch1/accum1 instead ofaccum8. This is2000 examples over2000 updates,
not the previous16000-example budget; it is a separately named time-bounded
pilot and not an equal-data throughput improvement. LR/warmup/loss unchanged.

The training-first controller preservescheckpoint0, trains to1000, evaluates
0/1000, then trains/evaluates2000. A short inference timing hook runs after
update2 using the already loaded backbone, cached real validation inputs,
the largest reference layout and full-target BA mask. It preserves RNG and
model mode, measures native/BA CFG forward cost, and extrapolates all48
images (native12 plus three BA panels), with a30-minute reserve for loading,
decode and scoring. This is a runtime estimate, not scored image validation.
Actual timing and Comet key are pending the first updates. No new cache or
long admission sequence precedes training.

### 3 October — Dedicated activated Conda environment for clust

Retry4373014 immediately received cn-025 but failed in one second with
`git: command not found`, before environment preparation or model training.
The user requested activating the cluster environment. Inspection of the old
photomaker_NS environment found Python3.10.18 and missing FLUX dependencies;
the user then explicitly approved creating a new environment. Created
/home/nasilaev/.conda/envs/rsrch_new with Python3.11.13 and Git2.56.0 from
conda-forge, and verified activation and both executable paths. Old PhotoMaker
environments were not modified; their package inventory was saved under
scratch/clust-existing-env. The training/setup launchers now activate rsrch_new
and disable user-site packages. Separate pinned metric tools remain isolated
under envs/clust-v100 because their CLIP/Transformers versions conflict with
the training stack. Job4372978 was held pending corrected setup. No pretrained
V100 result or optimizer update is claimed at this point.

Five-hour pilot measured at12:41UTC: first two updates averaged5.831sec
(first includes warmup). Native CFG inference extrapolates to51.307sec/image
at20 steps; full-target BA worst-case is55.727sec/image. Across native12 and
three BA12 panels this is2621.85sec (43.7min) of generation. Adding2000
updates and a30-minute loading/decode/scoring allowance gives4.468hours.
This is a measured projection, not completed validation or a guaranteed
wall-clock deadline. The short timing probe ended and training continued.
Comet key:cfbd6f879e3d4158959bbbef7fb8e894
https://www.comet.com/nikolay-2104/rsrch-new/cfbd6f879e3d4158959bbbef7fb8e894
Profiler results are in the run's five_hour_budget.json and Comet metrics.

Post-probe health confirmed throughupdate25: latest updates5.16/5.11sec,
finite gradients/losses,51.568GiB peak reserved. Rolling training rate694
updates/hour (~5.19sec/update) gives~4.11h total when combined with measured
generation plus30min reserve. Report4–4.5h expected, preserving margin below
5h. Active training remains supervised; no further tuning or monitoring needed
for this startup request.

### 3 October — Compute-node download timeout; prepared shared caches

One-V100 job4373031 activated rsrch_new, found its Git and GCC, and verified
the patched Toolkit revision, then failed before training after44 seconds of
connection timeouts to download.pytorch.org. The same endpoint worked on the
login node. Dependency installation and downloads were moved to login with
one build worker, excluding model imports/inference; GPU checks remain in
Slurm. All three pinned environments passed uv dependency checks. The full
16.148GB pinned FLUX4B, Qwen3-4B and VAE files downloaded successfully and were
size-verified; aliases point to exact locked HF revisions. Original metric
weight hashes and timm revisions are now locked separately for offline cache
preparation without importing models. Compute jobs set HF_HUB_OFFLINE=1.
The hold placed on4372978 became JobHeldAdmin and the site's release operation
was denied. Cancelled that never-started job and submitted replacement4373033
with the same two-V10020k configuration. Its prerequisite remains unmet until
the corrected GPU probe passes. No optimizer update or GPU memory measurement
is claimed from these failed preparation jobs.

### 3 October — Measured FP16 overflow; FP32 V100 training passed

Prepared one-V100 job4373054 passed software invariants and independent
conditioning preparation but failed native/off equality before updates.
Diagnostic4373072 on cn-001 located the first FP16 nonfinite values in the
native text stream at double_blocks.4, before branch installation; all294,912
output values became nonfinite. Its repeat/off comparison therefore failed.
Same-pair FP32 predictions were finite and bitwise equal across repetition and
BA-off, with forward peak18.0390625GiB. No branch architecture was changed.

Job4373072 then completed the named FP32 two-update probe: finite gradients,
all64 trainable tensors updated, all frozen parameters bitwise unchanged,
exact native/off and zero-mask outputs, exact independent conditioning cache
comparison and preserved RNG. Peak CUDA reserved18.2421875GiB, fraction
0.5748731501463726; trained prediction mean-absolute change0.010671225376427174.
Slurm COMPLETED0:0 in6m12s for both diagnostics plus the probe. This is not
full-data throughput, two-rank save/resume, or fixed96 quality validation.
Reports: runs/clust_v100_dtype_4373072/{float16,float32}.json and
runs/clust_v100_two_update_smoke_4373072/native_checks.json. Successful smoke
Comet key9394bd6cccc043908396037da41f3fa7; direct live uploads were unreliable,
so the result was republished from login using that same existing key.

The active two-V10020k config now uses FP32 and a distinct fp32 run name;
Base4B/r128,768-ref512,globalbatch8,47,341 pairs,fixed96 and native loss remain.
Its own full-data two-rank admission remains mandatory. Cluster logging now
uses offline get_or_create archives with one persisted key and a login-side
uploader. A two-session actual-SDK check passed key/metadata continuity, and
the older pinned metrics SDK passed archive compatibility. The earlier failed
online smoke7684368f8fbb4dd784fb2ba13afe1f31 was recovered by force-upload as
61d7fe6c35264432b46b9e52a0bac3df; that copy is explicitly distinct, not a changed
canonical run key. Future uploads deliberately never use force-upload.

### 3 October — FP32 two-V10020k allocation is running

Submitted FP32 job4373118 from source569d65ad473563e7acab86dcc2c6bdde4ac205f1
with the authorized20,000-update configuration. Slurm started it immediately
at16:14:31MSK on cn-009 with two Tesla V100-SXM2-32GB devices
(31.7325439453125GiB each). Runtime verified PyTorch2.7.1+cu126/CUDA12.6,
sm_70 kernels, the completed dataset receipt and711.06GiB available host RAM.
The old blocked FP16 job4373033 was cancelled before execution.

Current run: runs/flux4b_clust_2v100_fp32_qkvo_r128_20k_20261003. At this record
the controller is checking full-data identity before its fixed96 conditioning,
two-rank admission and native validation stages; no main-run optimizer update
is claimed. The separately named one-V100 two-update FP32 result above passed.
A job-scoped login uploader (initial PID1823854) is running for4373118 and will
stop when that job ends. Queue/log state, source hashes and submission replies
are persisted in runs/clust_20k_submission and mirrored under local ignored
scratch/clust-start-20k/evidence. Main Slurm logs are
logs/clust/rsrch-new-4b-v100-4373118.{out,err}.


### 2026-10-03 — cluster admission failure and Comet startup visibility

Job4373118 failed after1h21m41s during two-rank admission, before the main
20k run initialized. The one-rank native/branch checks passed at18.37695GiB
reserved, but the first two-rank admission update reached28.90430GiB on
rank0 and28.90234GiB on rank1 (91.0872% maximum). Loss0.957965 and gradients
were finite;32 B matrices updated. The required90% memory gate saved the
admission checkpoint and stopped the job. This was one diagnostic update,
not a main training update. No main Comet record existed because registration
was incorrectly delayed until after admission and native baseline preparation.

Cluster startup now closes a Comet archive before dataset verification.
Initialization inherits the setup experiment key. The login uploader retries
failed uploads, reports Slurm/stage status, and publishes separate live loss,
step and ETA curves without waiting for a complete2000-update segment.
An actual offline-SDK check confirmed setup-to-training key preservation;
the existing two-rank CPU gradient/replay regression also passed.

Replacement experiment key307b9627e32d4d38aed324c028f6eedf was created and
its name/parameters/status independently read through Comet's API:
https://www.comet.com/nikolay-2104/rsrch-new/307b9627e32d4d38aed324c028f6eedf
It is marked preparing, not training. The named run ends20261003_r2.
A bounded two-V100 probe (job4373281,128-row smoke) is checking expandable
allocator segments and release of temporary DDP broadcast buffers; it does
not replace full pretrained admission or the fixed96 panel. Peak allocated
as well as reserved memory is now recorded per rank. No precision, loss,
batch, target resolution or memory-gate relaxation is introduced.

The two-GPU probe4373281 completed0:0 in7m07s oncn-026. Both updates had
finite loss/gradients; all32 B matrices changed. Maximum reserved memory
was18.21484GiB (57.4011%), maximum allocated17.67429GiB. This is a128-row
smoke result; the full-data two-rank save/resume admission still runs before
production training. The startup/update heartbeat now also maintains Comet's
actual running/finished/crashed state.

Measured FP32 updates took72.8–81.1seconds;20k updates project roughly
17–19days before validation overhead. The replacement requests21days rather
than the insufficient original7, within the partition's30-day maximum.
This changes the scheduling ceiling, not the20k target or GPU count.

Replacement job4373291 started oncn-026 at2026-10-03 15:22:18UTC from
commit3d6e02d. Before submission,120 committed runtime/config/lock files
were SHA256-verified remotely. At the startup check it was verifying the
dataset with zero main updates. Its startup archive successfully uploaded
to Comet key307b9627e32d4d38aed324c028f6eedf; the activated-environment
login uploader PID2895963 reports state/stage and later live progress.
Submission and source manifest are under remote/local
scratch/clust-comet-recovery. Full-data admission and training remain pending.


### 2026-10-03 — BF16 review against the live GB10 pilot

The user rejected the long FP32 duration and requested the GB10-style approach.
Read-only inspection of Vast53994096 found the9B fast pilot atupdate1430,
mean5.28210seconds/update over the latest50, peak52.28516GiB. It uses native
BF16, effectivebatch1, GPU text encoding, and no gradient checkpointing.
V100/Volta has no native BF16 support (NVIDIA Tensor Core precision table:
https://www.nvidia.com/en-eu/data-center/tensorcore/). The earlier native
FP16 probe overflowed the text residual atdouble_blocks.4 before backward;
loss scaling alone does not fix that forward overflow.

Prepared, configuration-validated but NOT benchmarked/submitted:
configs/clust/flux4b_bf16_fast_proposed.yaml. It retains FLUX4B/768/r128,
20k updates and fixed96 validation; proposes one80GB BF16-capable GPU,
batch1/accum1, GPU encoding and checkpointingOFF.20k updates then process
20k examples instead of160k; this is not an equal-data speed comparison.
A100 and H100 scheduler-only checks passed under proj_1892/gpu-ef-quick;
live partition maximum remains3hours and preemption modeREQUEUE. Production
use needs measured admission and robust checkpoint continuation. The user's
GPU preference is pending; no A100/H100 allocation was submitted.

Cancelled owned FP32 job4373291 during dataset verification on the user's
speed/precision steering. Slurm confirmedCANCELLED at8m29s; no main metrics
file or production optimizer update existed. Existing logs/data are preserved.
The running GB10 pilot was inspected only and remains untouched.


### 2026-10-03 — V100 50-hour configuration budget (measurement pending)

The user now requires20k optimizer updates within50hours on2V100s, with
4V100s also considered.50hours/20k gives9seconds/update including all
overheads. The measured warmed2-rank FP32 batch8 update was72.7896seconds;
reducing accumulation4to1 projects18.1974seconds/update (~101.1hours of
training alone), not a measured batch2 result. Four data-parallel GPUs with
one example/rank still perform one full forward/backward per update; they
raise the effective batch to4 rather than inherently halving update latency.

Prepared bounded scripts/benchmark_clust_budget.py and
jobs/clust_v100_budget.sbatch. Named six-update/128-row timing probes put
the frozen FP32 encoder onGPU1 and the denoiser/VAE onGPU0, retain fresh
noise/timesteps, and compare768 and separately named512 resolution with
checkpointing on/off. They record component times, native/off parity,
finite gradients, updates to all64 branch tensors, per-device memory, and
fixed96 generation estimates at0/every2000 through20k. They are not complete
pretrained save/resume admission or scored image validation.
Slurm job4373489 is submitted with2V100s/8CPUs and a reduced30-minute cap.
No50-hour production configuration has yet been selected or launched.

### 2026-10-03 — 2k checkpoint preserved; native12 before continuation

Vast53994096 batch1 fast pilot completed2000 and scoring successfully.
Original owner-based ID_sim: native .38159112; BA0 .15439762;
BA1000 .42430972; BA2000 .43407627. Step2000 text_sim28.41344118.
These are measured fixed12 pilot results, not the full96 benchmark.

Downloaded `runs/flux9b_large_qkvo_r128_fast5h_b1/checkpoint-002000` plus
runtime snapshot and provenance locally. All checkpoint files are SHA256
verified against the remote originals; verification receipt is next to it.
Adapter weights, optimizer/scheduler/RNG/data cursor are retained.

The user requested native baseline validation first. Started supervisor
`rsrch_9b_baseline_then_8k` in isolated `/workspace/rsrch_9b_8k`: fresh native9B
(no BA/checkpoint), same12 and generation settings, original ID/CLIP scoring,
Comet `1acaa7707e184bad9176008e78f97ea6`, then full-state continuation to8000.
No other GPU job was present. Continuation code allows only training.steps
and validation_every to differ; frozen original code/config/data/model
hashes passed preflight. LR/rank/batch changes are rejected. Original runtime
and checkpoints remain unchanged; inherited0/1000/2000 panels are explicitly
marked as imports. New panels are4000/6000/8000. No commit/push performed.

Latest100 update mean5.2114s gives8h41m for6000 extra updates, approximately
9.5–10h including panels/loading, plus this requested baseline. Accountcredit
$3.5682 at$0.49829/h covers~7.16h as of16:56UTC. User authorized starting even
if balance is short; suggested$2–3 top-up. Actual baseline/continuation
completion status will be appended after startup verification.

Follow-up steering added a separate full96 native baseline before training.
Created Comet `6abec2cdaa104924a563e8b53f5a622c` with mode/native, BAfalse,
no checkpoint, trainingfalse and zero trainable parameters. Original96
manifest order/references/prompts/seeds remain unchanged; config768px,
20steps/CFG4/batch2. Queued after native12 and before8k, with separate outputs
and original ID_sim/CLIP scoring. Active shell's unread tail was extended
without interrupting its native12 worker; verified file offset and bash syntax.
The controller is supervised and all GPU stages share the existing lock.
Extra full96 validation adds to the prior continuation time/cost estimate.


### 2026-10-03 — Measured FP32 V100 budget and mixed-precision probe

Completed job `4373489` on `cn-026` (2 x V100-SXM2-32GB, activated
`rsrch_new`, exit 0:0, 10m20s). Each accepted case used six online updates
on a 128-row timing subset, effective batch 1, GPU0 denoiser/VAE and GPU1
frozen FP32 text encoder. Mean excludes the first update. No dataset cache
or native loss change was introduced. Raw measurements and exclusions are
in `reports/261003_clust_v100_budget/fp32_measurements.json`.

| FP32 case | Seconds/update | 20k training hours | Serial generation hours | Peak training GPU GiB |
| --- | ---: | ---: | ---: | ---: |
| 768, checkpointing on | 12.3145 | 68.41 | 28.80 | 17.98 |
| 768, checkpointing off | rejected at 90% memory ceiling | — | — | — |
| 512, checkpointing on | 6.9371 | 38.54 | 17.78 | 17.02 |
| 512, checkpointing off | 5.6179 | 31.21 | 17.78 | 27.17 |

Generation estimates include 96 native images plus fixed96 at 0/every 2000
through 20000 (1152 images total), 20 steps/CFG4. They use three warmed CFG
pairs on the largest reference grid and a full target mask; this is a
conservative workload estimate, not completed image validation or a hard
upper bound. Startup, full-data reads, admission, checkpointing, decoding,
and scoring are additional. Therefore even 512 without checkpointing has
insufficient margin for a 50-hour end-to-end budget. All accepted FP32
cases passed exact native/off parity, finite gradients and all 64 adapter
tensor updates. These are timing probes, not full save/resume admission.

Follow-up `4373563` tested FP16 autocast with FP32 residual inputs. Six
updates had finite gradients at approximately 3.6 seconds/update, but the
all-64-tensors-changed check failed. The candidate was rejected; the job
exited 1:0 after 1m18s. The next probe (`4373571`) also keeps the entire
trainable branch calculation in FP32, retains FP32 master weights and
optimizer state, disables autocast weight caching, and uses gradient
scaling. It tests 100 checkpointed 768px updates, a six-update comparison
without checkpointing, and separately named 384px FP32 fallbacks. It does
not modify the production backend or start a 20k run.


Job `4373571` completed 0:0 in 10m06s. The revised mixed-precision 768px
checkpointed case passed 100 updates through the full LR warmup: mean
3.61908 seconds/update, finite gradients, all 64 adapter tensors changed,
and exact native/BA-off parity under the same precision policy. Peak CUDA
reserved memory was 17.53125 GiB on the training GPU (55.25%) and 15.53125
GiB on the encoder GPU (48.94%). One native prediction differed from FP32
by relative RMSE 0.002490; that is a numerical comparison, not image quality.

The measured projection is 20.106 hours for 20k updates plus 8.609 hours
for serial native/fixed96 image generation. Recommend a provisional
40–45-hour total budget, allowing about 11–16 hours for full-data reads,
admission, loading, checkpoints, decoding/scoring and variation. Those
additional costs have not been measured end-to-end; queue wait is excluded.
The same mixed-precision case without checkpointing exceeded the 90% memory
ceiling and is rejected. FP32 384px without checkpointing measured 4.36251
seconds/update (24.24 training hours plus 14.71 generation hours), but changes
resolution and leaves substantially less overhead margin. It is not the
preferred proposal. FP32 384px with checkpointing needs 30.42 + 14.74 hours
before overheads. Complete results: `reports/261003_clust_v100_budget/mixed_measurements.json`.

The concrete proposal is `reports/261003_clust_v100_budget/recommended_setup.json`:
FLUX4B/768/reference512/QKVO rank128, one training worker, microbatch 1,
accumulation 1, LR 5e-5, warmup 100, checkpointing on, fresh online
conditioning, dedicated FP32 GPU encoder, and the original serial fixed96
schedule at 0/every 2000. FP16 autocast is restricted to the frozen backbone;
residuals, modulation inputs, trainable branch, master weights, optimizer,
encoder and VAE stay FP32. Gradient scaling starts at 32. This processes
20k examples rather than the original batch8 run's 160k. Four V100s could
host two such replicas for effective batch 2, but that is an unimplemented,
unbenchmarked extension and is not assumed to halve update time.

Only benchmark files implement this precision policy so far. Before a long
run, integrate it into the shared training/validation path, support explicit
GPU1 encoder placement, pass full-data/largest-layout and fresh-process
save/resume admission (including scaler and both CUDA RNG states), and
regenerate/score the matching native masks and step-0 panel. Production has
not started. Benchmark source hash was
`ec5e09691d59ccbbfeac6db071df7ad9fac920ab1038b576981ea65b2b3d9f8d`;
a copy is retained under remote `logs/clust/benchmark-4373571.py` and local
`scratch/clust-v100/budgets/4373571`. Later script edits only clarify its
docstring. All GPU work ran inside the activated cluster `rsrch_new` environment.

### 2026-10-03 17:55UTC — native baseline visibility verified

User could not see live validation/training in Comet. Supervisor and GPU worker
are running the native96 panel;44/96 generated at inspection. Native12 finished
successfully: ID_sim.3815911189, text_sim28.495046. Comet API confirmed its
12 image assets and ID/CLIP metrics. Native96 had only config assets because
images upload during the subsequent VAE decode stage, and the initial Comet
session had ended during generation. Training is still queued after scoring.

Added `scripts/log_native_validation_progress.py` as a separate CPU-only
supervised logger, without editing frozen training/inference sources. It keeps
the existing native96 Comet session live, reports count/percent/generationETA
once/minute, and ends when scored or if its controller stops. First report:
44/96 (45.83%), generationETA2400sec; decoding/scoring are additional.
Supervisor `rsrch_native_progress` on53994096. No GPU job duplication.


### 2026-10-03 — Approved two-V100 mixed-precision production launch

The user approved launching the measured 768px / rank128 / effective-batch1
mixed-precision setup for 20k updates. Added the named production config
`configs/clust/flux4b_2v100_amp.yaml` and `jobs/flux4b_clust_2v100_amp.sbatch`
(2 V100s, one training worker plus dedicated GPU1 encoder, 8 CPUs, 50-hour
wall limit, same serial fixed96 at 0/every 2000).

`ba_dit/precision.py` applies the benchmark's FP16 autocast with FP32 block
residual inputs and branch math to the shared native/BA training and inference
backend. FP32 master weights, encoder, VAE and optimizer remain unchanged.
Checkpoint and baseline-mask identities include the new precision policy.
Training checkpoints include scaler state and both GPU RNG states; finite
gradient overflow retries are bounded and replay the same data/RNG state at
a reduced loss scale without counting a skipped optimizer update. Memory
metrics include both GPUs. Native admission also stresses the largest real
reference grid with full routing masks (a separately recorded memory probe).

Adopted the existing geometry-only mask calculation and lazy mask construction
for the cluster path to avoid eager full-image decoding/mask materialization
before every segment. Target/reference image transforms and mask values are
unchanged. Initialization records source face boxes and image hashes rather
than eagerly computing all mask coverage summaries.

Verification before submission: five focused CPU tests passed (including
scaler/optimizer/RNG replay, precision identity/mask separation, and native
branch ownership), configuration/dry-run checks passed, Python compiled and
both shell launchers passed `bash -n`. These CPU checks do not substitute
for pretrained GPU admission. Full-data native/off/zero-mask parity, finite
updates/frozen weights, largest-layout memory and fresh-process exact replay
run as mandatory gates in the cluster job before production. Unrelated
workstation/Vast changes are preserved and excluded from the deployed commit.


Production submission: Slurm `4373673`, `proj_1892/rocky`, 2 V100s, 8 CPUs,
50-hour limit, submitted 2026-10-03 18:05 UTC. Source/config/lock audit passed
for 64 files; initial implementation commit `3e10d35`, queued-Comet handoff
commit `ff2829c` deployed and verified before allocation. Submission and
source receipts are under local `scratch/clust-production` and remote
`scratch/clust-v100`. Script: `jobs/flux4b_clust_2v100_amp.sbatch`.
Run: `/home/nasilaev/rsrch_new/runs/flux4b_clust_2v100_amp_qkvo_r128_20k_20261003`.

Registered immutable Comet key `5d31de48010446248639e65a65236cbe` while queued:
https://www.comet.com/nikolay-2104/rsrch-new/5d31de48010446248639e65a65236cbe
The compute controller verifies the reserved configuration hash and inherits
this same key at startup. Login uploader PID374294 runs in the activated
`rsrch_new` environment; its successful API heartbeat reports stage `queued`
and Slurm `PENDING`. No production optimizer updates or pretrained admission
have completed yet. Last scheduler estimate was 2026-10-03 21:53:42 MSK
(19:53:42 London), node cn-005, reason Priority; estimates can change. The
job automatically runs full admission and native/step-0 validation after
allocation, before the 20k training sequence. No recurring notification
or automatic job-resubmission workflow was created.


## 2026-10-03: V100 startup failure and environment-only resume

Status investigation found production Slurm job `4373673` failed at
2026-10-03 19:54:53 UTC after 1:32:42 on cn-004. Production optimizer progress
was still 0/20,000. It completed full pretrained admission and native fixed96
generation, then `masks_auto` failed importing InsightFace: system
`/lib64/libstdc++.so.6` lacked `GLIBCXX_3.4.32`. This was a metric-environment
runtime failure, not a numerical or GPU-memory failure.

Admission measured native/BA-off and zero-mask exact parity, unchanged frozen
weights, finite gradients and changes in all 64 trainable tensors. Fresh-process
checkpoint replay matched parameters, optimizer/scheduler/RNG/cursor and gradient
scaler exactly (two CUDA RNG devices). The largest real reference grid with full
routing masks passed finite gradients. Peak reserved memory was 17.96484375 GiB
on GPU0 (56.61%) and 15.53125 GiB on GPU1 (48.94%). These are admission measurements,
not evidence of completed production training.

The existing rsrch_new Conda environment contains the required C++ runtime.
`scripts/activate_clust_env.sh` now exports its library directory for both isolated
metric environments, and disables Albumentations' online version check. Both
InsightFace extensions resolve that environment's library with no missing symbols.
No model/config/data/validation changes were made; all 25 immutable setup source
hashes still match. `scripts/upload_clust_comet.py` now falls back to accounting
when a completed job disappears from squeue, uses a single-uploader lock per job,
and labels a pending resumed allocation as queued instead of displaying the old
failed setup stage. The old job's FAILED/masks_auto state was successfully sent
to its existing Comet experiment.

Submitted an explicit resume as Slurm `4374072` at 20:33:20 UTC with the original
2-V100/8-CPU/50-hour allocation, same run directory and immutable Comet key
`5d31de48010446248639e65a65236cbe`. Completed admission/cache/native receipts are
reused; the original failed logs remain. Submission receipt is
`scratch/clust-production/amp_resume_submission.json` locally and
`scratch/clust-v100/amp_resume_submission.json` remotely. A separate CPU-only
20-minute test job `4374073` checks both metric environments and prepares the
native masks while the GPU resume is queued. This is a targeted recovery, not
an automatic resubmission loop.

Verification: shell/Python syntax and whitespace checks passed. CPU job
`4374073` successfully imported InsightFace/CLIP and InsightFace/PyIQA in their
respective isolated environments; native mask preparation then started normally.
The Comet uploader successfully published `PENDING/queued` for resumed job
`4374072` under the same experiment. Slurm estimated 2026-10-03 21:33:10 UTC
(22:33:10 London) at the last queue check; this is a mutable scheduler estimate.

CPU verification job `4374073` completed successfully (`0:0`, 7m35s): both
metric environments imported, and the previously failing native mask pass
prepared 96/96 usable masks with no review required. It used the unchanged
native fixed96 images and metric definitions; no manual mask overrides were
needed. GPU resume `4374072` remains PENDING/Priority, estimated 21:33:10 UTC
(22:33:10 London). Production progress remains 0/20,000 until initial validation
and scoring finish after allocation. The detached login uploader (PID2091453,
login-02) published this queued state to the existing Comet experiment.


## 2026-10-03: Comet liveness and upload isolation

At 22:03 UTC the user correctly reported Comet was no longer running. Direct
Comet API metadata confirmed `running=false`, while Slurm training job
`4374072` was RUNNING and generating step-0 fixed96 images. The detached login
uploader PID2091453 no longer existed. Its last log showed a stalled 15.53 MiB
asset upload followed by a timeout; the exact reason the parent process exited
was not established. Earlier one-time running checks did not establish ongoing
liveness. The old uploader also slept 30 seconds and blocked on uploads, whereas
Comet actually returned a required heartbeat interval of 10,000 ms.

`scripts/upload_clust_comet.py` now sends acknowledged heartbeats on a separate
API connection/thread at half the server interval (at most five seconds),
records acknowledgement timestamps, and polls archive uploads without blocking
status/metrics. Upload subprocesses have a 180-second limit and retry backoff;
unfinished archives remain intact and final pending uploads are recorded.
Step-0 validation image counts and the genuine zero optimizer-step count are
published as live metrics. API clients bypass metadata caching. The logger
stops with the target job and records its final scheduler state.

Added `jobs/clust_comet.sbatch` so the logger is managed by Slurm independently
of SSH sessions (one CPU, zero GPUs, 50-hour upper bound). Submitted logger
job `4374270` for existing training job `4374072`; its submission receipt is
`scratch/clust-v100/comet-service-4374072.json` on cluster. The training process,
its immutable source/config identity, checkpoint and Comet key are unchanged.
Three focused tests passed locally and inside rsrch_new on cluster: expired-job
accounting, nonblocking uploads, and server-paced heartbeat acknowledgements.
Shell/Python syntax and Slurm submission validation passed. An allocated
compute-node probe successfully reached Comet HTTPS and its authenticated API.


Logger verification: companion job `4374270` started on cn-037 and remained
RUNNING across an explicit SSH logout/reconnection. Acknowledged heartbeat
count increased from 17 to 34 across that check, using the same PID and Slurm
job; the latest acknowledgement was under five seconds old. Comet independently
reported `running=true`, `hasCrashed=false`, `archived=false`, and live validation
progress increased from 71 to 78 of 96. Production optimizer steps remain zero
during step-0 validation. Large asset uploads still stall on the cluster route;
those retries are now independent of heartbeat/status/progress publishing and
archives remain queued. Logging fix committed and pushed as `7cf102c`.

### 2026-10-03 — automatic 8k completion, full96, Comet consolidation and stop

User explicitly requested this sequence and an extra pre-stop agent check-in.
Installed `scripts/finalize_flux9b_remote.py` in frozen9B checkout (new file;
existing training sources untouched) and prepared its original96/native-mask
bundle. Remote supervisor rsrch_9b_final96 waits for completed8000/fixed12 and
verified local checkpoint copy, then serially runs trained full96 and scoring.
Installed local `scripts/finalize_flux9b_local.py` under enabled systemd user
service rsrch-flux9b-finish. It copies/verifies full checkpoint+results, imports
continuation histories and separately prefixed full96 metrics/images into
original2k Comet, verifies server state, then stops53994096. It never terminates.

At22:22UTC step4359, latest100 mean5.11446s, measured fixed12 panel780.90s.
ETA8000+fixed12:04:04UTC4October; all work including full96/transfers/Comet:
05:54UTC (~7h32m). Credit$5.9364 versus totalhourly$0.49829. First check-in
04:19UTC; stop floor05:00UTC, still gated on successful artifacts/uploads.
Syntax/unit checks and negative stop-gate checks passed; remote/native96
preparation and live read-only local probe passed. Both waiting services are
running. Actual final artifacts/stop remain pending. No commit/push performed.
Detailed scope, paths, receipts and recovery: FLUX9B_8K_COMPLETION.md.


## 2026-10-04: Recover Comet artifacts and record cluster cancellation

The user reported missing outputs/metrics. Training job 4374072 was genuinely
advancing (finite losses/gradients, about 3.67 seconds/update), but Comet only had
live/* curves while canonical train/* history remained offline. Large asset
uploads from the cluster repeatedly received HTTP 502. An open offline archive
was also a valid empty ZIP without experiment.json/messages.json, so is_zipfile
alone incorrectly accepted it for upload.

Added scripts/sync_clust_comet.py: a workstation publisher selectively rsyncs
completed fixed96 outputs and metrics through the existing Russia VPN/clust
route, resumes the immutable Comet experiment, backfills canonical train/*
metrics, and verifies server-side loss values and image assets before recording
success. It excludes checkpoints, caches, raw images and credentials. Images
retain panel step, prompt and seed; no training/scientific settings changed.
The cluster logger supports --status-only and writes an atomic Slurm status
receipt, leaving image delivery to the workstation. Archive readiness now
requires both SDK metadata members. API summary.stepCurrent reflects latest
arrival, not necessarily maximum optimizer step; verification checks the actual
loss history rather than this summary field.

Replaced CPU-only logger 4374270 with status-only logger 4374404. Training source
identity remained unchanged (all 25 setup hashes matched). The workstation relay
ran under transient systemd unit rsrch-clust-comet-4374072 and successfully
verified consecutive cycles through steps 673, 694, 702. Comet API independently
confirmed 702 contiguous finite train/loss values and 109 images: 96 fixed96
step 0 generations, 12 paired comparison sheets and one mask overview. These
are initial/untrained validation outputs; no trained2,000-step panel exists.
The immutable experiment remains 5d31de48010446248639e65a65236cbe under
nikolay-2104/rsrch-new. CUDA peak reserved was 17.9043 GiB (56.42%) for denoising
and 15.53125 GiB (48.94%) for the encoder.

During this verification Slurm cancelled training at 2026-10-04 00:03:17 UTC
(01:03:17 London, 03:03:17 MSK), after 3h 01m 56s and 702 updates. sacct explicitly
records CANCELLED by 0; scontrol supplies Reason=None. No Python exception or
OOM appears in the training log. The administrative/system reason is unknown;
cluster preemption is configured, but that does not establish this job's cause.
Checkpoint 000500 is intact (adapters plus optimizer/scaler/RNG state); updates
501–702 were logged but were not checkpointed. No automatic resubmission was
performed after this root/system cancellation. Logger 4374404 completed and the
local relay exited after publishing the terminal status. Comet API now reports
running=false, hasCrashed=true, archived=false, accurately reflecting interruption.

Verification: four focused uploader regressions, Python syntax, shell syntax,
whitespace checks, real rsync/publication, and independent Comet API history and
asset readback passed. The transient relay survives terminal closure while
running but does not restart across a WSL reboot. For a future authorized
resume, start a status logger and relay for the new Slurm job ID; retained Comet
history beyond the checkpoint must be handled explicitly to avoid treating
repeated optimizer-step numbers as new distinct updates.

### 2026-10-04 04:22UTC — 8k complete, full96 in progress

Scheduled check-in verified completion of8000 and fixed12 scoring at04:02UTC.
Full8000 checkpoint downloaded04:03UTC and rehashed locally; all64 BA tensors
changed from2k and are finite. Adam step/cursor8000, finite optimizer moments,
all RNG states present. Runtime/native mask hashes and active full96 checkpoint
hash match. Full96 is20/96 with one GPU worker; original canonical Comet merge
and automatic stop remain pending behind their existing completion gates.

Measured fixed12 ID_sim declines after2k:2k.43407627,4k.42768707,
6k.41867339,8k.41646561 (native.38159112).2k remains best. Continuation
Comet API confirms8k metrics and12 images+2 comparison sheets. Estimated
remaining work finishes06:30–06:45BST. Credit$2.88 sufficient, local5.8GB free.
Only repair: corrected local controller's misleading waiting_for_training
status after training completion; restarted that CPU service, leaving GPU
validation untouched. Historical intermittent SSH errors recovered automatically.


## 2026-10-04: Requested checkpoint-500 resume and cancellation investigation

The user explicitly requested investigation and continuation. Slurm accounting
still records job 4374072 as CANCELLED by 0 at 00:03:17 UTC, with batch exit 0:15
(SIGTERM), Reason=None, and empty Comment/AdminComment/SystemComment. Runtime
was 3h 01m 56s against a 50-hour limit. The rocky partition has PreemptMode=OFF;
there was no cn-006 node event in the cancellation window and no other root
cancellation in that window. The account/project and quotas remain valid.
The account has no local notification mailbox, and the controller log lives on
the scheduler host, unavailable from the login account. Thus a root/system-issued
cancellation is established; the exact administrator or automated-policy trigger
is not exposed. Do not call this ordinary queue preemption, a training exception,
an OOM, or a timeout. No support message was sent.

All 25 immutable training source hashes still match. Backed up checkpoint 000500
locally under scratch/clust-resume-20261004, verified every file against its
remote SHA256, and loaded it on CPU: 64 finite adapter tensors, 64 finite optimizer
states at step 500, cursor 500, scheduler step 500, both CUDA RNG states and loss
scale 32 are intact. The unchanged trainer restores model, optimizer, scheduler,
scaler and RNG; the full fixed96 panel and 20,000-update endpoint are retained.

Preserved the complete interrupted metric file and separate uncheckpointed
steps 501–702 before atomically returning metrics.jsonl to steps 1–500. Submitted
one explicit resume as job 4374962 at 08:28:08 UTC using the existing AMP sbatch
with --resume, two V100s, 8 CPUs, proj_1892/rocky. The 36-hour allocation covers
approximately 20 hours of remaining updates plus serial validation and margin,
within the user's 50-hour budget. Automatic requeue remains disabled. Resume
receipt: remote scratch/clust-v100/resume-4374962.json, mirrored locally under
scratch/clust-resume-20261004; it pins checkpoint hashes and the exact command.

Comet retains immutable key 5d31de48010446248639e65a65236cbe. The relay now accepts
an attempt-specific metric prefix so replayed steps do not merge with abandoned
points or get skipped as already published. The active curve is
resume_4374962/train/loss; its verified history starts at 1 and includes checkpoint
500. Original train/loss and discarded-step evidence remain available. Chart
metrics publish every 2 updates (plus step 1), keeping 20k training below Comet's
published 15k-values-per-metric limit; metrics.jsonl retains every update. Source:
https://www.comet.com/docs/v2/guides/experiment-management/limits-and-performance/.
The relay also uploads resume receipts and discarded-step history as assets.

CPU-only status logger 4374964 is RUNNING on cn-016. Local systemd relay
rsrch-clust-comet-4374962 is active, with repeated successful server verification
of 251 loss points through 500 and 109 existing images. Both stop with the target
job and have a 7-day upper bound to include queue time; neither resubmits jobs.
Five focused tests passed, including isolation of replayed metrics from the
cancelled attempt, plus syntax and whitespace checks. No training code changed.
At 08:31 UTC training remained PENDING/Priority; the scheduler estimated
09:19:45 UTC (10:19:45 London) today. This is a mutable estimate, not a start claim.

### 2026-10-04 — rename the current BA setup to FLUX1 and preserve deployments

User requested FLUX1 naming, explicit Vast/cluster entry points, and commit/push.
Renamed the8 executable masked-Q/K/V/O configs and the cluster proposal file;
updated launcher/test/config links. Model/data/optimizer/sampling fields are
unchanged. Kept historical run names/configs and source identities intact; the
one-ID validation schedule remains0/500/1000/2000 under the new display name.
README separates Vast9B/BF16, cluster4B/selectiveFP16 with encoderGPU, local4B
one-ID and the4B Vast alternative. Earlier rank16 and cached-head experiments
remain explicitly historical. Regenerated26-page FLUX1 report plus whole-model,
BA and inference vector diagrams; source/hash/metric/layout checks passed.

Captured actual machine sources read-only and verified against immutable run
identities (23 Vast and26 cluster source hashes). Archived runtime Git branches
preserve those exact files and original configs; main contains current maintained
code and FLUX1 fresh-launch naming. No training was launched or changed by naming.

Also repaired the pending8k completion controller's Comet image-name comparison:
SDK PNG/duplicate suffixes had made already-uploaded images appear missing.
Using normalized names and uncached API reads verified90048 metric points and
138 logical image/step pairs. Duplicate uploads already created are retained.
The authorized controller then stopped Vast53994096; actualstateexited confirmed
08:38:26UTC, and its heartbeat was paused. Local full checkpoint/results remain
verified; no termination occurred.

### 2026-10-04 — FLUX1 architecture and completed 9B results review

Research report: `reports/261004_FLUX1_review/FLUX1_research_report.pdf`
(linked HTML, reproducible CPU analysis, per-image CSV, source hashes and visual
appendix in the same directory). The initial artifact audit used CPU analysis
and read-only Comet retrieval. The subsequent user-requested 2k full96 replay
is recorded below; no training, host lifecycle operation, commit or push occurred.

The completed full96 comparison is almost neutral: native owner-matched ID_sim
**0.27418510**, FLUX1 8k **0.27621790**, paired change **+0.00203281**;
47 images improve and 49 regress. Only Jensen and Keanu improve on their
12-prompt identity means. A descriptive bootstrap over the eight identity
groups gives a 95% interval of **[-0.02233, +0.03245]**. CLIP changes from
29.44020426 to 29.36709690. No missing/unowned faces are reported in either lane.

The fixed12 pilot falls from 0.43407627 at 2k to 0.41646561 at 8k, with
5 gains / 7 regressions and identity-cluster 95% interval **[-0.06363, +0.02596]**
for the paired change. Its rows cover only reading and angry-traffic prompts,
with unequal identity weights and seed0. Within the separate full96 execution,
the pilot rows gain +0.03075 and the other84 lose 0.00207. Pilot/full96
overlapping outputs are not bit-identical; comparisons remain within each run.
The saved 2k adapter has not been scored on full96, so its broad superiority
and a causal diagnosis of overfitting remain unestablished.

All 8,000 logged updates are finite. Late loss continues to fall; peak reserved
memory is **52.285 GiB (42.99% of reported capacity)**. All64 BA tensors change
from 2k to 8k and remain finite, with33,554,432 trainable parameters. The run
has seen8,000 of47,341 pairs (16.9% of its first shuffled traversal), rather
than multiple dataset epochs. Full96 inference recorded20.373 GiB reserved
and exact exterior pixels. Those pixels are preserved by composition, not
evidence that locality was learned.

CPU reference re-embedding matches all eight `subject_v2` identity prototypes
at cosine approximately1.0. The older legacy-best Eddie prototype does not
match the intended reference face (cosine -0.0078); the current owner-matched
`id_sim` uses the corrected prototype. Reference-face supports contain63–252
tokens, below the512-key cap. Full96 image hashes, current prototype hashes,
parent source snapshot and8k checkpoint receipts were checked.

Prioritized proposals: score2k on the unchanged full96 bundle; add a BA-off
second-pass and branch-only reference-swap control; test matched lower-LR
continuation, identity supervision, identity K/V tokens inside the existing
reference read, and higher-detail reference-face tokens separately. The user
requested preserving native attention and refraining from native/reference
output blending. The report follows that constraint; residual output fusion
is not a recommended follow-up. HSE-cluster4B and the earlier local4B one-ID
run remain distinct from Vast9B evidence.

### 2026-10-04 — requested FLUX1 Vast9B 2k fixed96 replay

The user restarted GB10 instance53994096 and requested evaluation of the saved
2k checkpoint, with results in the existing 8k full96 Comet run. New supervised
job `rsrch_9b_2k_fixed96` uses `/workspace/rsrch_9b_8k` and writes
`runs/FLUX1_vast9B_2000_fixed96_20261004`. It reuses the exact completed8k
native PNGs/latents, masks, owner boxes, fixed96 manifest/order, prompts,
references, seeds,768px/20steps/CFG4 and batch2 grouping. Only the descriptive
config name changes; all frozen inference/scoring source hashes passed.
The2k adapter SHA256 is
`eb27ca1c63780a50ae090cc65c746b6a7225ea66e776400f1b5bd131606857fd`.

`scripts/validate_flux1_2k_fixed96.py` runs inference, decode and scoring
serially under the shared GPU lock. It logs96 images and `validation/*`
metrics at step2000 into existing Comet
`2b3eda1b1f364ecfaa584ffe273102a1`, verifies read-back, preserves its8k metric
points, audits exterior pixels and writes a completion receipt. Startup
provenance is already uploaded; first4/96 latents completed in175 seconds.
This is progress evidence, not a completed quality result.

The one-off local service `rsrch-flux1-2k-collect` runs
`scripts/collect_flux1_2k_fixed96.py`: after remote completion it downloads
and verifies every result hash, then mirrors full96 at step2000 into original
training Comet `cfbd6f879e3d4158959bbbef7fb8e894` under `validation96/*` and
`fixed96_2k/*`, matching the previous8k consolidation. It preserves the
existing pilot12 `validation/*` values and verifies the uploads. Neither
controller starts training, stops the machine, or changes old run files.

### 2026-10-04 — TaskMaster idle-resource cancellation and two-worker repair

The user's TaskMaster notice identifies job4374072 and00:03–03:03MSK as the
idle-resource interval. Slurm confirms `CANCELLED by 0`, start00:01:21 and
end03:03:17MSK, allocation2V100/8CPU. It did not time out or report CUDA OOM.
The allocation was inefficient by construction: `world_size: 1` trained only
on GPU0; GPU1 held the frozen encoder (measured0.33555s out of3.619s/update,
about 9.3% duty). The same allocation retained both GPUs for serial inference,
CPU ID/CLIP/face-quality scoring and repeated full47,341-pair image hashing.
Original receipts show inference0 ended01:23:13MSK, decode01:26:03, ID/CLIP
01:37:54, face-quality 02:00:34 and summarize 02:00:50. Thus the first two hours
already contained substantial unused allocated GPU capacity. These observations
explain the resource mismatch; the site's precise TaskMaster thresholds are not
known. No monitoring threshold or activity spoofing is changed.

The resumed old allocation4374962 reproduced the problem: ~33 minutes from
startup to its first logged updates; sampled GPU0 was100%/18,778MiB/175.54W,
GPU1 was0%/16,308MiB/69.26W. It was stopped by our one-time watcher at13:06:43MSK
only after atomic checkpoint 1000 completed. Parent files remain intact.
The original cancelled attempt's last durable checkpoint was 500 (702 updates
had been logged); the second attempt successfully restored it and reached 1000.

User explicitly requested useful work on **both GPUs**. The isolated runtime
`/home/nasilaev/rsrch_new_staged` preserves frozen cluster source from archive
`FLUX1-cluster-runtime`8da85381b1c1f90ac086a269805267e7b932a7d1 and adds:

- Two DDP workers with separate samples and averaged gradients; microbatch 1,
  accumulation1, global batch2. This is a recorded batch/world-size transition
  at step 1000, not bitwise continuation of the old batch 1 trajectory. Preserve
  the optimizer/scaler, rank0 RNG and global data cursor; start rank1 at seed143.
- Correct mixed-precision GradScaler creation and persistence in DDP. The old
  DDP code inferred scaling only from master dtype and omitted scaler from its
  mixed-precision save call. Native FP16/FP32 policy and branch math are unchanged.
- Explicit rank CUDA index for cached tensors. Safetensors interprets bare
  `cuda` as GPU0; the previous cached loader could feed rank1 on the wrong GPU.
- CPU-only full-data verification once, then verified normalized manifest input
  to training; one-GPU frozen conditioning for only the next <=4,000 pairs;
  two-GPU training; one-GPU fixed96 inference/decode; zero-GPU CPU scoring.
  Consumed continuation-owned caches are deleted after that window's validation.
  A100GiB window cap/free-space check avoids the ~689GiB full text cache.
- Strict admission: bit-exact fixed 96 conditioning against parent; two pretrained
  DDP updates, all 64 adapters changed, finite gradients, each GPU below 90%
  reserved memory; exact full-state replay in fresh processes before production.

The bounded `afterok` Slurm chain is recorded in
`scratch/pipeline_FLUX1_cluster_4b_ddp_20261004.json` on cluster. CPU preparation
4375079, one-GPU probe4375080, two-GPU admission4375081,
first cache4375082 and first training4375083 precede the remaining2k windows;
final summary4375151. Each successor cancels on failed dependency. No automatic
resubmission. The new run is `runs/FLUX1_cluster_4b_ddp_20261004`; its generated
resolved config and `execution_transition.json` record the source/data/checkpoint
identities. The old encoder-GPU launcher is disabled in main; the archive retains
its historical version. Operations: `docs/FLUX1_CLUSTER_DDP.md`.

Comet key remains5d31de48010446248639e65a65236cbe under rsrch_new. Workstation
service `rsrch-clust-ddp-comet-4375151` publishes scheduler stage/worker ID and
`ddp_4375151/*` metrics without reserving a Slurm logging allocation. Historical
steps 1–1000 in that curve are explicitly batch 1. API read-back at 10:17UTC
confirmed `running: true`, stage`prepare_1000`, worker4375079. This indicates
preparation, not completed DDP training. Fixed96 validation remains0/every2,000.

Local checks: ten config/scaler/checkpoint/relay tests passed; the additional
rank-device regression passed; the existing real two-process CPU/Gloo check
passed averaged-gradient, distinct-sample and exact fresh-process resume checks.
Shell syntax and diff whitespace checks passed. At 10:20UTC cluster CPU data
verification was still running (7.67GB read); pretrained DDP admission and
production throughput were not yet measured. Further results follow below.

### 2026-10-04 — FLUX1 shortcut audit and three prepared Vast9B follow-ups

AICODE-NOTE: The user requested checking native-to-reference shortcuts and
three prioritized recipes, preserving native attention without output fusion.
No new training was launched. Historical Vast and cluster checkouts were not
modified for these recipes; all results below are local audits unless stated.

The current joint reference stream is target-dependent. A CPU random-weight,
full-depth Klein9B topology (width16) changes reference Q/K/V at all eight sites
when only target tokens change; mean absolute change is0.004365 at double2 and
0.010972 at single22. These are structural measurements, not pretrained9B
activation magnitudes or proof of exploitation by the trained checkpoint.
The independent reference-image-only bank has exactly zero K/V change for the
same perturbation. A donor-bank swap changes the target output while the native
inputs remain byte-identical. Detaching the old joint tensors would not remove
their target information.

All96 saved native/8k face pairs were matched to their original owner boxes:
native-to-trained ArcFace cosine mean0.570848, median0.590212, range0.234137–
0.802313. This is distinct from target-ID similarity (native0.274185 versus
8k0.276218). The saved mask geometry puts25.5122% of face-box pixels in
fractional latent cells; the mean native-clean coefficient there, averaged
over all face-box pixels at sigma0, is0.0255127. The final pixel compositor's
native coefficient inside the face rectangle is exactly0. These values do
not estimate causal identity influence.

Prepared fresh-run configs and shell entry points, all on Vast9B BF16:

1. **FLUX1a, recommended first:** isolated reference bank, binary token support,
   rank128/alpha128 Q/K/V/output BA (33,554,432 parameters), identity auxiliary
   weight0.05 at sigma<=0.5 plus unchanged native flow objective over that mask.
2. **FLUX1b:** same architecture/ownership/rank, flow-only objective; measures
   the contribution of identity supervision.
3. **FLUX1c:** FLUX1a with rank256/alpha256 (67,108,864 parameters); tests capacity.

All three preserve the native attention operations and conditioning. The
separate frozen bank uses only original reference-image tokens, original RoPE
coordinates and current sigma, with zero text/target tokens; no hidden features
from the live joint stream enter it. Native target queries, residuals, MLPs and
unmodified attention sites remain. Every latent token touched by the original
pixel mask is entirely BA-owned; only final exterior pixel feathering remains.
Proposed training: 4,000 updates, seed142, full pinned Large order, microbatch1,
lr5e-5/warmup100, saves every500, original fixed96 at0/2000/4000 with frozen
native bundle,20 steps/CFG4/768px/reference512 and original ID/CLIP definitions.

Identity labels are prepared only from training targets with frozen buffalo_l
landmarks/embeddings and hashed geometry/code/weights; rejected labels retain
their flow rows. The differentiable auxiliary checkpoints the frozen VAE decode
and uses deterministic target-landmark alignment. It reconstructs the clean
estimate with the actual FP32 noising coefficient while retaining native BF16
conditioning rounding. The new diagnostic `scripts/probe_flux1_reference.py`
has BA-off/own-bank/donor-bank arms on a separately named eight-image panel.

Local verification: six new information-flow tests, seven existing FLUX core
invariants and five additional masked-routing/alignment/identity-gradient tests
passed (18 total). Two real training targets passed CPU label preparation;
real differentiable ArcFace matched ONNX within3.34e-6. A surrogate decoder
produced a finite nonzero prediction gradient0.00455; this was not pretrained
VAE/FLUX admission. All three plan commands, Python/shell syntax and whitespace
checks passed. Admission additionally requires actual pretrained parity,
identity/flow gradient ratio, all64 branch updates, largest-layout memory below
85%, and exact fresh-process save/resume. New GPU memory, throughput and image
quality are **unmeasured**; no fit or improvement claim is made.

Entry points: `scripts/run_FLUX1a.sh`, b/c counterparts, and
`scripts/run_flux1_experiment.py`. Operational setup is appended to
`docs/FLUX1_DEPLOYMENTS.md`. Report:
`reports/261004_FLUX1_review/FLUX1_shortcut_audit_and_next_experiments.pdf`
(7 pages), with HTML, vector chart and JSON evidence/source hashes alongside.
Image-only bank distribution shift, decoder cost, spatial-boundary changes and
shared-recognizer overfitting remain explicit experimental risks. FLUX1a is a
highest-potential hypothesis, not a measured winner.

Follow-up results for this repair: preparation4375079 completed with exit0:0
in26m02s, using **one CPU and zero GPUs**. It verified all data against the
parent checkpoint, preserved28 run-source files, migrated full state at1000 and
estimated a66.377GiB maximum cache window. The one-V100 conditioning check
4375080 completed in4m14s with exit0:0: **all96 encoder tensors and all96 VAE
entries are bit-identical to the parent**, including unchanged native geometry.
Its `conditioning_parity.json` records this admission. Source/config/transition
provenance was downloaded to `scratch/clust_ddp_audit_20261004/`.

Comet read-back verified the full inherited history through1000 (501 chart
points at the declared interval) and109 step-zero/mask/paired images. The relay
publishes continuation batch2/start1000 parameters separately from historical
batch1 parameters. Its current service remains active.

As of10:42UTC, two-V100 admission4375081 is **PENDING (Priority)**; Slurm predicts
14:46:49MSK /12:46:49London, an estimate rather than a reservation guarantee.
The scheduler-only20-minute alternative did not improve that estimate, so the
original30-minute allocation is unchanged. A proposed move of probe4375080 to
`test` was rejected because it had already started in `rocky`; the recorded
scheduler attempt explicitly says not applied. No duplicate jobs were submitted.
**Real DDP replay, per-rank peak memory, sustained GPU utilization and production
throughput are still unmeasured.** The dependency chain can proceed automatically
only after its DDP admission succeeds; production4375083 then resumes1000 toward
2000, followed by the unchanged fixed96 and the remaining20k-target windows.

### 2026-10-04 — repair the failed DDP launch and restore Comet visibility

After the preceding queued handoff, admission job4375081 failed before starting
its workers: torchrun parsed the worker's `--run` as an ambiguous abbreviation
of its own `--run-path`. Its dependency chain cancelled correctly and the relay
marked Comet crashed. The user reported the missing running experiment. The
cause was an actual launch failure, not an absence of historical Comet data.

Added `--` between torchrun options and the worker module. A regression now
runs this exact command shape with two real CPU workers and confirms both
receive `--run` and `--until`. Three focused staged-launch tests passed. No
pretrained checkpoint existed in either failed probe directory. The remote
repair archives old/new executor hashes and the prior identity in
`runs/FLUX1_cluster_4b_ddp_20261004/execution_repairs/torchrun_separator/`.
Model/optimizer/config/data/checkpoint contents are unchanged.

The replacement chain starts with cache job4375267, then combined admission
and first training job4375268, with final summary4375337. This lets a successful
DDP test continue directly into real training using the same two-V100 allocation.
Only the first 1,000-update window has reduced allocation limits: one hour for
2,000 cached pairs, two hours for admission plus training. Later 2,000-update
windows retain their earlier limits. Old completed prerequisites were verified
from receipts/accounting; Slurm rejected their expired controller IDs as new
dependencies, so the first replacement job has no such dependency. Old pipeline
receipts remain preserved; no duplicate jobs are active.

Workstation relay `rsrch-clust-ddp-comet-4375337` keeps the same immutable Comet
key and `ddp_4375151/` curve. Added an independent acknowledged heartbeat using
the server interval (5s locally), so artifact transfers/API reads cannot make
an active experiment appear stopped. It clears the crashed flag while the
pipeline is active and stops on terminal pipeline state. Five uploader tests
passed. API confirmed running=true/hasCrashed=false; added searchable tags
FLUX1/HSE-cluster/2xV100/DDP. The historical experiment display name remains
`flux4b_clust_2v100_amp_qkvo_r128_20k_20261003`. The VPN tunnel had also expired;
it was re-established through the existing isolated wrapper.

At11:18UTC the one-V100 cache stage was running oncn-004, with201/2,000 text
inputs prepared. Actual DDP proof and fresh production metrics remain pending;
the agent continues verification rather than treating the live UI as training.

### 2026-10-04 — full96 at2k completed; FLUX1a launched into startup

The requested historical2k replay completed96 generations/decodes and original
ID/CLIP scoring. The final Comet check initially failed because some image
names had a duplicate suffix `(1)` without `.png`; all96 images were present.
`scripts/reconcile_flux1_2k_comet.py` normalizes that API presentation in a
separate helper, preserving every frozen inference/scoring file. Dedicated
Comet2b3eda1b1f364ecfaa584ffe273102a1 now verifies96 images and all metrics at
2000 with8k points intact. The resumed local collector verified523 downloaded
files and mirrored the panel into cfbd6f879e3d4158959bbbef7fb8e894 under
`validation96/*`/`fixed96_2k/*`, preserving pilot12. Collection is complete.

Measured full96 owner-ID: native0.274185,2k0.317432,8k0.276218. The8k-minus-2k
paired mean is-0.041214, descriptive identity-cluster95% interval
[-0.067432,-0.013783] (50,000 resamples, seed142); seven of eight identity means
fall, with31/96 image wins at8k. Thus the larger panel supports the decline;
it does not establish its causal mechanism. CLIP is29.353265 at2k versus
29.367097 at8k. All96 faces remain detected/owned, with no ambiguous ownership.
Receipt: `reports/261004_FLUX1_review/evidence/full96_2k_8k_comparison.json`.

The user authorized FLUX1a training immediately after that validation. New
checkout `/workspace/rsrch_FLUX1abc` has218 verified deployed source files,
a separate pinned Toolkit checkout with the new patch, and an isolated Python
overlay. Original model packages, historical runtime, weights and baseline
outputs remain intact. Added ONNX1.23.1/protobuf7.36.2/ml_dtypes0.5.4 match the
local tested executor. GB10 has about127GiB filesystem space free at deployment.

Supervisor `rsrch_flux1a` runs `scripts.launch_flux1a_gb10`, with immutable Comet
key2018ec7a730243bc98d922178e58aa5c and runFLUX1a_vast9B_20261004. It prepares
training-only identity labels, waits for the preceding validation's verified
receipt, then runs the reviewed admission and serial0/2k/4k full96 pipeline.
No new rental or automatic machine stop is configured.

The first CPU preparation was narrowed from all47,341 possible targets to the
exact4,000 scheduled samples to avoid preparing labels never used in this run.
This changes no training pair/order, noise draw, objective, model or resolution.
The label manifest explicitly records scheduled scope; coverage is verified
using the same trainer's sample_at/epoch_order, including epoch boundaries.
Four worker processes with two ONNX threads each replace serial preparation.
Two-target parity gave exactly equal embeddings/landmarks, and the prepared
scheduled manifest passed the loader coverage/geometry/weight/hash checks.
The earlier partial full-preparation directory was archived, not deleted.

At11:26UTC the new supervisor was running CPU label preparation with792/4,000
receipts; pretrained memory/parity/resume admission and optimizer updates were
still pending. Do not describe that state as completed admission or trained
quality. Startup record: scratch/FLUX1a_deploy/launch_verified.json locally;
remote controller log: runs/FLUX1a_controller.log; setup status/preparation log:
runs/FLUX1a_vast9B_20261004_setup/. The existing shared GPU lock serializes work.

For discoverability, the existing Comet entry's **display label only** was changed
to `FLUX1_cluster_4b_2V100_DDP`. Its key, data history, filesystem run names and
checkpoint identities are unchanged. `cluster/original_display_name` records
the former label. A fresh API read confirmed the new label, running=true and
hasCrashed=false. This supersedes the preceding note about retaining the old
Comet display label.

### 2026-10-04 — FLUX1a GB10 apparent-stall investigation

The supervised job was progressing through CPU identity preparation and real
pretrained admission; no deadlock or model failure was found. All4,000 scheduled
training labels finished, with3,944 accepted (98.6%). Admission completed at
11:53UTC: native BA-off and zero-mask outputs exact, all64 branch tensors updated,
frozen backbone unchanged, and fresh-process parameters/optimizer/scheduler/RNG/
cursor replay exact. The BF16 run has no active gradient scaler; the receipt's
`exact_gradient_scaler:false` denotes that non-AMP case, not failed replay.

Measured largest-layout peak CUDA reserved60.667969GiB (49.881% of device), below
the recipe85% gate. Forced identity-gradient norm0.002956365; matched flow-gradient
norm0.066533527; weighted identity/flow ratio0.0444342. ArcFace GPU-versus-ONNX
max-absolute embedding difference0.00153184 passed the declared0.01 bound.
These are admission checks, not measured identity-quality improvements.

The existing controller moved automatically through initialization into full96
step0 inference. At12:00UTC the first2/96 latents were written (99seconds for
the first pair), with96% observed GPU utilization. Main-run optimizer step remains
0 until this required panel is generated, decoded and scored; the admission's
two-update replay must not be reported as production training progress.

Comet's launch/stage had remained at CPU preparation, creating misleading stale
status. Added a separate `scripts/publish_flux1a_progress.py` REST publisher under
supervisor `rsrch_flux1a_progress`: every60seconds it publishes controller stage,
real optimizer step, validation latent/image counts and heartbeat to the existing
2018ec7a730243bc98d922178e58aa5c key. It does not open/end the training SDK session
or alter any frozen model/config/checkpoint source. It exits when the supervised
job terminates, and reports stopped/failed status instead of inferring success.
Remote source hash is recorded in setup/progress_publisher_deployment.json.
Local setup/validation/partial-metric snapshot checks passed; Comet read-back
confirmed runtime/live_progress. Admission receipts were downloaded to
runs/FLUX1a_gb10_healthcheck_20261004/. No restart or scientific change was needed.

### Cluster DDP restart verified — 2026-10-04, 12:02 UTC

Job4375268 started oncn-004 with two V100s after conditioning4375267 completed
in26m48s. The repaired launcher reached both actual pretrained workers.
Continuous1000→1002 and fresh-process1000→1001→1002 matched bit for bit for
all adapter tensors, optimizer, scheduler, scaler, per-rank RNG and cursor.
All64 adapter tensors changed; both updates had finite gradients and no loss-scale
retry. The admission receipt is `execution_admission.json` in the continuation.

Production resumed from1000 and reached1041 with finite loss1.0119128525.
Measured updates1003–1041 averaged3.0394s; maximum CUDA reserved was17.3398GiB
(under55% of device capacity), with zero overflow retries. A60-second sample
of GPUs mapped to this run's two actual training-worker PIDs measured mean
utilization89.607%/86.750%, mean power222.14W/210.19W, and process/device memory
peaks18324/18304MiB. The measurement is `gpu_utilization_training.json`; CUDA
reserved figures come separately from the trainer. This establishes real use
of both V100s, not merely two allocated GPU resources. Effective batch remains2.

The workstation relay independently read back new Comet loss through1028,
value1.0321552157, matching the local mirrored metrics. Comet metadata reported
`FLUX1_cluster_4b_2V100_DDP`, running=true, hasCrashed=false under the unchanged
key5d31de48010446248639e65a65236cbe. The active chart is
`ddp_4375151/train/loss`; the old unprefixed curve is historical. The verified
first new point was1004. Publisher receipt and independent readback live in
`runs/clust_comet_mirror/FLUX1_cluster_4b_ddp_20261004/`. Fixed96 step2000 and
completion through20k remain outstanding; queue delays and later validation
time are not included in the measured optimizer-step speed.

The user then explicitly prioritized immediate training over remaining initial
checks. The healthy step0 inference was stopped at a pair boundary/partial panel,
preserving checkpoint0 and completed latents. The new isolated orchestration
wrapper `scripts/resume_flux1a_train_first.py` takes the same GPU lock and starts
the unchanged trainer from checkpoint0 to2000 immediately. It reuses completed
admission without rerunning it. After2000 it returns to the original controller,
which finishes the preserved step0 panel, evaluates2000 and proceeds to4000.
This is an explicit user-authorized scheduling deviation from initial validation
before training; fixed96 content, metrics, model, loss, sample order and checkpoints
are unchanged. setup/training_first_authorization.json records the request,
controller hash and preserved output count. Actual optimizer advancement is being
checked before reporting training as running.

At12:06UTC production training was verified advancing through steps1–4 in the
main run (not admission). Step4 included the active identity auxiliary with
finite loss1.078796 and gradient norm0.103000; peak reserved59.416GiB/48.85%.
Observed updates took5.3–7.5seconds. Supervisor8576/trainer8593 remain running;
Comet keeps2018ec7a730243bc98d922178e58aa5c. The training-start evidence is
runs/FLUX1a_gb10_healthcheck_20261004/production_start.json locally.

### 2026-10-04 — FLUX1a cluster4B setup and submitted schedule

The user explicitly requested FLUX1a on two V100s for20,000 updates, validation
every2,000, and concise progress/ETA/results with Comet metrics/images. New
config `configs/clust/FLUX1a_cluster_4b.yaml` preserves the FLUX1a isolated-image
bank, binary ownership and identity auxiliary0.05/sigma<=0.5. Cluster variant
uses4B/768px/rank128,25,165,824 trainable parameters, effective batch2 and the
existing V100 selective-FP16/FP32-master policy. It starts fresh; neither the
FLUX1 cluster checkpoint nor Vast9B adapters are relabelled or resumed.

Implementation changes: allow the explicit V100/two-worker FLUX1a configuration;
keep a live frozen VAE for differentiable identity loss with cached conditioning;
run the isolated no-grad reference pass through the underlying module and the
target pass through DDP; pass/reduce identity metrics on both ranks; cover the
full40,000 global sample positions in identity preparation. Checkpoint code
identity now covers the isolated-bank/identity objective modules. Existing
frozen deployments were not changed.

Seventeen focused tests passed, including an actual two-process CPU DDP test
of repeated checkpointed reference/target passes, synchronized finite gradients,
label coverage, native/zero-mask parity and Comet relay regressions. A further
successor-submission test passed, proving repeated completion handling queues
only one next stage. These are not pretrained V100 admission or quality results.

Preparation4375592 was deliberately cancelled after8m22s: InsightFace0.7.3
dropped sess_options before constructing ONNX sessions, creating270 threads per
worker on16 allocated cores; no target receipts completed. The discarded
thread parameter is now injected at ModelRouter, and actual session limits are
asserted. Original diagnostic output/preparation directory were archived. New
preparation4375612 runs with16 one-thread workers and reached11,576/40,000
labels with~22min ETA. The prepared data rows reuse the existing CPU-verified
manifest and are bound to its exact hashes; consumed image bytes are rehashed
before caching. Validation identities/images remain disjoint.

The first attempt to submit all79 stages reached the students MaxSubmitJobsPU=100
limit after35 accepted records (the other FLUX1 pipeline already occupied65
slots). Only34 new pending FLUX1a jobs were cancelled; active preparation and
all existing FLUX1 jobs remained. The archived receipt records these jobs. The
replacement finite plan now has preparation4375612 followed by plan4375658;
each successful stage submits exactly one afterok successor. Failures halt the
chain. Remaining stages explicitly include all checkpoints/validations through20k.

Runtime `/home/nasilaev/rsrch_new_FLUX1a` has207 verified source hashes based on
local commit a08197950a036cbf858a1d1466993f04b99e3f5a plus dirty sources, with a
separate Toolkit checkout pinned to ecee894ed2b1f3716d9d7326693061ec1a3105bb.
The original FLUX conda env is activated and a private training overlay adds
ONNX1.23.1/protobuf7.36.2/ml_dtypes0.5.4. The training memory/throughput and
identity-gradient admission on V100 have not yet been measured.

Separate Comet key9a4c6ef2e160413dad596a3dea2dc451 is created, display label
`FLUX1a_cluster_4b_2V100`. The user service
`rsrch-clust-FLUX1a-comet-4375612` is active and prints concise stage/count/ETA.
Framework/SSH/offline-SDK noise is excluded from the new console; detailed
diagnostics remain in setup files. Training publishes every second update to
avoid the per-metric point cap, preserving full JSONL locally. All96 validation
images, comparison panels and original ID/CLIP/face-quality summaries are
published. Summary publication now checks metric history, avoiding repeated
re-upload of older panels when a later summary exists. GPU admission/step0
validation/production training are still pending at this record.

Final startup verification: remote CPU allocation imported the exact deployed
runtime successfully with torch2.7.1+cu126 and the pinned ONNX overlay; normalized
data contains47,341 training pairs and96 validation items. All207 deployed hashes
were rechecked after the final executor changes. Compute-node sbatch is available
and accepted the two-V100 admission dry run. Its latest scheduler-only estimate
was2026-10-07T02:23:17MSK; this is not an actual allocated GPU start. Preparation
4375612 reached19,544/40,000 labels at15m09s, ETA15.7min; plan4375658 is pending
on that dependency, with77 subsequent stages saved through summarize20000.
Comet API independently confirmed the intended display name, running=true,
hasCrashed=false and structured preparation/count/ETA status. Evidence is in
`runs/FLUX1a_cluster_launch/comet_startup_readback.json`. The live status denotes
CPU preparation, not production optimizer updates. The new relay also stays on
the setup identity until initialization has written the main-run Comet record.

### 2026-10-04 FLUX1a cluster preflight OOM and recovery

At 17:50:23 MSK, admission4375704 failed before production initialization.
The cache probe4375699 completed successfully, including exact fixed96 encoder
and VAE conditioning parity. The admission traceback is a CUDA OOM in the FP32
VAE decoder during the forced identity-gradient backward:31.73GiB device,
31.16GiB PyTorch allocated,288MiB requested. No production optimizer checkpoint
was created. This was not an HPC TaskMaster idle cancellation.

The outer whole-decoder checkpoint recomputed without internal block
checkpointing, retaining full-resolution decoder intermediates during backward.
OnlineIdentityObjective now enables the pinned native decoder's block
checkpointing when training gradient_checkpointing is enabled. A small native
FP32 decoder regression verified bit-exact outputs and input gradients, with
frozen weights. This is CPU evidence, not a measured V100 fit claim. Failed
admission/source/deployment records are archived remotely under
`scratch/recovery_4375704` and setup/admission_failed_4375704. One deployed source
hash was changed with a before/after audit entry; no existing production identity
or checkpoint was rewritten. Resolution, objective, two-worker batch2 and fixed96
validation contract are unchanged.

Replacement two-V100 admission4375739 is queued. A separate15-minute one-V100
`test` allocation4375743 runs the full pretrained native/identity/memory stress
probe first;4375739 depends on its success, so these allocations do not overlap.
The pipeline records both and the previous failure. At18:10MSK both were pending;
Slurm estimated18:33:10MSK for the test, with no reliable two-GPU start after that.
Production remains at zero updates until admission and step0 validation succeed.

The workstation VPN also lost its tunnel, causing the Comet publisher to fail
before it could fetch the real Slurm failure. Restored via the existing VPN
controller, then recreated the same publisher service and immutable Comet key
9a4c6ef2e160413dad596a3dea2dc451. The publisher now records transport failures as
unreachable/stale, clears its explicit liveness heartbeat and retries observation;
it does not infer a worker crash from an SSH failure. An outage-then-recovery
regression passed. Four identity/executor tests and six Comet tests passed.
Comet API readback confirmed connected monitoring and pending scheduler status;
`running=true` reflects the open experiment, not production optimizer progress.

Recovery GPU measurement: memory probe4375743 completed0:0 on cn-020 at
18:13:31MSK after5m40s. The actual pretrained V100 run passed native/BA-off and
zero-mask parity, frozen-weight equality, nonzero finite identity gradients,
two adapter updates, and full-routing stress at1024 reference tokens/2304 target
queries. Peak CUDA reserved was23.138671875GiB (72.9178%), below90%; identity
branch-gradient norm0.00401904, weighted auxiliary loss0.00916606. This verifies
the decoder fix on one V100; two-rank exact save/resume remains the next gate.
Receipt copied to `runs/FLUX1a_cluster_launch/recovery_native_checks.json`.
Admission4375739 is now pending Priority, with latest scheduler estimate
21:29:12MSK (19:29:12BST); this is not a guaranteed start. Production is still0
updates. Ten focused local tests passed across identity, executor and Comet.

### 2026-10-04 16:57UTC — FLUX1a 2k backup and validation follow-through

User requested a local2k checkpoint, verified2k Comet validation, continuation
to4k and verified4k validation/publication. The job had not failed: checkpoint
2000 completed at15:31UTC; the existing serial controller was finishing deferred
step0 inference. Progress advanced78→84/96 during inspection; GPU96% busy.
No healthy model process was restarted, and frozen scientific sources/configs
remain unchanged. The original controller still performs full96 at0/2000/4000,
with optimizer-state-preserving continuation from2000 to4000.

Copied the complete checkpoint002000 to local
runs/FLUX1a_vast9B_20261004/checkpoint-002000, including optimizer/scheduler/RNG
state and configs. All five files matched remote SHA256; adapter hash
cdb834539c64b05243df637ab3d00906f8e3b76526e42267987d48fe9641efb0.
Receipt: runs/FLUX1a_vast9B_20261004/checkpoint_002000_download_verified.json.
Local disk is nearly full; avoid unrequested large result copies.

The separate progress publisher now writes real stage/count transitions into
Comet console output and excludes raw decode intermediates from its96-image
count. It also runs scripts.verify_flux1a_validation after summarize receipts
at2000 and4000. That helper verifies original96 order/prompts/seeds, checkpoint
and panel hashes, complete per-image scoring, exact background audits and
Comet read-back for all96 images and metric values. Missing uploads may be
repaired via REST; mismatched existing values fail rather than get overwritten.
Per-step receipts are comet_verified_002000.json/comet_verified_004000.json.
Publication verification is still pending until panels actually finish.
Only rsrch_flux1a_progress was restarted; model controller PID8576 stayed live.
Deployment hashes are in setup/publication_verifier_deployment.json.

Created thread heartbeat verify-flux1a-validation-and-finish-4k, every30minutes,
to verify advancement, handle evidence-backed failures, and confirm both Comet
receipts plus completed4000 before pausing itself. It forbids unrelated HSE
changes, duplicate GPU jobs, repeated admission, rental/stop/termination or Git
publication. This extends follow-through; it is not a claim that2k validation
or4k training has completed already.

### 2026-10-04 — live Comet lifecycle and streamed FLUX1a validation

The user required the experiment to stay RUNNING across all active stages,
console logs during validation, and images uploaded immediately after generation.
The previous REST-only publisher did not own experiment lifetime; stage SDK
sessions ended independently, and generation/decoding were separate full-panel
passes. Both caused misleading idle status and delayed images.

The new scripts.flux1a_live entry point wraps the frozen serial controller. Its
workers retain normal SDK flush/close but suppress only their run-ended signal;
the long-lived supervised progress publisher now owns one ExistingExperiment
session until the job and publication checks finish. It tails current stage logs
every5seconds through SDK console capture. Comet API read-back after deployment
confirmed running=true and hasCrashed=false, with current validation log lines
at the end of console output.

For inference, a report-write hook decodes each completed pair with the same
frozen VAE and native-background composition, and immediately uploads each image
synchronously to fixed96/<sample> at the actual checkpoint step. It preserves
partial latents and an acknowledged per-image upload receipt, then the normal
scoring stage follows after96. The later decode stage verifies receipts and
skips duplicate decoding/uploads. A real existing step0 latent provides exact
raw/composed pixel parity before first streamed output; decoder loading preserves
RNG state. The original denoiser, masks, sample order, sampler, training sources,
configs and checkpoint identities are unchanged. New wrapper/publisher sources
are archived with hashes under the run setup/live_streaming_source_* directory.

Supervisor rsrch_flux1a was switched from resume_flux1a_train_first to
scripts.flux1a_live at17:25UTC, preserving10 completed checkpoint2000 latents.
No admission was repeated. Supervisor rsrch_flux1a_progress retains the existing
2k/4k full96 Comet verifier and the follow-up heartbeat remains active. The
runtime is still /workspace/rsrch_FLUX1abc, key2018ec7a730243bc98d922178e58aa5c.

Live-streaming verification at17:28UTC: Comet API reports running=true and
hasCrashed=false; ten checkpoint2000 fixed96 images are already present while
inference is still incomplete. API console read-back includes current generated/
decoded counts and per-image upload acknowledgements in chronological order.
The streaming decoder's pretrained check passed exact raw and composed RGB pixel
parity against saved step0 sample00. Evidence:
runs/FLUX1a_gb10_healthcheck_20261004/live_streaming_comet_verified.json locally,
and live_decode_parity_002000.json / live_decode_002000.json in the remote run.

The user's follow-up required batched validation as in rsrch_apr_test. Read-only
inspection of that project's base_trainer evaluation loop, sdxl_trainers batch
pipeline call and _log_batch, and CL39r4 saved config confirms batched denoising
with per-sample generators and image logging after each batch. The saved CL39r4
manual_val batch size is12 (its generic clean_full dataloader default is1).
FLUX1a preserves its existing same-reference batch2 denoising and now follows
the same generate-batch→decode/upload-batch order. No batch-size change or new
scientific panel was silently introduced; fixed96 is48 generation batches.
Reference implementation paths:
/home/kolyangg/rsrch_apr_test/diffusion_template/src/trainer/base_trainer.py
/home/kolyangg/rsrch_apr_test/diffusion_template/src/trainer/sdxl_trainers.py
/home/kolyangg/rsrch_apr_test/diffusion_template/artifacts/checkpoints/CL39_cosmic_null_key_confidence_router_24k_full96_r4/config.yaml.

Heartbeat17:32UTC: FLUX1a supervisors healthy; GPU96% active in checkpoint2000
inference. Comet API running=true/hasCrashed=false and live console output verified.
The complete step0 panel passed the new verifier: all96 image uploads and all
metrics match local scoring, exact panel/checkpoint/background checks passed.
Copied comet_verified_000000.json and the small quality summary locally. Step0
owner-ID0.02638946 is the untrained isolated-branch result, not native or2k.
Checkpoint2000 generation/upload was advancing beyond the first12 images; its
complete scoring/publication and subsequent2000→4000 training remain pending.
No GPU process was interrupted or scientific source changed on this heartbeat.

### 2026-10-04 cluster Comet delivery and streaming validation repair

Live inspection around20:25MSK found no training process waiting on a network
upload: FLUX1 train4000/job4375275 and FLUX1a admission4375739 were both pending
Priority with no unsatisfied dependency. FLUX1 had completed step2000 inference
(41m01s), decode(3m28s), scoring, summary, and cache4000(55m06s). Those are actual
Slurm timings; current production progress is2000 for FLUX1 and0 for FLUX1a.

The workstation publisher had two independent faults. It looked for image names
with an appended `.png`, but Comet's logical names were extensionless; one
sample had25 duplicate copies at step0 and25 at step2000. Its full-experiment
asset-list calls then timed out. It also withheld images until decode_N.done,
and serialized scheduler polling behind rsync, upload flush and acknowledgement.
The fix uses per-name/step lookups, durable enqueue/acknowledgement receipts,
bounded batches and retries without restarting or re-enqueueing pending images.
New upload names include their checkpoint; old logical names are recognized.
Scheduler polling now runs independently of the publication worker. Complete
PNGs are eligible before all96 images finish, and truncated PNGs are retried.
All217 existing FLUX1 images, all required assets, loss through2000 and validation
summaries were confirmed with zero pending work. Named-image readback confirmed
old duplicate counts stopped growing; historical duplicate assets were retained.

The VPN tunnel's IPinfo country check returned HTTP429. Added an HTTPS ipwho.is
fallback inside the existing isolated namespace, retaining IP/country validation
and refusing a reported country mismatch. Provider failure never triggers bare
SSH. The installed and maintained controller sources match. The encrypted local
OpenVPN test passed with unchanged host routes/DNS and fail-closed IPv4/IPv6;
focused tests covered429 recovery, mismatch and both providers unavailable.
A real reconnect independently verified RU and nasilaev@login-02. Both relay
services were recreated with current WSL interop settings, then successfully
read and published fresh Slurm states under their original experiment keys.

For images during inference, scripts/clust_stream_decode.py observes the pinned
sampler's completed validation reports and immediately decodes each new latent
with the same VAE, generated mask and preserve_background operations. PNGs are
atomically renamed into visibility; decoder construction/decoding preserve CPU
and CUDA RNG states. No network calls run on the compute worker. Later decode
stages verify completed images instead of decoding them again. Fixed96 order,
seeds, prompts, native backgrounds, inference equations, training/checkpoint
contents and metrics definitions are unchanged. Original executor/identity files
were archived in each remote scratch/stream_decode_change_20261004, with explicit
before/after source records; the FLUX1 source identity and FLUX1a deployment
include the new helper. Checkpoint/config/training-code digests were not changed.

Sixteen focused tests passed, including pixel/RNG preservation, restart safety,
partial-PNG publication, delayed Comet acknowledgement without re-enqueueing,
legacy extensionless names and independent scheduler observation during a blocked
upload. The affected tests were rerun after adding the admission dependency and
checkpoint-specific upload names. GPU parity/memory probe4376019 started on
cn-002; it compares a real step2000 validation PNG with the full FLUX1 backbone
and adapters resident. Streaming inference is gated on its success: FLUX1's next
infer4375276 depends on train4375275 and probe4376019; FLUX1a's rolling executor
adds the same probe dependency to inference stages. Production jobs remain queued.

GPU streaming admission completed:4376019 exited0:0 after4m05s on cn-002.
With the actual4B backbone and step2000 adapters resident, the streaming decoder
produced pixel-identical output to the original saved step2000 validation image.
Peak CUDA reserved17.537109375GiB (55.2654%); first decode including VAE loading
11.19s. This qualifies the shared decode execution change; it is not a new
training/quality result. Passed receipts are saved in both deployments. The
rolling FLUX1a plan now records the passed result instead of retaining a future
dependency on a Slurm job ID that could expire. Both production jobs remain
pending Priority; neither is waiting on Comet or the completed probe. Final
publisher state:217/217 FLUX1 images confirmed,0 pending assets/metric updates,
and connected monitoring for both immutable Comet experiments.


### 2026-10-04 FLUX1a Vast9B validation throughput qualification

The user requested larger/faster validation batches. Real checkpoint2000 GPU
probes on the largest reference layout tested batches2,6,12 with the original
20-step/CFG4/BF16 protocol. With paired-CFG reference reuse, the measured
seconds per image per denoising step were2.850,2.903,3.032 respectively; peak
reserved20.85,27.85,38.29GiB. Larger batches fit but did not increase throughput.
They also changed BF16 predictions (relative RMS0.006824/0.006688), so batch2
was retained. These are throughput probes, not quality or full-panel results.

A separate inference-only operational wrapper now caches the frozen image-only
reference bank by exact reference tokens/positions/mask, frozen model identity,
maximum selected keys, and exact sigma. It reuses features for positive/negative
CFG and subsequent prompt batches with identical references, retaining at most
24 timesteps and invalidating on reference changes. No target or prompt enters
this cache; training never installs it. Pinned model/training sources and root
resolved config remain unchanged. Deployment snapshots and source hashes are in
runs/FLUX1a_vast9B_20261004_setup/validation_speed_* on the existing Vast host.

After extending reuse across prompt batches, real-model batch2 measurements were
6.35165s per positive/negative CFG pair without caching,5.70673s with a cold
reference cache, and5.06743s with a warm reference. Both cached predictions were
bitwise equal to the uncached baseline. Six batches per reference imply a
predicted1.228x denoising throughput (18.5% less denoising time); this is not yet
an end-to-end panel timing. The probe's peak reserved was20.8965GiB/17.18%; the
full20-timestep cache's production peak will be reported by the inference audit.
A focused cache check also covered exact-reference clones, sigma changes,
in-place reference mutation and rejection of gradient-enabled execution.

Execution resumed checkpoint2000 validation with28 completed images retained;
each new pair still decodes/uploads immediately and the independent Comet owner
remains active. The same cache policy will apply at4000 in a fresh process with
its own checkpoint, without reusing checkpoint2000 features. Small benchmark
receipt copied locally to runs/FLUX1a_vast9B_20261004/validation_execution_policy.json.
Full2k scoring/publication, resumed training and4k validation remain pending.

Live execution after deployment advanced28→30/96, with both new images decoded
and acknowledged by Comet immediately. The first cold-cache generation pair,
including streamed decode/upload, took122s. Main supervisor and progress owner
remained RUNNING. This establishes resumed production inference rather than
only a successful isolated probe; warm-cache/end-to-end panel timing is pending.


FLUX1a Vast9B milestone19:01UTC: full96 step2000 generation, scoring and Comet
publication completed. The verifier confirmed all96 images, all9 metrics,
original panel order/prompts/seeds, checkpoint provenance, per-image scoring and
background audits. Receipt comet_verified_002000.json and quality_summary.json
are copied locally. Owner-ID0.28100960698; CLIP29.32846971353; no-face/unowned/
ambiguous rates0. The controller entered train_4000 from checkpoint2000; actual
resumed updates will be checked separately. Step4000 validation remains pending.

At19:05UTC actual resumed training advanced through2006 with finite loss and
gradients; peak reserved59.785GiB/49.16%, ~6.16s average/update. The local receipt
runs/FLUX1a_gb10_healthcheck_20261004/resumed_above_2000.json records a subsequent
live update. No restart or repair was needed during this milestone check.


FLUX1a Vast9B interruption21:01UTC: instance53994096 reports actual_status=exited,
intended_status=stopped; SSH refuses connections. Comet is no longer running;
last logged update2980 at20:44:17UTC. The old8k stop service is inactive and its
last stop receipt is from this morning; no evidence it caused this interruption.
The stop's initiator is unknown. Asked the user whether to restart the existing
instance because its explicit stopped state may be intentional; do not override
that state until clarified. Fixed96 step2000 and local2k checkpoint remain
verified; no4k checkpoint/publication is verified. Remote checkpoint inventory
cannot currently be inspected. No machine operation was issued by this check.

The user confirmed the interruption was exhausted Vast credit. The saved2k
resolved config uses checkpoint_every=500 (validation remains every2000), so
checkpoint-002500 is expected to be the latest remote save after logged step2980.
This has not been directly inspected on the stopped host; only checkpoint2000
is downloaded and SHA256-verified locally. Inspect latest_checkpoint.txt and
complete checkpoint manifests on restoration before choosing a resume step.


FLUX1a restoration22:02UTC: existing Vast53994096 became running/accessibile
again; both experiment supervisors were stopped and no GPU compute process was
present. Verified checkpoint002500 has all five nonempty files, SHA256 hashes,
and matching config/data/training-source/parameter manifests against002000.
The last local remote-metrics row was2981 (Comet had2980); updates2501–2981 were
not checkpointed and must be replayed. The original controller already selects
the latest complete checkpoint and archives rolled-back local metric rows.
Started rsrch_flux1a and rsrch_flux1a_progress using their unchanged commands;
no source changes, new rental or machine lifecycle command. Confirmation of new
optimizer updates remains pending initialization. Remote checkpoint verification
receipt is local runs/FLUX1a_gb10_healthcheck_20261004/checkpoint_2500_remote_verified.json.

At22:06UTC resumed production updates2501/2502 were verified with finite loss
and gradients. Controller1407/trainer1550 and progress publisher1428 are active;
Comet Running=true. The original checkpoint restore completed successfully.
This supersedes the credit/stopped blocker and pending initialization status.


### FLUX1a6k continuation and automatic local checkpoint transfer —2026-10-04

The user requested4k and6k checkpoints locally after reviewing the remaining
credit budget. Main rsrch_flux1a continues uninterrupted through4k/full96.
New rsrch_flux1a_6k is a CPU-only waiting supervisor: after4k validation and main
exit, it acquires the shared GPU lock, resumes the latest complete checkpoint,
trains to6000 and runs full96 inference/decode/score/summarize. The separate
continuation config changes only training.steps; strict verify_extension passed
against checkpoint2500 and frozen source guards passed. Original root config,
model/training source hashes and checkpoint contents remain unchanged. Worker
config for6000 is explicit in resolved_config_6000.yaml when continuation starts;
continuation_6000.json records authorization and operational source hashes.

The existing live wrapper supports6000; Comet publisher follows both supervisors
and verifies2000/4000/6000. It remains the sole lifecycle owner and preserves
immediate per-batch images. Operational files were archived before/after under
setup/continuation6k_*; the main GPU trainer was not restarted. Production had
advanced to2635 while the queue and publisher were confirmed active.

Local user service rsrch-flux1a-checkpoint-download runs
scripts.mirror_flux1a_checkpoints every60s until both filesets are verified. It
starts downloads as soon as each complete atomic4k/6k checkpoint appears, before
validation finishes. Destination:
/mnt/c/Users/ogure/FLUX1a_checkpoints/FLUX1a_vast9B_20261004
(Windows C:\Users\ogure\FLUX1a_checkpoints\FLUX1a_vast9B_20261004).
This drive has~100GiB free; Linux has only~1.5GiB. Each checkpoint is~385MiB.
All five files are SHA256-verified against remote hashes, staged in a .partial
folder and renamed only after success. Receipts are mirrored beside checkpoints
and in runs/FLUX1a_checkpoint_transfers. Focused checks passed successful copy,
receipt-write restart recovery, corruption rejection and incomplete checkpoint
waiting. Live one-shot and daemon checks correctly show4000/6000 pending.

The heartbeat was updated to finish6000, verify4k/6k Comet publications and both
local download receipts. Latest budget forecast was~9.3 funded hours, versus
~8.9hours to6000 including both validations; credit margin remains small. No
machine stop/termination or extra rental was requested or performed.


### 2026-10-04 23:30 BST — cluster progress and CPU verification queue fix

Live checks found FLUX1 checkpoint4000 complete (train4375275:1h55m31s;
infer4375276:42m12s), all96 step4000 images published, and Comet publisher
313/313 images acknowledged with no pending metrics/assets at that observation.
Full validation scoring was still pending. FLUX1a two-V100 admission4375739
completed; initial fixed96 validation4376512 advanced70→80/96 during inspection
(~31s/image, ~8min generation remaining). Production optimizer step remains0
until initial validation/scoring/cache complete. Neither publisher was stuck.

Found a leftover GPU reservation for decode stages after the streaming decoder
change. These stages now only verify PNG/checkpoint hashes on CPU. FLUX1
decode4000 had an estimated GPU start03:50MSK despite inference finishing01:22.
Slurm in-place resource updates retained GPU ReqTRES; replaced only the nine
pending verification jobs for4000..20000 with CPU-only jobs4376621..4376629,
rewired each scoring dependency before cancelling its superseded verifier,
and updated the pipeline ledger. Completed inference dependencies already
purged from the scheduler were omitted only after sacct confirmed completion.
The new decode4000 started01:30MSK oncn-031 with AllocTRES=cpu=2,node=1.
Training/inference jobs and all checkpoint/scientific/source identities remain
unchanged. Superseded decode cancellation is intentional maintenance.

FLUX1a remaining_stages was atomically updated under its scheduler lock so all
11 future decode verifiers request0 GPUs. Local fresh-launch stage definitions
now match. Archives/receipts in each remote root scratch/decode_cpu_20261005.

Verification: CPU decode4376621 completed0:0 in20s, all96 PNG hashes passed;
score4375278 then RUNNING. Six focused stream/stage tests passed locally
(pytest cache-write warning only).


### 2026-10-05 — GPU scoring and cluster console/queue repair

User reported old Comet throttle/upload warnings and FLUX1a initial validation
delay. Actual FLUX1 checkpoint4000 and its96 images were complete; CPU
face_quality4375279 was still advancing. Comet API output is returned with recent
status entries before historical warning output. Explicit wall-clock timestamped
console publication now makes current stage/step/count/ETA clear. Comet still
reports a historical metrics throttle flag for FLUX1; the continuation loss curve
was independently read at4000 and all313 images were acknowledged. Found a further
asset acknowledgement bug: run metadata requested stepNone but Comet stored0.
Two FLUX1a admission JSONs were reuploaded repeatedly. Match run-level assets
without enforcing an optimizer step, and version changed metadata by SHA256.
Both relays now show0 outstanding assets once current publications settle.

Both scoring environments were torch2.2.0+cpu. Created isolated metrics-gpu
(torch2.2.0+cu121, torchvision0.17.0+cu121, numpy1.26.4, Pillow11.1.0), reusing
pinned CPU scoring dependencies via .pth. Training environments untouched.
CLIP loads original FP32 weights on CPU before moving to GPU; PyIQA retains
FP32 and the same crop/model/batch policies. ONNX face detection remains CPU
with explicitly bounded threads. GPU inference finishes in its own child before
scoring starts, releasing backbone/VAE memory. Receipts make subsequent CPU
decode/score/face-quality stages no-ops. GPU scoring activation requires the
shared environment gpu_verified.json; without proof, CPU scoring is retained.

First diagnostic4376686 exited because the selected generated face crop had no
TOPIQ-alignable face; ordinary evaluation already treats that as missing. The
diagnostic now checks matching no-face behavior instead of incorrectly failing
the execution check. Corrected4376688 completed0:0 on V100cn-004 in2m57s:
CPU/GPU point comparison max absolute differences TOPIQ0, MUSIQ7.63e-6,
MANIQA2.98e-8, CLIP9.54e-6. Full96 face-quality evaluation took82.075s;
MANIQA45.452s GPU versus1352.882s CPU. GPU peak reserved27.939GiB (~88% of
31.732GiB). Full-panel face coverage exactly matched; maximum absolute
summary difference7.42e-6. Per-production GPU scoring now also records and
checks peak reserved memory below90%. No model/loss/data/order change.

CPU postprocessing no longer holds the GPU-stage lock or deletes conditioning
while the next cache is being built. Cleanup moved to the next cache's start,
preserving all validation conditioning. FLUX1 cache6000..20000 dependencies
rewired to prior decoded validation, not CPU summary. FLUX1a initial summary
and cache submitted as independent branches; cache4376685 then gates training
4376693. Current cache time limits reduced2h→75min using actual full-window
55m06s measurement; queue estimates are not guarantees. All111 unreserved
V100s were allocated at02:07MSK (four additional V100s were reserved for others).
Training was therefore still pending capacity, not blocked on Comet.

FLUX1a summary4376684 exposed a missing RelayExperiment.log_metric method.
Added the no-network relay method; replacement4376696 completed0:0 in16s.
The failed attempt is retained in pipeline superseded_job_id. This did not
cancel the independently queued cache/training. Its publisher, which had
correctly stopped on failure, was recreated with the same immutable Comet key.
Archives/source identity amendments and benchmark receipts are in each root's
scratch/gpu_metrics_20261005. Training checkpoint code digest was checked
unchanged for FLUX1a. Eighteen focused relay/executor tests passed in4.7s.


FLUX1a Vast9B milestone2026-10-05 01:02UTC: training reached4000 and the complete
checkpoint was automatically downloaded at00:40:48UTC to
/mnt/c/Users/ogure/FLUX1a_checkpoints/FLUX1a_vast9B_20261004/checkpoint-004000.
Recomputed all five local SHA256 hashes against the remote download receipt;
all match and manifest.step=4000. Receipt:
runs/FLUX1a_checkpoint_transfers/checkpoint_004000_download_verified.json.
Full96 step4000 validation is advancing (24generated,25Comet images confirmed
by the subsequent API read); Comet Running=true/hasCrashed=false. Full scoring
and publication verification are pending. The6k controller is still waiting for
completed4k validation; downloader remains active for6000. No repair needed.


FLUX1a Vast9B2026-10-05 02:32UTC: complete4k fixed96 scoring/publication verified,
receipt and quality_summary copied locally. Owner-ID0.2943213313592423,
CLIP29.38190931081772; all96 Comet images, metrics, provenance and backgrounds
passed. Main4k controller exited normally around02:09UTC.

The6k continuation failed before update4001 at02:13UTC: OnlineIdentityObjective
correctly rejected the4000-row label cache for the extended6000-row trajectory.
The prior continuation check covered config/source compatibility but missed this
derived-cache coverage requirement. Original4k labels/scientific sources and
all checkpoints remain unchanged. Repair adds a separate FLUX1_768_to6000 cache,
seeds all4000 original per-target receipts and parity fixture unchanged, and
runs the original frozen preparation code only for missing scheduled targets.
Preparation is serially supervised as prepare_identity_6000 before6k training;
Comet progress was restarted and logs this stage. Final guard requires full6000
coverage plus exact metadata/embedding equality for all original rows.

Runtime continuation now records the extra data.identity_supervision path in
its resolved config/checkpoints. Its scoped resume adapter permits only this
verified derived-label superset in addition to training.steps; optimizer,
model, sample order, input targets, RNG and all other settings remain guarded
by the original strict extension check. Original training/model/preparation
files and root config are not edited. Operational before/after source snapshots
and amended continuation_6000.json are recorded under setup/labels6k_repair_*.
Actual resumed optimizer updates remain pending label preparation.

Credit at02:36UTC was2.3939USD (~4.8h at0.4983/h), versus roughly5h remaining
including preparation, training and full96. User notified that1USD additional
credit would provide a buffer. No credit purchase or machine lifecycle action.


FLUX1a6k recovery confirmed2026-10-05 03:02UTC: extended identity labels cover
all6000 rows, accepted fraction98.5%; all4000 original records/embeddings are
exactly preserved. identity_extension_6000.json was copied locally. Training
resumed successfully from4000 and advanced to4156 with finite loss/gradients;
peak reserved59.994GiB/49.33%. Main6k controller16449/trainer17880 and publisher
16487 are healthy; original4k controller exited normally. The label-coverage
failure is resolved. Local checkpoint downloader remains active for6000.


FLUX1a Vast9B milestone2026-10-05 06:32UTC: training completed6000 updates and
full6k checkpoint was automatically downloaded at06:20:30UTC to
/mnt/c/Users/ogure/FLUX1a_checkpoints/FLUX1a_vast9B_20261004/checkpoint-006000.
Recomputed all five local SHA256 hashes against the transfer receipt; all match
and manifest.step=6000. Both requested4k/6k local copies are verified. Downloader
exited successfully after completing both, so inactive is now expected.
Full96 step6000 inference is advancing with18 images confirmed in Comet;
Running=true/hasCrashed=false. Final scoring/publication is not yet complete.
Credit is0.4474USD (~54min), versus roughly75min remaining validation/scoring;
previous top-up warning still applies. No running job was interrupted.


FLUX1a Vast9B interruption2026-10-05 07:32UTC: credit is0USD and instance53994096
again reports exited/intended stopped; SSH refuses connections. Comet is ended
(running=false,hasCrashed=false), with78/96 fixed96 step6000 images confirmed.
Training6000 and both full4k/6k local checkpoint downloads are complete and
SHA256-verified. Remaining work: resume incomplete6k inference, generate/upload
remaining18 images, score/summarize all96, then verify Comet publication. No6k
full-panel metric result is claimed. Await credit/machine restoration; do not
purchase credit or repeatedly attempt starts. On restoration inspect complete
latents/stream receipts, resume rsrch_flux1a_6k and progress owner, preserving all
completed work. Approximate remaining GPU/scoring/reload time25–30minutes.


### 2026-10-05 — Local FLUX1a 6k validation feasibility (not launched)

User requested feasibility/timing for the saved Vast9B 6k checkpoint on the
local 16GB machine. Read-only hardware inspection: RTX4090 Laptop16GB,
13.78GiB VRAM free; WSL30GiB RAM/~28GiB available, Ryzen9 7945HX16cores.
Linux workspace has~941MiB free; Windows C: has~94GiB free. Checkpoint is
local but weights/flux9b and weights/flux9b_text are absent. Pinned lock sizes
are18.157GB backbone and16.397GB Qwen3-8B encoder. Original9B native fixed96
artifacts exist locally from earlier validation; their hashes and frozen
FLUX1a input contract must be checked before reuse. No9B conditioning configs
were found in the local conditioning cache.

Current loader moves the complete backbone to CUDA; inference explicitly uses
CUDA throughout. Existing pipeline offload patches do not supply block
offloading for this custom validation loop. Proposed feasible route: preserve
BF16 weights and patched isolated-reference BA, batch1 initially, CPU-backed
block offload, serial encoder/denoiser/VAE residency, bounded reference cache.
This needs an isolated frozen-runtime adaptation and pretrained output/parity
check; it is not yet a qualified local validation runtime. CPU-only also needs
device-plumbing changes and careful RAM management. Do not use quantization
or reduced resolution/denoising steps for canonical fixed96 validation.

A bounded synthetic4096-wide matrix/transfer probe measured GPU BF1654.89TF/s,
CPU BF161.473TF/s, CPU FP320.685TF/s and pinned host-to-device13.77GiB/s.
Receipt:scratch/flux1a_local_feasibility_probe.json. These are component
throughput measurements, not pretrained9B inference results. Planning range
for local GPU block offload: full96 ~3–6hours, remaining18 ~35–75minutes,
excluding adaptation/download/setup; full CPU roughly40–100hours for96 and
8–20hours for18. Actual attention, reference passes, cache, thermal limits and
offload overhead require a real two-image benchmark to narrow these ranges.
The78 existing6k images/latents are still remote (Comet images published);
resuming only18 requires recovering and verifying partial artifacts. No large
download, validation launch, remote lifecycle operation or source sync done.


FLUX1a Vast9B restoration 2026-10-05 12:05UTC: instance53994096 was restored externally;
credit4.373USD. Reboot left research supervisors stopped and GPU idle. Started
only rsrch_flux1a_6k (1359, inference1376) and rsrch_flux1a_progress (1393).
Completed training6000/preparation receipts were skipped; original78 outputs
and upload receipts were reused in2seconds. Worker is now generating missing
images with96% GPU utilization, and Comet Running=true/hasCrashed=false.
Final96 scoring/verification remains pending. Local4k/6k transfers remain done;
no download service restart, scientific source sync or machine lifecycle action.


FLUX1a Vast9B completion2026-10-05 12:31UTC: training6000, full96 validation
and Comet publication are complete. Supervisor6k exited normally12:24UTC;
progress owner exited12:25UTC after comet_verified_006000.json passed.
All96 step6000 images independently read back from Comet; Running=false and
hasCrashed=false correctly reflect completion. Owner-ID0.28458708553080214,
CLIP29.46402845780055;4k remains best measured owner-ID0.2943213313592423
(vs2k0.28100960698066046). No statistical significance claim.
Final verifier receipt, per-image CSV, quality summary, inference/background
audits and done receipts copied locally. Both4k/6k complete checkpoints were
already SHA256-verified on Windows; transfer receipts remain verified.
The completed follow-up automation is removed. No instance stop/termination
was performed; machine lifecycle remains under user control.


User-requested GB10 stop verified 2026-10-05 14:08UTC: Vast53994096 now actual_status=exited, intended_status=stopped. Instance was stopped, not terminated; files preserved. FLUX1a monitoring automation was already deleted after verified completion.


2026-10-05: Completed 8-page FLUX1a Vast9B PDF report with architecture, verified0/2k/4k/6k metrics, native/historical controls, identity-cluster bootstrap intervals and eight deterministic sample panels. All24 included trained sample images match publication SHA256; native examples match routing-mask baseline hashes. PDF layout inspected; no text bounds violations. Saved reports/261005_FLUX1a_results/FLUX1a_results_report.pdf and uploaded to /Apps/temp/rsrch_new/2026-10-04/FLUX1a_results_2026-10-05.pdf (Dropbox completed receipt). No GPU startup or monitoring restart.

### 2026-10-05 — Flux 2 architecture proposal after Flux 1A 6K

AICODE-NOTE: The user named the proposed successor **Flux 2**. It is a project
experiment on FLUX.2-klein Base, not a new BFL release. No Flux 2 model or runtime
configuration is implemented or admitted by this report.

Completed the 23-page [architecture review and proposal](../../reports/261005_FLUX2_proposal/Flux2_architecture_review_and_proposal.pdf),
with portable HTML, six architecture diagrams, two measured charts, eleven
primary research sources (including September 2026 papers), deployment-source
hashes and a staged ablation plan. CPU recomputation matches all existing
fixed96 scores and order. New mask-geometry analysis: median native owner-box
short side 105px; 59/96 are below 112px. The existing admission identity/flow
gradient ratio is 0.0444 at one test point, not a training-wide measurement.

The proposal separates immutable reference memory, a persistent face state and
RGB-sanitized one-way context, with all-layer ID/detail attention and identity
modulation. It specifies 9B/80GB-class, 4B/48GB and 4B/16GB candidates plus a
separately named ROI384 local fallback. Flux 2 quality, memory and throughput
remain unmeasured; proposed_profiles.json is explicitly design-only. The PDF
was rendered and inspected, with no text-bounds violations. Historical model
sources/checkpoints were preserved; no training, GPU lifecycle action, commit
or push was performed.


## 2026-10-05 — FLUX2 Lite 4B local implementation and admission

AICODE-NOTE: User explicitly authorized committing/pushing the existing baseline,
implementing FLUX2, committing verified code, then launching a local 16GB one-ID
2,000-update pilot with validation every 500. Baseline `b5ee039` was pushed first.
FLUX2 now implements persistent directed face states at all 25 blocks, separate
ArcFace/DINO reference attention, face-only identity modulation, RGB-erased
context, binary ownership and the existing differentiable identity auxiliary.
Native/empty-mask dispatch and the upstream patch are preserved.

Actual admission: 32 focused tests; pretrained parity/isolation/gradient/frozen
checks; all 19 training pairs with forced identity backward; exact 2-update
fresh-process optimizer/scheduler/RNG replay; one 20-step generation/decode
with exact exterior. The nearly untrained probe face is poor/patchy; no claim
of improved ID_sim has been established. See `docs/FLUX2_LOCAL.md` and
`docs/FLUX2_ADMISSION.json` for implementation scope and evidence hashes.

Actual failure: default allocation exceeded the 90% memory gate. With expandable
segments alone, accumulated training still reached 14.763671875 GiB reserved
(92.32%) and was rejected after one smoke update. Exact frozen K/V buffer CPU
offload reduced accumulated-training peak to 11.560546875 GiB (72.29%).
The accepted configuration retains full 768px training; no ROI fallback used.

Run `runs/FLUX2_local_4b_one_id_20261005`; 30,482,432 trainable parameters,
179 tensors, accumulation 4, LR 5e-5, identity weight .05, sigma≤.5, every 4
updates from the first update. Existing 19-pair/fixed24 manifests and metric
definitions remain unchanged. Serial validation at 0/500/1000/1500/2000.
Comet project `rsrch_new`, immutable key `7c88a5c362164fdbad2df18c2f629153`.
New ignored caches/weights/run artifacts reside on `/mnt/c/Users/ogure/rsrch_flux2`
because Linux had only ~250MB free. No historical artifacts were removed.


### 2026-10-05 — FLUX2 continuous Comet status and streaming validation images

AICODE-NOTE: User clarified that every batch means each **validation batch**,
not new generation during optimizer batches. Existing checkpoint schedule and
all numerical training code remain unchanged. Added `scripts/flux2_live.py` as
a separate execution overlay, reusing the tested streaming decoder. Original
run source hashes and checkpoint identities are retained. Continuous Comet
status publisher is active on the original key; per-batch decoding/upload is
queued to take over at the 1000-step checkpoint after the current training
worker finishes. Handoff never kills an in-progress optimizer segment.
Validation GPU use remains serial; Comet no longer ends between stages after
the live worker wrapper takes over. Upload failures retry outside the GPU worker.
Five focused lifecycle/publication/streaming/handoff tests passed. Live runtime
receipts are under the existing run; see `docs/FLUX2_LOCAL.md`.
