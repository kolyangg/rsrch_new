# Stronger local face BA — 2026-10-01

## Scope and architecture

The requested experiment increases the simple face branch from one final
single-stream block, rank 16, to the last eight blocks (12–19), rank 256.
There are **12,607,496 trainable parameters**, all in BA: eight low-rank output
projections and eight learned scalar face routers. Native attention projections,
MLPs, normalization, output layers, text encoder and VAE remain frozen.

Each site reads only the selected reference-face keys/values with the native
target queries, native key normalization and original RoPE. A zero-initialized
LoRA output and query-predicted sigmoid gate produce a target-token correction
before the native output projection. Gamma remains 1; increasing branch count
and rank supplies capacity without an artificial inference gain. Both CFG lanes
use the same trained branches. Initialization and BA-off preserve native output.

Target boxes supervise the routers and face-balanced flow loss during training.
They are never supplied to inference. Learned gates can activate outside faces;
downstream denoising can also change background pixels.

## Efficiency and correctness

The frozen prefix ends immediately before single block 12. Its exact complete
text/target/reference sequence, RoPE, modulation, and final conditioning vector
are cached. All eight following blocks are recomputed on every update, including
their reference and text states. Frozen weights after an adapter remain in the
autograd graph, so earlier BA sites receive gradients through later blocks.
No stale per-block attention messages are reused.

The optimizer is fused AdamW. Branch parameters and router arithmetic are FP32;
native weights/activations are BF16, with TF32 allowed for FP32 matrix products.
CPU pinned caches avoid retaining the full bank on the GPU. Microbatches group
noise variants of one pair, preserving that pair's exact reference token layout.
Batch sizes 1, 2 and 4, with and without activation checkpointing, are measured
in separate processes. The best measured samples/second below 90% CUDA reserved
memory is chosen. A higher batch after a failed smaller one is skipped.

Critical checks cover native initialization, first nonzero output gradient at
every BA site, finite gradients, frozen native weights, checkpoint reload,
exact process-resume update 26, and trained cached/full-model prediction parity.
The focused small-model gradient/checkpoint checks passed; pretrained-model
results and measured throughput will be recorded after the run.

## Named diagnostic and reproducibility

- Config: `configs/flux4b_face_one_id_strong.yaml`.
- Modules: `ba_dit/face_suffix.py`, `scripts/face_suffix.py`.
- Launch: `bash scripts/run_face_suffix.sh runs/flux4b_face_one_id_strong_NAME`.
- 19 cross-view pairs, one identity; target 768 px, reference 512 px.
- Six sigmas: 0.05, 0.15, 0.3, 0.5, 0.7, 0.9; two fitted noise seeds and one
  separate probe seed per pair/sigma, giving 228 fitted and 114 probe inputs.
- Face/background flow-MSE means have equal aggregate weight; the router loss
  is class-balanced BCE averaged across all eight sites. This differs from the
  original whole-image MSE objective and is explicitly a separate experiment.
- Peak LR 0.0003, warmup 100, cosine decay to 10%, clipping 1, 2,000 updates.
- Fixed probes use all 19 pairs at sigma 0.5, one fitted and one separate seed.
- Four generated validation prompts at 768 px / 20 steps / CFG 4; unchanged
  one-ID order, prompts, references and seeds. Validate at 0, 1,000 and 2,000.
  This four-image panel is separate from the original 96-image evaluation.
- Step 25 checkpoint/resume admission, then checkpoints every 500 updates.
- Comet startup/key, scalar losses, progress/ETA, images and report retained.

The cache is a finite input bank: its speed must not be reported as fresh-noise
full-model training throughput. The frozen suffix, input hashes, full source
identity, exact upstream/weight revisions and selected runtime settings are
saved in the run. Existing final-block and original multi-site checkpoints
remain separate and retain their source identities.

## Current run

`runs/flux4b_face_one_id_strong_20261001`, Comet
[7beb18408145451896ccdeaa9a6722bb](https://www.comet.com/nikolay-2104/rsrch-new/7beb18408145451896ccdeaa9a6722bb).
The exact 342-input prefix cache completed in 301.09 seconds at 8.461 GiB peak
CUDA reserved memory. Full/native/cached predictions at the first and last
inputs matched exactly. The visual training-mask audit covers all 19 pairs;
all masks cover the intended faces after the 768 px crop.

Measured local optimizer benchmarks (five timed updates after two warmups):

| Batch | Activation checkpointing | Seconds/update | Samples/second | Peak reserved GiB | Result |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Off | 0.600 | **1.667** | 7.033 | Selected |
| 2 | Off | 1.400 | 1.428 | 11.641 | Fits, slower |
| 4 | Off | — | — | 21.027 | Rejected by memory gate |
| 1 | On | 0.945 | 1.058 | 4.158 | Fits, slower |
| 2 | On | 2.134 | 0.937 | 6.207 | Fits, slower |
| 4 | On | 5.424 | 0.737 | 9.855 | Fits, slower |

The batch-4 trial allocated beyond physical VRAM under WSL and was rejected;
it was not admitted for long training. Filling more memory did not improve
throughput on this GPU. Runtime selection is saved in `execution.json`.
All four baseline images and CPU ID/CLIP scoring completed: identity 0.306791,
CLIP 28.454863. Checkpoint 25 reloaded exactly; update 26 after a fresh-process
resume is bit-identical to the saved uninterrupted update. All eight BA output
matrices updated and native gradients remained absent.

At the final launch check (2026-10-01 11:31 UTC), **374/2,000 updates** had
completed. Every recorded loss/gradient norm was finite; peak training reserved
memory was **7.145 GiB** (roughly 8 GB total GPU usage). Recent sustained speed
was **1.156 seconds/update**, giving about **31 minutes of optimizer time left**,
plus two serial four-image validations and scoring. The driver reported active
software thermal slowdown at 88°C, with SM clocks around 600–735 MHz and about
98% GPU utilization. This explains why the short benchmark is faster than the
sustained rate. Evidence is saved in `thermal_throttling.txt` and
`launch_audit.json`.

The verified launcher remains running and will perform validation at 1,000,
resume to 2,000, then perform final validation/scoring and write `review.html`.
Active monitoring stopped after confirming healthy batches, consistent with
the user's earlier request. Final gains and completion have not yet been
established. The existing Vast instance is not used by this local experiment.
