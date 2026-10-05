# FLUX1 on two V100 training workers

The October 4 continuation repairs the allocation that TaskMaster cancelled
(job 4374072, reported idle 00:03–03:03 MSK). The old job placed the full denoiser
on GPU 0 and a frozen encoder on GPU 1. The encoder took 0.336 seconds of a 3.619
second update. Validation used only GPU 0; CPU hashing/scoring retained both
GPUs. That allocation did not provide two training workers.

The new executor uses two DDP workers: each V100 holds the same denoiser and
trains on a different pair. Gradients are averaged before the optimizer update.
Microbatch 1 per GPU and accumulation 1 give effective batch 2. This is an explicit
batch/world-size transition at checkpoint 1000, not an exact continuation of
batch 1. The original sample stream, model, loss, LR, optimizer/scaler, 768 px target,
512 px reference, rank 128 and 20,000-update endpoint are preserved. Precision stays
selective FP16 with FP32 master/branch/residual operations.

| Stage | V100s | CPU cores | Work |
|---|---:|---:|---|
| Prepare | 0 | 1 | Hash all original data once and verify the parent checkpoint |
| Conditioning | 1 | 2 | Serial frozen encoder and VAE; cache only the next training window |
| DDP admission/training | 2 | 4 | Two denoisers, two distinct samples per update |
| Fixed96 inference/decode | 1 | 2 | Original patched backend, masks, prompts, seeds and order |
| Scoring | 0 | 4 | Original ID/CLIP and face-quality metrics |
| Summarize | 0 | 1 | Panel summaries and deletion of consumed continuation-owned caches |

Each stage is a separate foreground Slurm job in `rocky`, account `proj_1892`,
with `afterok` dependencies and cancellation on an invalid dependency. A failed
stage stops the chain; there is no automatic resubmission. Queue delay is not
part of an execution-time estimate. A precomputed full-dataset cache is avoided:
the maximum window is 4,000 pairs (2,000 updates), with a 100 GiB admission cap and
a free-space check. Validation tensors remain cached.

## Existing continuation

- Parent runtime: `/home/nasilaev/rsrch_new`, frozen Git archive
  `FLUX1-cluster-runtime` (`8da85381b1c1f90ac086a269805267e7b932a7d1`).
- Parent run: `runs/flux4b_clust_2v100_amp_qkvo_r128_20k_20261003`.
- Isolated patched runtime: `/home/nasilaev/rsrch_new_staged`.
- New run: `runs/FLUX1_cluster_4b_ddp_20261004`.
- Its `resolved_config.yaml` is generated from the actual parent config.
- The pipeline receipt in `scratch/pipeline_FLUX1_cluster_4b_ddp_20261004.json`
  records every submitted job and its resource request. Inspect it before any
  recovery; do not submit a duplicate chain.

```bash
cd /home/nasilaev/rsrch_new_staged
source /home/nasilaev/rsrch_new/scripts/activate_clust_env.sh
export BA_ROOT="$PWD" PYTHONPATH="$PWD"
python -m scripts.clust_staged status --run runs/FLUX1_cluster_4b_ddp_20261004
```

For a separate authorized continuation, `submit --run NEW_RUN --parent PARENT
--step CHECKPOINT_STEP --after STOPPED_PARENT_JOB_ID` submits the bounded chain.
The parent must be the measured batch 1/online-conditioning configuration and
must have a complete checkpoint before preparation runs. Never overwrite its
config or checkpoints. Changing execution code after preparation requires an
explicit audited repair/new preparation; run source hashes fail closed.

## Admission and observability

Before production DDP, all 96 frozen validation text/VAE tensors must match the
parent cache bit for bit. Two real pretrained DDP updates must have finite
gradients, change every adapter tensor, stay below 90% reserved memory on both
devices, and exactly match a fresh-process one-update/save/reload/one-update
execution, including full optimizer/scaler/per-rank RNG/data-cursor state.
`conditioning_parity.json` and `execution_admission.json` record the result.
No synthetic test is a substitute for these gates.

The immutable Comet experiment remains
[5d31de48010446248639e65a65236cbe](https://www.comet.com/nikolay-2104/rsrch-new/5d31de48010446248639e65a65236cbe).
The workstation service `rsrch-clust-ddp-comet-4375337` runs
`scripts.sync_clust_comet --pipeline --remote-root /home/nasilaev/rsrch_new_staged`.
It publishes the actual stage/job and uses `ddp_4375151/train/loss` for the
continued curve, retaining old interrupted-attempt curves. Updates through 1000
are inherited batch 1 history; subsequent updates are batch 2. Full per-update
metrics remain in `metrics.jsonl`; Comet receives every second update to respect
its per-metric limit. Fixed96 validation is serial at every 2,000 updates.

The live Comet display label is **FLUX1_cluster_4b_2V100_DDP**; the original
display name is retained in metadata, and filesystem run/checkpoint names stay
unchanged. An acknowledged five-second heartbeat runs independently of uploads.

Admission4375081 failed before workers started because torchrun consumed the
worker's `--run` argument. The repaired launcher separates its options with `--`
and has a real two-worker CPU regression. Its audited source repair and original
identity are in the run's `execution_repairs/torchrun_separator` directory.
The replacement chain prepares the first window in4375267, then runs admission
and the first training segment together in4375268 using the `continue` action.
The saved pipeline receipt preserves the failed attempt separately. Training
still requires all original admission checks; this action removes one queue wait.

The old two-GPU encoder launcher is disabled in main. Do not copy current main
over a frozen deployed runtime: unrelated architecture work may invalidate its
checkpoint identity. Actual admission measurements and failures are recorded in
`plans/260930/IMPLEMENTATION_NOTES.md`.

Verified on2026-10-04 at12:02UTC: job4375268 oncn-004 passed the full real
pretrained save/resume admission and production reached1041. Both GPUs averaged
89.6%/86.8% utilization over60seconds, with peak CUDA reserved17.34GiB and
~3.04s/update. New Comet loss points were independently read back and matched
the saved metrics through1028. These are startup measurements, not a completed
20k run or a new fixed96 validation result.


October5 execution repair: the same fixed96 images are now scored on the
inference V100 after the sampling child exits and releases its backbone/VAE.
The separate CUDA scoring environment preserves PyTorch2.2.0, torchvision0.17.0,
PyIQA0.1.15, FP32 scoring and the original metric definitions. Only face detection
remains on CPU. CPU postprocessing jobs become short verification/no-op stages
after GPU score receipts exist. Next-window conditioning no longer depends on
CPU scoring; cache cleanup occurs before building the next window to prevent
concurrent summaries deleting active training inputs. Training/checkpoint hashes
and the original fixed96 order/resolution/steps remain unchanged.

V100 validation4376688: full96 face-quality evaluation82.07s, MANIQA45.45s
versus1352.88s CPU; peak27.94GiB, below90%. Full-panel CPU/GPU means/coverage
agreed within7.5e-6 absolute. These timings cover scoring, not image generation
or Slurm queue delays. GPU enablement receipt is
`envs/clust-v100/metrics-gpu/gpu_verified.json`; its absence retains CPU fallback.

The relay explicitly sends timestamped concise Comet console updates and keeps
uploads off compute workers. Run-level assets accept Comet's implicit step0;
changed metadata is published under a content-specific revision name instead of
repeating acknowledged uploads. The original experiment may retain a historical
metric-throttle flag; continuation loss points are checked against saved values.
