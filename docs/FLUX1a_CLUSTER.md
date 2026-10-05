# FLUX1a on the HSE cluster

The user requested a fresh FLUX1a run on two V100s for20,000 optimizer updates,
with the original fixed96 validation at0 and every2,000 updates. This is the4B
cluster variant; the separately deployed Vast variant uses9B. Existing FLUX1
training is unchanged.

Recipe:4B Base backbone,768px target/512px reference,rank128 Q/K/V/output
adapters (25,165,824 trainable parameters), isolated frozen reference-image
pass, binary face-token ownership, native face-normalized flow loss plus the
training-target identity auxiliary(weight0.05, sigma<=0.5). Each V100 trains on
one distinct sample; DDP averages gradients, giving effective batch2. V100
precision is selective FP16 computation with FP32 master/branch/residual paths;
VAE/recognizer identity gradients remain differentiable. Adapters initialize
fresh; this does not resume FLUX1's checkpoint.

Config: `configs/clust/FLUX1a_cluster_4b.yaml`.
Executor: `scripts/clust_flux1a.py`; Slurm wrapper: `jobs/flux1a_clust_stage.sbatch`.
Isolated deployed root: `/home/nasilaev/rsrch_new_FLUX1a`.
Run: `runs/FLUX1a_cluster_4b_20261004`; setup has the `_setup` suffix.
The activated FLUX environment has a private training overlay for ONNX1.23.1,
protobuf7.36.2 and ml_dtypes0.5.4. Original environments/checkouts are untouched.
The pinned Toolkit checkout is independently patched. `deployment.json` records
the base Git commit, Toolkit pin and207 actual deployed file hashes.

Training requires CPU-prepared identity labels for all40,000 scheduled sample
positions, exact fixed96 conditioning parity against the existing4B native
bundle, pretrained native/BA-off parity, finite branch and forced identity
gradients, a worst-layout memory probe, and exact fresh-process two-worker
parameter/optimizer/scaler/RNG/cursor replay. These gates must pass before the
initial fixed96 panel and production training. CPU synthetic tests are not
pretrained V100 admission.

CPU identity preparation uses16cores in `cpu-e-quick`, with atomic per-image
receipts for safe Slurm preemption/requeue. GPU conditioning and serial validation
use one V100; training uses two V100s/four CPUs; metric scoring uses CPUs only.
The user QoS allows100 outstanding jobs. The finite79-stage plan therefore
submits one successor with `afterok` after each successful stage, rather than
reserving the entire queue. A failure stops the chain; there is no automatic
failed-training resubmission. The pipeline JSON records submitted job IDs and all
remaining stages through20k. Do not submit a second pipeline.

The primary Slurm log prints status every30seconds, training loss/rate and
training ETA, per-stage validation counts/ETA, and completed validation metrics.
Detailed framework output is preserved under setup/diagnostics. ETAs exclude
Slurm queue delay; the first few measurements are estimates. CPU label preparation
also prints counts/ETA without library initialization spam.

Comet: [FLUX1a_cluster_4b_2V100](https://www.comet.com/nikolay-2104/rsrch-new/9a4c6ef2e160413dad596a3dea2dc451).
The immutable key is9a4c6ef2e160413dad596a3dea2dc451. The workstation user service
`rsrch-clust-FLUX1a-comet-4375612` publishes status/ETA, every second training
update, all96 images per decoded validation, comparison panels and original
ID/CLIP/face-quality summaries. Full per-update metrics stay in `metrics.jsonl`.
Heartbeat publication runs independently of uploads. Console output contains
concise statuses instead of SSH banners or offline SDK archives. The relay needs
this workstation and its VPN to remain available; Slurm training itself does not.

Current startup evidence and failures are recorded in
`plans/260930/IMPLEMENTATION_NOTES.md`. Preparation4375612 is running; next
plan job4375658 waits for it. GPU admission and production steps are not yet
verified. Monitor with:

```bash
source /home/nasilaev/rsrch_new/scripts/activate_clust_env.sh
cd /home/nasilaev/rsrch_new_FLUX1a
source envs/training/bin/activate
export BA_ROOT="$PWD" PYTHONPATH="$PWD"
python -m scripts.clust_staged status --run runs/FLUX1a_cluster_4b_20261004
```

Latest startup check: labels19,544/40,000, estimated15.7min remaining. The current
two-V100 scheduler dry-run estimate is7October02:23MSK; it can change and is not
an allocation. GPU admission and training remain pending.

2026-10-04 recovery: original admission4375704 failed on VAE identity-backward
CUDA OOM before production initialization. The identity objective now enables
native decoder block checkpoints as well as the outer decode checkpoint.
Replacement4375739 (two V100s) follows memory probe4375743 (`test`,one V100), then
continues the saved pipeline only after successful admission. Both were pending
at18:10MSK; no production updates or measured post-fix V100 peak are claimed yet.
The same Comet experiment is retained; see structured `cluster/slurm_state` and
`cluster/stage` to distinguish queue/preflight from training. Transport failures
are now marked explicitly as stale monitoring. Full evidence is in the
implementation notes and ignored `runs/FLUX1a_cluster_launch` receipts.

Recovery measurement at18:13:31MSK: memory probe4375743 passed on V100 with
23.14GiB peak CUDA reserved (72.9%), finite identity/flow gradients, adapter
updates, unchanged frozen weights and worst-routing stress. Two-V100
admission4375739 is now pending Priority; latest estimate21:29MSK (19:29BST).
Two-rank replay and production progress are not yet verified for this recovery.

2026-10-04 streaming validation: cluster inference now decodes each completed
latent immediately and atomically exposes its PNG. The workstation relay uploads
complete images while the next sample is generated; it never waits for the full
panel or runs network requests on a GPU worker. The later decode stage verifies
all96 saved images. The V100 execution check4376019 passed exact original pixels
with the4B backbone resident,17.54GiB peak (55.3%). See
`scripts/clust_stream_decode.py` and `plans/260930/IMPLEMENTATION_NOTES.md`.
Comet deduplication now uses actual logical names and persistent acknowledgements;
new uploads include checkpoint-specific names and retain sample IDs/step metadata.


October5: initial fixed96 generation/scoring completed. Its summary failed once
because RelayExperiment lacked log_metric; the method was added and CPU retry
4376696 completed successfully. Conditioning4376685 and two-V100 training4376693
are submitted; training depends only on conditioning. No admission replay is
requested. The CPU scoring/summarization branch can finish independently.
Future inference allocations also run CLIP and all four quality models on their
V100 after releasing the backbone, using the verified CUDA scoring environment
shared with FLUX1. The fixed96 order/images/definitions are unchanged.
See FLUX1_CLUSTER_DDP.md for actual GPU scoring speed/memory and fallback rules.
