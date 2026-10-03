# Two-V100 FLUX4B deployment on clust

The prepared experiment uses FLUX.2-klein **Base 4B**, fresh rank128 Q/K/V/output
reference adapters at the eight tested sites, target768/reference512, and the
approved generated-mask spatial contract. It extends the promising local one-ID
experiment to all eligible Large identities; it does not continue one-ID weights.
The local fixed24 ID_sim peaked at0.430254 at1000 updates (native0.331399).
That result does not establish multi-ID generalization or V100 performance.

## Configuration and execution

- Executable configuration: `configs/clust/flux4b_2v100.yaml`.
- Slurm script: `jobs/flux4b_clust_2v100.sbatch`.
- Controller: `scripts/run_clust_v100.sh` → `scripts/run_multi_id_face_ba.py`.
- Training: two DDP replicas, microbatch1 per GPU, accumulation4, **global batch8**.
- LR5e-5, warmup100, 20,000 updates, checkpoint every500, fixed96 at0 and every2000 updates.
- FP16 frozen denoiser, dynamic loss scaling, FP32 adapters/optimizer/loss,
  FP32 frozen VAE on GPU and FP32 text encoder on CPU for each worker.
- One node, two V10032GB, account `proj_1892`, partition `rocky`,16 CPUs,7 days.
  Slurm advertises `RealMemory=1`; omit `--mem` and `--mem-per-cpu`.

Each GPU holds its own denoiser. This is an initial conservative batch choice,
not a measured maximum; two32GB GPUs do not create a64GB model device. CPU text
encoding preserves VRAM but may limit throughput. Allow approximately64GiB of
free host RAM for two encoder replicas and temporary loads; the allocation must
be inspected before a run. The7-day limit is an envelope, not a completion ETA.

Workers interleave a single deterministic shuffled data stream, average gradients,
and write checkpoints/Comet only on rank0. Checkpoints include the global cursor,
each rank's Python/CPU/CUDA RNG and GradScaler state. Bounded overflow retries
repeat the same samples/noise with a reduced scale. Nonfinite forward values,
persistent bad gradients, or reserved memory >=90% on either GPU stop the job.
A fresh-process exact replay check must pass before long training.

Training workers exit before each serial validation stage. Generation, native
controls and validation use the same patched backend and explicit precision.
FP16 native images/masks are regenerated under a distinct precision signature;
old BF16 masks/caches cannot be silently reused. The original96 order, prompts,
seeds, reference images and metric definitions remain fixed. Missing/ambiguous
native face boxes stop initialization for review. No target-photo masks enter
inference. The Comet project is `rsrch_new`, with a persisted experiment key.

## Data and transfer

Remote project: `/home/nasilaev/rsrch_new`.
[Validation paths](../configs/clust/validation.yaml) are verified: the existing
cluster reference images match all eight SHA256 values in the original panel.
Current local prompts, boxes and subject-v2/legacy metric embeddings were synced;
the older cluster copies of prompts/embeddings differed and are not used.

The historical paired Cosmic target tree under another user's home is inaccessible.
No access bypass was attempted. With the user's authorization, the complete local
adjusted Large dataset was copied and file-checksum verified at
`/home/nasilaev/datasets/large_dataset_adj/large_dataset`:47,500 images,
17,116,845,489 bytes. Metadata is verified at
`/home/nasilaev/datasets/metadata/filtered_ids3_adj.json`.

The import produced47,341 pairs, excluding110 held-out-identity pairs and49
content-duplicate pairs, with zero validation-image overlap. The remote-path
manifest is `data/train_pairs_large_clust.jsonl`, SHA256
`fea6c505016c73c995d25a65ca02202d27a2b622e9a463ac3096f8b1586c6302`.
The manifest and its audit are transferred as data, not committed.
[Training locations](../configs/clust/training_sources.yaml) record exact roots.

Four background rsync streams completed with zero file checksum/size/attribute
differences. Shared directory modification times differed during concurrent
writes; only those directory timestamps were excluded from file verification.
The verified manifest/audit and `data/clust_large_transfer.json` receipt are
uploaded. All47,500 files (17,116,845,489 bytes) are present; transfer is complete.
The launcher requires that receipt and the matching manifest hash before loading
models. Local progress/logs are in ignored `scratch/clust-sync-20261003/`.
No transfer remains active.

## Environment and later approved submission

Use the existing cluster/VPN skill. No new SSH configuration is needed.
The V100 installer isolates dependencies in `envs/clust-v100/`; it does not change
the existing workstation/Vast environments. Python3.11.13, PyTorch2.7.1+cu126 and
torchvision0.22.1+cu126 are pinned in `locks/flux-v100-constraints.txt`, along with
the existing Transformers and adapter dependencies. The existing Toolkit commit
and Diffusers commit are retained. Installed packages are recorded in
`scratch/clust-v100/installed.txt`. The allocated-node check requires actual
`sm_70` kernels and two V100s. [PyTorch's version table](https://pytorch.org/get-started/previous-versions/)
lists the selected CUDA12.6 release. CUDA13 removed Volta support; see
[NVIDIA's release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).

Prepare dependencies and pinned weights before submitting the GPU run; use an
appropriate CPU allocation for heavy setup checks. Keep credentials in an ignored
private `.env` or environment, never in Git or sbatch directives:

```bash
cd /home/nasilaev/rsrch_new
# One-time lightweight creation on login; heavy imports/checks run under Slurm.
source /opt/software/python/miniconda/latest/etc/profile.d/conda.sh
conda create -n rsrch_new --override-channels -c conda-forge python=3.11.13 git pip
source scripts/activate_clust_env.sh
# Download/install on login with one build worker; no model imports/inference.
UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 bash scripts/setup_clust_v100.sh --prepare-only
python scripts/prepare_clust_metric_weights.py
bash scripts/run_clust_v100.sh --run runs/flux4b_clust_preview --dry-run
```

After the user approves GPU execution, inspect current availability/account limits
and create log directories before submission:

```bash
mkdir -p logs/clust
sbatch --test-only jobs/flux4b_clust_2v100.sbatch
sbatch --parsable jobs/flux4b_clust_2v100.sbatch
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
scancel JOB_ID
# Resume only after diagnosing the prior failure/timeout:
sbatch jobs/flux4b_clust_2v100.sbatch --resume
```

The controller first checks data/revisions, caches only validation conditioning,
and runs pretrained native/BA-off, gradients, frozen-weight equality, per-GPU
memory and fresh-process two-rank replay admission. It then generates/scores the
native panel, initializes the immutable run, validates step0 and trains. Failures
retain logs and do not silently change dtype, backbone or resolution.

CPU regression checks establish batching/checkpoint mechanics only. No V100 model
fit, numerical stability, image quality or throughput has yet been measured, and
no GPU job has been submitted. Cluster environment/weights and Comet availability
must still be checked before the approved run.

## Preparation checks (3 October 2026)

The isolated local environment installed successfully and passed dependency
validation. Its compiled CUDA architecture list includes sm_70. The selected
commit snapshot passed seven existing FLUX invariants, masked-routing and native
background checks, plus the two-process CPU/Gloo global-gradient and fresh-process
checkpoint replay regression. CPU imports exercise the pinned full backend.
These are software checks, not pretrained V100 admission.

On clust, `bash -n` and `sbatch --test-only` accepted the exact two-V100 script
under proj_1892/rocky with16 CPUs on one node. Test-only returns a hypothetical
schedule; it does not submit a job, reserve GPUs or guarantee that start time.

## Authorized20k submission — 3 October 2026

The user authorized20,000 optimizer updates on two V100s, then requested any
available V100 for training sooner. Run:
`runs/flux4b_clust_2v100_fp16_qkvo_r128_20k_20261003`.
Training job4372978 requests two V100s/16 CPUs on rocky for up to7 days.

The initial CPU setup job4372977 remained queued and was cancelled before it
started. Its dry-run estimate of immediate availability was not realized.
A named one-V10030-minute setup/two-update smoke job4372995 was submitted in
test. It uses eight eligible training pairs and compares one original validation
conditioning item, logs a separate Comet key, and checks real branch gradients,
frozen/native parity and peak memory. It is not a quality/generalization run or
a substitute for two-rank admission. Training4372978 now depends on its success.
The full run retains all47,341 training pairs and fixed96 at0/every2000 updates.

Submission replies, exact source/config hashes and job IDs are recorded in
`runs/clust_20k_submission/` on clust. Inspect live queue/logs for current status;
no optimizer result was available when these jobs were submitted.

The user subsequently requested explicit environment activation and approved a
new environment after inspection found Python3.10 in the old PhotoMaker env.
All cluster launchers now activate `/home/nasilaev/.conda/envs/rsrch_new` and
disable user-site packages. The old environments stay unchanged. Git belongs to
the new Conda environment because the compute-node system image omits it.
Metrics retain their separate pinned project environments to avoid the known
Transformers/CLIP dependency conflicts; all training uses the activated FLUX env.
Job4372995 failed during the uv download after its site TMPDIR disappeared;
job4373014 failed before preparation because git was absent. Neither executed
optimizer updates. Job4372978 was held while the environment was corrected.

Job4373031 activated the new Conda env successfully but timed out connecting to
download.pytorch.org from cn-025. Download/install preparation therefore runs
on login with one build worker and skips model imports/checks; GPU admission
remains in Slurm. The 16.148GB pinned backbone/text/VAE files were downloaded
and verified successfully. A separate lightweight downloader verifies original
metric weights against SHA256 values from the local experiment. Compute jobs
use HF_HUB_OFFLINE=1 and cached files. No prepared-package or file check is a
pretrained GPU result. The held, never-started job4372978 was cancelled and
replaced by4373033 after the site's release command denied the hold release.
