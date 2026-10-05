# FLUX1 deployment provenance

The original deployment snapshots below remain historical records. The October 4
cluster continuation replaces the encoder-GPU allocation with staged two-worker
DDP; use [current cluster operations](FLUX1_CLUSTER_DDP.md).

FLUX1 is the full-denoiser, masked Q/K/V/output BA mechanism. The user requested
this naming on 4 October 2026. Architecture and hyperparameters are unchanged by
the rename. `main` contains current source and renamed configs for fresh runs.

The two deployed runtimes differ: Vast used 9B/BF16; the cluster used 4B with
selective FP16 and a separate frozen encoder GPU. Their exact executed source
files were captured read-only and verified against each run’s immutable hashes.
They are preserved as separate Git branches, so ongoing maintenance and naming
changes do not invalidate existing checkpoint identities.

| Deployment | Frozen Git branch | Commit | Verified run-source files |
|---|---|---|---|
| vast | `FLUX1-vast-runtime` | `276fbf4e2fc8e75ec5eb38dd57044a4367587c35` | 23 |
| cluster | `FLUX1-cluster-runtime` | `8da85381b1c1f90ac086a269805267e7b932a7d1` | 26 |

Each archive branch includes `FLUX1_DEPLOYMENT.json` with file hashes and the
actual resolved config. Its historical config filenames and internal names are
intentionally unchanged. Download private data/weights separately and use the
same recorded revisions. A saved checkpoint’s portable `resume_config.yaml`
is the authoritative resume config; do not replace it with a renamed fresh-run
config. Historical run folders and Comet keys are preserved.

## Fresh-launch config names

Only filenames and config display names changed; all scientific fields are
unchanged. One-ID validation still uses 0/500/1000/2000. The active cluster
SBATCH file points at its FLUX1 config and a fresh FLUX1 run folder.

| Previous config | Current FLUX1 config |
|---|---|
| `configs/flux4b_oneid_online_face_qkvo_r128_768.yaml` | [`configs/FLUX1_local_4b_one_id.yaml`](../configs/FLUX1_local_4b_one_id.yaml) |
| `configs/flux4b_48_multi_id_large.yaml` | [`configs/FLUX1_vast_4b.yaml`](../configs/FLUX1_vast_4b.yaml) |
| `configs/flux4b_cosmic_hardware15.yaml` | [`configs/FLUX1_vast_4b_cosmic_smoke.yaml`](../configs/FLUX1_vast_4b_cosmic_smoke.yaml) |
| `configs/flux9b_gb10_large_qkvo_r128.yaml` | [`configs/FLUX1_vast_9b_batch8.yaml`](../configs/FLUX1_vast_9b_batch8.yaml) |
| `configs/flux9b_gb10_large_qkvo_r128_fast5h.yaml` | [`configs/FLUX1_vast_9b.yaml`](../configs/FLUX1_vast_9b.yaml) |
| `configs/clust/flux4b_2v100_amp.yaml` | [`configs/clust/FLUX1_cluster_4b.yaml`](../configs/clust/FLUX1_cluster_4b.yaml) |
| `configs/clust/flux4b_2v100.yaml` | [`configs/clust/FLUX1_cluster_4b_fp32.yaml`](../configs/clust/FLUX1_cluster_4b_fp32.yaml) |
| `configs/clust/flux4b_bf16_fast_proposed.yaml` | [`configs/clust/FLUX1_cluster_4b_bf16_proposed.yaml`](../configs/clust/FLUX1_cluster_4b_bf16_proposed.yaml) |
| `configs/clust/flux4b_2v100_proposal.yaml` | [`configs/clust/FLUX1_cluster_proposal.yaml`](../configs/clust/FLUX1_cluster_proposal.yaml) |

## Sources and operating scripts

- Vast: `/workspace/rsrch_9b_8k`, run `flux9b_large_qkvo_r128_to8k`;
  `run_flux9b_training_first.py`, `continue_flux9b.py`, and the scoped finalization
  controllers. Its final 8000/full96 workflow and verified stop are recorded in
  [completion notes](../plans/260930/FLUX9B_8K_COMPLETION.md).
- Cluster: `/home/nasilaev/rsrch_new`, run
  `flux4b_clust_2v100_amp_qkvo_r128_20k_20261003`;
  `jobs/flux4b_clust_2v100_amp.sbatch`, `run_clust_v100.sh`,
  `run_multi_id_face_ba.py`, plus cluster Comet status/relay scripts.
- Architecture report: `reports/261003_online_face_ba/build_report.py`; FLUX1
  labels were added to all 26 pages and three standalone vector diagrams.
  Its one-ID 4B results and equations remain the original measured evidence.

This rename did not launch or alter training on either host. The archive
branches are reproducibility records, not branches to merge over main.

## Proposed FLUX1a / FLUX1b / FLUX1c, 4 October 2026

These are **new Vast9B experiments**, prepared but not launched or pretrained-
qualified. Start **FLUX1a** first. The shortcut audit and implementation are in
[the follow-up report](../reports/261004_FLUX1_review/FLUX1_shortcut_audit_and_next_experiments.pdf).
These names do not rename either historical FLUX1 deployment.

| Priority | Config and launcher | Change | Trainable parameters |
|---|---|---|---|
| 1 | [FLUX1a config](../configs/FLUX1a_vast_9b.yaml), [launcher](../scripts/run_FLUX1a.sh) | Isolated reference bank; binary face-token ownership; rank128; training-target identity auxiliary, weight0.05 at sigma<=0.5 | 33,554,432 |
| 2 | [FLUX1b config](../configs/FLUX1b_vast_9b.yaml), [launcher](../scripts/run_FLUX1b.sh) | FLUX1a with flow loss only, to establish whether the identity objective contributes | 33,554,432 |
| 3 | [FLUX1c config](../configs/FLUX1c_vast_9b.yaml), [launcher](../scripts/run_FLUX1c.sh) | FLUX1a with rank256/alpha256, to test extra BA capacity | 67,108,864 |

All three use the same frozen FLUX.2-klein Base9B, native attention operations,
conditioning, eight branch sites, Q/K/V/output adapters and original fixed96
panel. Each reference bank comes from a separate frozen image-only forward at
the current sigma, with zero target/text tokens. Face K/V are selected after
that pass; reference-image context can still inform those features. Native
target queries, residuals, MLPs and the original native conditioning remain.
Binary token ownership switches each token touching the existing face-mask
support fully to BA; the original pixel feathering only composes the exterior
after decoding. There is no proposed native/reference attention-output fusion.

Training is fresh BA initialization, not a resume of old FLUX1: 4,000 updates,
seed142, unchanged full Large pairs/order, microbatch1, AdamW lr5e-5, warmup100,
gradient clipping1, save every500. Validate the complete96 at0/2000/4000 using
the same prompts, references, seeds, frozen native images/masks,768px,20 steps,
CFG4, batch2 and original ID/CLIP definitions. Keep Comet keys separate for
each new run under `rsrch_new`; save them in `comet_experiment.json`.

Use a **separate checkout**, for example `/workspace/rsrch_FLUX1abc`, with the
pinned AI Toolkit commit and this checkout's recorded patch. Do not update
`/workspace/rsrch_9b_8k` while its historical evaluation runs. Reuse verified
weights, environments and conditioning caches via explicit paths; copy the
full training manifest and its import audit with valid image paths. Keep new
identity supervision and new runs in the new checkout. FLUX1a/c require the
existing `onnx==1.23.1` auxiliary dependency in the FLUX environment and
InsightFace/buffalo_l in the metrics environment. No V100 launch is provided.

```bash
# These commands are for the prepared, separate GB10 checkout.
# No --action means plan only: no CUDA, data reads or writes.
scripts/run_FLUX1a.sh

# CPU preparation uses only training targets, never validation target photos.
scripts/run_FLUX1a.sh --action prepare

# Launch only when ready to start the new experiment on that checkout.
scripts/run_FLUX1a.sh --action run \
  --native-bundle /workspace/rsrch_9b_8k/runs/flux9b_8000_fixed96_20261004

# Later ablations: b needs no ID preparation; c reuses a's verified labels.
scripts/run_FLUX1b.sh --action run \
  --native-bundle /workspace/rsrch_9b_8k/runs/flux9b_8000_fixed96_20261004
scripts/run_FLUX1c.sh --action run \
  --native-bundle /workspace/rsrch_9b_8k/runs/flux9b_8000_fixed96_20261004
```

The controller runs pretrained BA-off/zero-mask parity, finite branch and
identity gradients, all64 parameter updates, largest-layout memory stress and
exact fresh-process checkpoint replay before long training. The recipes require
peak reserved memory below85%, stricter than the project90% ceiling. It fails
closed on changed source/data/weights, inadequate identity-label coverage
(<90%), missing fixed96 artifacts or failed admission. No automatic precision,
batch, resolution or mask fallback is allowed. Passing CPU tests does not
establish that the new pretrained configuration fits GB10 or improves quality.

Run the separate eight-image causal diagnostic after a completed validation
when the GPU is free. It picks the first fixed96 sample for each identity and
holds the native stream, prompt, mask, noise and sampler fixed across BA-off,
own-bank and different-person-bank arms. Donor tokens enter only the isolated
bank. This diagnostic never replaces or logs over the full96 metrics.

```bash
envs/flux-toolkit/bin/python -m scripts.probe_flux1_reference generate \
  --run runs/FLUX1a_vast_9b --step 2000 \
  --output runs/FLUX1a_vast_9b/probe_swap8_002000
envs/metrics/bin/python -m scripts.probe_flux1_reference score \
  --output runs/FLUX1a_vast_9b/probe_swap8_002000
```

Promote based on paired full96 target-ID improvement across identities, visible
pose/expression preservation and reference-swap causality. Lower similarity to
native alone is not success. ArcFace is both the auxiliary and the existing ID
metric, so compare an independent recognizer and inspect all96 images before
claiming general identity gains. Runtime/memory/quality for these proposals
remain unmeasured until pretrained admission and the first complete panel.

### Authorized FLUX1a launch

On4October the user requested starting FLUX1a on existing GB10 instance53994096.
Supervisor `rsrch_flux1a` now owns its serial startup/training pipeline in
`/workspace/rsrch_FLUX1abc`, run `FLUX1a_vast9B_20261004`, Comet
`2018ec7a730243bc98d922178e58aa5c`. The GPU was free after the completed2k replay.
See `MACHINES.md` for the latest measured stage; launch registration alone does
not establish that optimizer updates have started.

CPU identity preparation was optimized without changing training: four worker
processes, two ONNX threads each, and `--scheduled-only` covering every sample
that the exact trainer schedule will consume. The complete47,341-pair training
manifest/order remains authoritative. The supervision manifest records its
scheduled scope and full-manifest hash; loading rejects incomplete coverage
for the requested trajectory. Admission's two-update replay is covered by the
same labels. Extending training beyond the prepared trajectory requires labels
for the additional samples. Earlier full-preparation receipts remain archived.

The overlay environment adds the already tested ONNX1.23.1, protobuf7.36.2 and
ml_dtypes0.5.4 without changing packages in the historical FLUX environment.
The parallel preparer produced exactly matching embeddings and landmark
geometry on the two-target CPU regression. Historical validation publication
was repaired by a separate filename-normalization helper; native and BA image
generation and metric definitions were not changed.

FLUX1a admission is now measured on GB10: BA-off parity, branch/identity gradients,
all64 parameter updates, exact checkpoint replay and memory checks passed;
peak reserved60.67GiB (49.88%). At12:00UTC the controller was generating the
required full96 step0 panel. `scripts.publish_flux1a_progress` runs as separate
supervisor `rsrch_flux1a_progress` and records live stage/optimizer/validation
counts in Comet's `runtime/live_progress` and `runtime/*` metrics. It does not
modify or bypass the serialized training/validation workflow.

For the active GB10 FLUX1a deployment, supervisor rsrch_flux1a now starts
scripts.flux1a_live. Keep its paired rsrch_flux1a_progress process alive: it owns
the Comet session across generation, scoring and training, and streams stage
console output. Worker SDK sessions flush without ending the parent experiment.
Validation now decodes/uploads every completed pair during inference, with
live_decode_<step>.json acknowledged image receipts. The frozen original
controller still schedules all96 scoring at2000/4000 and optimizer continuation.
Source snapshots and deployment hashes are preserved under the run setup folder.
