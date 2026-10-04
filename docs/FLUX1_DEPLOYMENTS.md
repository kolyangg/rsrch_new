# FLUX1 deployment provenance

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
