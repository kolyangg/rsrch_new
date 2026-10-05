#!/usr/bin/env bash
# Historical encoder-GPU allocation; disabled after TaskMaster cancellation.
# The frozen FLUX1-cluster-runtime branch preserves the original launcher.
# Use docs/FLUX1_CLUSTER_DDP.md and jobs/flux1_clust_stage.sbatch instead.
#SBATCH --job-name=FLUX1-clust-4b
#SBATCH --account=proj_1892
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/nasilaev/rsrch_new
#SBATCH --output=/home/nasilaev/rsrch_new/logs/clust/%x-%j.out
#SBATCH --error=/home/nasilaev/rsrch_new/logs/clust/%x-%j.err
set -euo pipefail
echo 'This encoder-GPU launcher is retired. Follow docs/FLUX1_CLUSTER_DDP.md for the staged two-worker continuation.' >&2
exit 64
