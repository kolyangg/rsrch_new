#!/usr/bin/env bash
# CPU/GPU count and time are explicit in scripts.clust_flux1a's saved pipeline.
#SBATCH --account=proj_1892
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --kill-on-invalid-dep=yes
set -euo pipefail
source /home/nasilaev/rsrch_new/scripts/activate_clust_env.sh
cd /home/nasilaev/rsrch_new_FLUX1a
source envs/training/bin/activate
export BA_ROOT="$PWD" PYTHONPATH="$PWD" BA_COMET_RELAY=1 HF_HUB_OFFLINE=1
export BA_COMET_IDENTITY_FILE="$PWD/runs/FLUX1a_cluster_4b_20261004_setup/comet_experiment.json"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export BA_ORT_THREADS=2
exec python -u -m scripts.clust_flux1a "$@"
