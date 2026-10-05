#!/usr/bin/env bash
# Resource shape is supplied by scripts.clust_staged submit. Never reserve a
# second GPU during validation or any GPU during hashing/scoring/summarization.
#SBATCH --account=proj_1892
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --kill-on-invalid-dep=yes
set -euo pipefail
cd /home/nasilaev/rsrch_new_staged
source /home/nasilaev/rsrch_new/scripts/activate_clust_env.sh
export BA_ROOT="$PWD" PYTHONPATH="$PWD" BA_COMET_OFFLINE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 COMET_DISPLAY_SUMMARY_LEVEL=0 TOKENIZERS_PARALLELISM=false
# No long-lived idle logger allocation. The workstation relay handles Comet.
exec python -u -m scripts.clust_staged "$@"
