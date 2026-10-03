#!/usr/bin/env bash
# Bounded configuration timing only: one training GPU plus one frozen encoder GPU.
# No production training or validation-panel change is launched by this probe.
#SBATCH --job-name=rsrch-v100-budget
#SBATCH --account=proj_1892
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100:2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/nasilaev/rsrch_new
#SBATCH --output=/home/nasilaev/rsrch_new/logs/clust/%x-%j.out
#SBATCH --error=/home/nasilaev/rsrch_new/logs/clust/%x-%j.err
set -euo pipefail
source scripts/activate_clust_env.sh
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$BA_ROOT/scratch/clust-v100/tmp-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec python -m scripts.benchmark_clust_budget --output "runs/clust_v100_budget_$SLURM_JOB_ID"
