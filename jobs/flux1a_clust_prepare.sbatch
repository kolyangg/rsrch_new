#!/usr/bin/env bash
#SBATCH --account=proj_1892
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --job-name=FLUX1a-prepare
#SBATCH --output=/home/nasilaev/rsrch_new_FLUX1a/logs/FLUX1a-prepare-%j.out
#SBATCH --error=/home/nasilaev/rsrch_new_FLUX1a/logs/FLUX1a-prepare-%j.err
#SBATCH --requeue
set -euo pipefail
source /home/nasilaev/rsrch_new/scripts/activate_clust_env.sh
cd /home/nasilaev/rsrch_new_FLUX1a
export BA_ROOT="$PWD" PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec envs/clust-v100/metrics/bin/python -u -m scripts.clust_flux1a_prepare \
  --source /home/nasilaev/rsrch_new_staged/runs/FLUX1_cluster_4b_ddp_20261004 --workers 16
