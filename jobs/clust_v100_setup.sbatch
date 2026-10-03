#!/usr/bin/env bash
# CPU preparation prerequisite for the authorized two-V100 run.
#SBATCH --job-name=rsrch-v100-setup
#SBATCH --account=proj_1892
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/nasilaev/rsrch_new
#SBATCH --output=/home/nasilaev/rsrch_new/logs/clust/%x-%j.out
#SBATCH --error=/home/nasilaev/rsrch_new/logs/clust/%x-%j.err
set -euo pipefail
source scripts/activate_clust_env.sh
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TMPDIR="$PWD/scratch/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"
if ! command -v uv >/dev/null; then
  curl -fLsS --retry 2 --max-time 120 https://astral.sh/uv/0.11.12/install.sh -o scratch/install-uv.sh
  sh scratch/install-uv.sh
fi
exec bash scripts/setup_clust_v100.sh --download-weights
