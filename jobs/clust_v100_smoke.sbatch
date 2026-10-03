#!/usr/bin/env bash
# Named short training check while the approved two-V10020k run waits.
#SBATCH --job-name=rsrch-v100-smoke
#SBATCH --account=proj_1892
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/nasilaev/rsrch_new
#SBATCH --output=/home/nasilaev/rsrch_new/logs/clust/%x-%j.out
#SBATCH --error=/home/nasilaev/rsrch_new/logs/clust/%x-%j.err
set -euo pipefail
export BA_ROOT="$PWD" BA_ENVS_DIR="$PWD/envs/clust-v100"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 COMET_DISPLAY_SUMMARY_LEVEL=0
export PATH="$HOME/.local/bin:$PATH"
# A clean machine can prepare inside this allocation before the short probe.
if [[ ! -x "$BA_ENVS_DIR/flux-toolkit/bin/python" || ! -f weights/flux_vae/ae.safetensors ]]; then
  if ! command -v uv >/dev/null; then
    curl -fLsS --retry 2 --max-time 120 https://astral.sh/uv/0.11.12/install.sh -o scratch/install-uv.sh
    sh scratch/install-uv.sh
  fi
  bash scripts/setup_clust_v100.sh --download-weights
fi
exec "$BA_ENVS_DIR/flux-toolkit/bin/python" -m scripts.clust_v100_smoke
