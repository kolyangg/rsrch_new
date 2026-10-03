#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate_clust_env.sh
export HF_HUB_OFFLINE=1
export BA_COMET_OFFLINE=1
# Reduce fragmentation from variable reference shapes; retain the 90% memory gate.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BA_ROOT="$PWD"
export BA_ENVS_DIR="${BA_ENVS_DIR:-$BA_ROOT/envs/clust-v100}"
export PYTHONPATH="$BA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false COMET_DISPLAY_SUMMARY_LEVEL=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="$BA_FLUX_PYTHON"
[[ -x "$python_bin" ]] || { echo 'Run scripts/setup_clust_v100.sh first.' >&2; exit 1; }
if [[ " $* " != *' --dry-run '* ]]; then
  "$python_bin" -m scripts.check_clust_v100
fi
# Worker count follows the config; validation stays serial.
# It never overwrites CUDA_VISIBLE_DEVICES supplied by Slurm.
exec "$python_bin" -m scripts.run_multi_id_face_ba \
  --config "${BA_CLUST_CONFIG:-configs/clust/flux4b_2v100.yaml}" "$@"
