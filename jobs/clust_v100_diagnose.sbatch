#!/usr/bin/env bash
# Separate numerical diagnosis; no optimizer updates or changed main-run settings.
#SBATCH --job-name=rsrch-v100-dtype
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
source scripts/activate_clust_env.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="$PWD/scratch/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"
for dtype in float16 float32; do
  "$BA_FLUX_PYTHON" -m scripts.diagnose_clust_v100 \
    --config runs/clust_v100_two_update_smoke_4373054/resolved_config.yaml \
    --output "runs/clust_v100_dtype_${SLURM_JOB_ID}/${dtype}.json" --dtype "$dtype"
done
# Only attempt the separately named FP32 probe if its native path is finite/exact.
"$BA_FLUX_PYTHON" - <<'PY'
import json, os
from pathlib import Path
report = json.loads(Path(f'runs/clust_v100_dtype_{os.environ["SLURM_JOB_ID"]}/float32.json').read_text())
assert report['native']['finite'] and report['native_off_exact']
PY
exec "$BA_FLUX_PYTHON" -m scripts.clust_v100_smoke --dtype float32
