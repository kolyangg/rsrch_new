#!/usr/bin/env bash
# Dependency installation only. Run heavy checks/model loading inside Slurm.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate_clust_env.sh
export BA_ROOT="$PWD"
export BA_ENVS_DIR="${BA_ENVS_DIR:-$BA_ROOT/envs/clust-v100}"
export PATH="$HOME/.local/bin:$PATH"
[[ $# -eq 0 || ( $# -eq 1 && ( "$1" == --download-weights || "$1" == --prepare-only ) ) ]] || exit 2
command -v uv >/dev/null || { echo 'Install uv, then rerun this script.' >&2; exit 1; }
# Git is installed in rsrch_new because compute images omit the system binary.
git --version
# InsightFace builds a small C++ extension in the isolated metrics environments.
if ! command -v g++ >/dev/null; then
  [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
  module load gcc/14.3.0
fi
command -v g++
bash scripts/clone_sources.sh flux
python_bin="$BA_FLUX_PYTHON"
uv pip install --python "$python_bin" torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$python_bin" -c locks/flux-v100-constraints.txt \
  -r locks/flux-requirements.txt \
  'diffusers @ git+https://github.com/huggingface/diffusers.git@c943837899b16cbae2f619b8dd4f7bb6f07dd81a' -e .
uv pip check --python "$python_bin"
mkdir -p scratch/clust-v100
uv pip freeze --python "$python_bin" > scratch/clust-v100/installed.txt
if [[ "${1:-}" != --prepare-only ]]; then
  CUDA_VISIBLE_DEVICES='' "$python_bin" scripts/check_invariants.py flux
fi
bash scripts/setup_metrics.sh
bash scripts/setup_face_quality.sh
if [[ "${1:-}" == --download-weights || "${1:-}" == --prepare-only ]]; then
  "$python_bin" scripts/weights_manifest.py download --lock locks/weights-flux48.json --weights-dir weights
  "$python_bin" - <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
Path('scratch/clust-v100/setup-complete.json').write_text(json.dumps({
    'python': sys.executable, 'conda_prefix': os.environ['CONDA_PREFIX'],
    'completed_utc': datetime.now(timezone.utc).isoformat(),
    'weights_lock': 'locks/weights-flux48.json',
    'gpu_admission': 'not run; execute the Slurm smoke/admission checks',
}, indent=2)+'\n')
PY
fi
echo "Training environment: $CONDA_PREFIX; isolated metric tools: $BA_ENVS_DIR. Run GPU admission inside Slurm."
