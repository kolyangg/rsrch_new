#!/usr/bin/env bash
# Dependency installation only. Run heavy checks/model loading inside Slurm.
set -euo pipefail
cd "$(dirname "$0")/.."
export BA_ROOT="$PWD"
export BA_ENVS_DIR="${BA_ENVS_DIR:-$BA_ROOT/envs/clust-v100}"
export PATH="$HOME/.local/bin:$PATH"
[[ $# -eq 0 || ( $# -eq 1 && "$1" == --download-weights ) ]] || exit 2
command -v uv >/dev/null || { echo 'Install uv, then rerun this script.' >&2; exit 1; }
bash scripts/clone_sources.sh flux
uv python install 3.11.13
python_bin="$BA_ENVS_DIR/flux-toolkit/bin/python"
[[ -x "$python_bin" ]] || uv venv --python 3.11.13 "${python_bin%/bin/python}"
uv pip install --python "$python_bin" torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$python_bin" -c locks/flux-v100-constraints.txt \
  -r locks/flux-requirements.txt \
  'diffusers @ git+https://github.com/huggingface/diffusers.git@c943837899b16cbae2f619b8dd4f7bb6f07dd81a' -e .
uv pip check --python "$python_bin"
mkdir -p scratch/clust-v100
uv pip freeze --python "$python_bin" > scratch/clust-v100/installed.txt
CUDA_VISIBLE_DEVICES='' "$python_bin" scripts/check_invariants.py flux
bash scripts/setup_metrics.sh
bash scripts/setup_face_quality.sh
if [[ "${1:-}" == --download-weights ]]; then
  "$python_bin" scripts/weights_manifest.py download --lock locks/weights-flux48.json --weights-dir weights
fi
echo "Dependencies prepared in $BA_ENVS_DIR; run GPU admission inside the approved Slurm job."
