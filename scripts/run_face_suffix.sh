#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false COMET_DISPLAY_SUMMARY_LEVEL=0
python_bin="${BA_ENVS_DIR:-envs}/flux-toolkit/bin/python"
run_dir="${1:-runs/flux4b_face_one_id_strong_$(date -u +%Y%m%d_%H%M%S)}"
"$python_bin" -m scripts.face_suffix run --run-dir "$run_dir"
"$python_bin" -m scripts.review_face_diagnostic "$run_dir" --score --log-comet
