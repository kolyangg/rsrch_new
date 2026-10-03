#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false COMET_DISPLAY_SUMMARY_LEVEL=0
python_bin="${BA_ENVS_DIR:-envs}/flux-toolkit/bin/python"
run_dir="${1:?Usage: run_online_face_ba.sh RUN_DIR [ADMISSION_DIR]}"
admission_dir="${2:-runs/online_qkvo_admission_20261002}"
if [[ ! -f "$run_dir/identity.json" ]]; then
  "$python_bin" -m scripts.online_face_ba init --run "$run_dir" --admission "$admission_dir"
fi
exec "$python_bin" -m scripts.run_online_face_ba --run "$run_dir"
