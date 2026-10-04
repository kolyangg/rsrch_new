#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export COMET_DISPLAY_SUMMARY_LEVEL=0
"${BA_ENVS_DIR:-$root/envs}/flux-toolkit/bin/python" -m scripts.face_diagnostic run \
  --config "$root/configs/flux4b_face_one_id_fast.yaml" --run-dir "${1:?Provide a fresh named run directory}"
exec "${BA_ENVS_DIR:-$root/envs}/flux-toolkit/bin/python" -m scripts.review_face_diagnostic "$1" --score --log-comet
