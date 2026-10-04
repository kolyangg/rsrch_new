#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false COMET_DISPLAY_SUMMARY_LEVEL=0
exec "${BA_ENVS_DIR:-$root/envs}/flux-toolkit/bin/python" -m scripts.face_crop_flow run \
  --query-residual --noise-skip --unscaled-noise --run-dir "${1:?Provide a fresh named run directory}"
