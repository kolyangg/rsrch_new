#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false COMET_DISPLAY_SUMMARY_LEVEL=0
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${BA_ENVS_DIR:-envs}/flux-toolkit/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo 'First run: bash scripts/setup_machine.sh flux48 --download-weights' >&2
  exit 1
fi
exec "$python_bin" -m scripts.run_multi_id_face_ba "$@"
