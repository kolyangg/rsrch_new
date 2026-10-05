#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${BA_ENVS_DIR:-$PWD/envs}/flux-toolkit/bin/python" -m scripts.run_flux1_experiment --experiment FLUX1c "$@"
