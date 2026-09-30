#!/usr/bin/env bash
# Upstream Toolkit paired-LoRA plumbing smoke; this is not the matched BA trainer.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:?Pass 48 or 80 for the target GPU class}"
case "$PROFILE" in
  48) MIN_MIB=45000; CONFIG="$ROOT/configs/flux4b_upstream_smoke.yaml"; WEIGHTS="$ROOT/weights/flux4b/flux-2-klein-base-4b.safetensors" ;;
  80) MIN_MIB=75000; CONFIG="$ROOT/configs/flux9b_upstream_smoke.yaml"; WEIGHTS="$ROOT/weights/flux9b/flux-2-klein-base-9b.safetensors" ;;
  *) printf 'Expected GPU profile 48 or 80\n' >&2; exit 2 ;;
esac
GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if (( GPU_MIB < MIN_MIB )); then
  printf 'GPU has %s MiB; %s GB-class smoke requires at least %s MiB\n' "$GPU_MIB" "$PROFILE" "$MIN_MIB" >&2
  exit 1
fi
[[ -f "$ROOT/.env" ]] && { set -a; source "$ROOT/.env"; set +a; }
[[ -f "$WEIGHTS" ]] || { printf 'Model weights missing: %s\n' "$WEIGHTS" >&2; exit 1; }
[[ -f "$ROOT/data/pairs_export/mapping.json" ]] || { printf 'Run export_flux_smoke_pairs.py first\n' >&2; exit 1; }
[[ -x "$ROOT/envs/flux-toolkit/bin/python" ]] || { printf 'FLUX environment missing\n' >&2; exit 1; }
PY_INCLUDE="$("$ROOT/envs/flux-toolkit/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
export CPATH="$PY_INCLUDE${CPATH:+:$CPATH}"
cd "$ROOT/sources/ai-toolkit-flux"
"$ROOT/envs/flux-toolkit/bin/python" run.py "$CONFIG"
