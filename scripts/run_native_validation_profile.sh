#!/usr/bin/env bash
# Fixed 96-panel native backbone validation on a 48/80 GB-class GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:?Usage: run_native_validation_profile.sh 48|80 flux|qwen [limit]}"
BACKEND="${2:?Pass flux or qwen}"
LIMIT="${3:-96}"
[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] && (( LIMIT <= 96 )) || { printf 'limit must be 1..96\n' >&2; exit 2; }
case "$PROFILE" in
  48) MIN_MIB=45000; TARGET=512; REF=512; FLUX_ARCH=4b ;;
  80) MIN_MIB=75000; TARGET=1024; REF=768; FLUX_ARCH=9b ;;
  *) printf 'Expected GPU profile 48 or 80\n' >&2; exit 2 ;;
esac
GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if (( GPU_MIB < MIN_MIB )); then
  printf 'GPU has %s MiB; %s GB-class validation requires at least %s MiB\n' "$GPU_MIB" "$PROFILE" "$MIN_MIB" >&2
  exit 1
fi
[[ -f "$ROOT/.env" ]] && { set -a; source "$ROOT/.env"; set +a; }
[[ -x "$ROOT/envs/qwen21/bin/python" ]] || { printf 'Qwen/Comet environment missing\n' >&2; exit 1; }
RUN_NAME="${BACKEND}_${PROFILE}gb_native_${TARGET}px_$(date -u +%Y%m%dT%H%M%SZ)_$$"
RUN_DIR="$ROOT/runs/$RUN_NAME"
case "$BACKEND" in
  flux)
    [[ -f "$ROOT/weights/flux${FLUX_ARCH}/flux-2-klein-base-${FLUX_ARCH}.safetensors" ]] || { printf 'FLUX %s weights missing\n' "$FLUX_ARCH" >&2; exit 1; }
    [[ -d "$ROOT/weights/flux${FLUX_ARCH}_text" ]] || { printf 'FLUX text encoder missing\n' >&2; exit 1; }
    PY_INCLUDE="$("$ROOT/envs/flux-toolkit/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
    export CPATH="$PY_INCLUDE${CPATH:+:$CPATH}"
    "$ROOT/envs/flux-toolkit/bin/python" "$ROOT/scripts/validate_native_flux.py" \
      --arch "$FLUX_ARCH" --width "$TARGET" --height "$TARGET" --reference-size "$REF" \
      --steps 50 --limit "$LIMIT" --output-dir "$RUN_DIR"
    ;;
  qwen)
    [[ -f "$ROOT/weights/qwen21/model_index.json" ]] || { printf 'Qwen weights missing\n' >&2; exit 1; }
    "$ROOT/envs/qwen21/bin/python" "$ROOT/scripts/validate_native_qwen21.py" \
      --target-size "$TARGET" --reference-size "$REF" --steps 40 \
      --limit "$LIMIT" --output-dir "$RUN_DIR"
    ;;
  *) printf 'Expected backend flux or qwen\n' >&2; exit 2 ;;
esac
"$ROOT/envs/qwen21/bin/python" "$ROOT/scripts/log_validation_comet.py" "$RUN_DIR" --run-name "$RUN_NAME"
