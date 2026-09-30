#!/usr/bin/env bash
set -euo pipefail
BA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1:?Usage: run_profile.sh flux48|flux80|qwen48|qwen80 preflight|precompute|train|infer [options]}"
command="${2:?Choose preflight, precompute, train or infer}"
shift 2
case "$profile" in
  flux48) env_name=flux-toolkit; config=flux4b_48 ;;
  flux48-pilot12) env_name=flux-toolkit; config=flux4b_48_pilot12 ;;
  flux80) env_name=flux-toolkit; config=flux9b_80 ;;
  flux80-matched) env_name=flux-toolkit; config=flux9b_80_matched ;;
  qwen48) env_name=qwen21; config=qwen7b_48 ;;
  qwen80) env_name=qwen21; config=qwen7b_80 ;;
  qwen80-matched) env_name=qwen21; config=qwen7b_80_matched ;;
  *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac
case "$command" in preflight|precompute|train|infer|evaluate) ;; *) echo "Unknown command: $command" >&2; exit 2 ;; esac
cd "$BA_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
exec "${BA_ENVS_DIR:-$BA_ROOT/envs}/$env_name/bin/python" -m ba_dit.cli "$command" --config "$BA_ROOT/configs/$config.yaml" "$@"
