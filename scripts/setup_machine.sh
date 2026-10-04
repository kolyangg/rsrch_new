#!/usr/bin/env bash
# Fresh Linux/Vast host: isolated pinned environment, source seams and weight aliases.
set -euo pipefail
BA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1:?Usage: setup_machine.sh flux48|flux80|qwen48|qwen80 [--download-weights]}"
case "$profile" in
  flux48) backend=flux; env_name=flux-toolkit; weight_lock=weights-flux48.json ;;
  flux80) backend=flux; env_name=flux-toolkit; weight_lock=weights-flux80.json ;;
  qwen48|qwen80) backend=qwen; env_name=qwen21; weight_lock=weights-qwen.json ;;
  *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac
[[ $# -le 2 && ( $# -eq 1 || "$2" == --download-weights ) ]] || { echo 'Only --download-weights is accepted after the profile' >&2; exit 2; }
for program in git curl gcc; do
  command -v "$program" >/dev/null || { echo "Install the missing host dependency: $program (Ubuntu: apt-get install git curl build-essential)" >&2; exit 1; }
done
command -v nvidia-smi >/dev/null || { echo 'NVIDIA driver is required on the training host' >&2; exit 1; }
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
version_at_least() { [[ "$(printf '%s\n%s\n' "$driver" "$1" | sort -V | head -n 1)" == "$1" ]]; }
version_at_least 560.28.03 || { echo "Driver $driver is too old for the pinned CUDA 12.6 wheel" >&2; exit 1; }
if [[ "$backend" == flux ]]; then
  torch_version=2.13.0; vision_version=0.28.0
  if version_at_least 580.65.06; then cuda_wheel=cu130; else cuda_wheel=cu126; fi
else
  torch_version=2.10.0; vision_version=0.25.0
  if version_at_least 570.26.02; then cuda_wheel=cu128; else cuda_wheel=cu126; fi
fi
echo "NVIDIA driver $driver: PyTorch $torch_version+$cuda_wheel"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/0.11.12/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
"$BA_ROOT/scripts/clone_sources.sh" "$backend"
uv python install 3.11.13
env_dir="${BA_ENVS_DIR:-$BA_ROOT/envs}/$env_name"
if [[ ! -x "$env_dir/bin/python" ]]; then
  uv venv --python 3.11.13 "$env_dir"
fi
uv pip install --python "$env_dir/bin/python" "torch==$torch_version" "torchvision==$vision_version" \
  --index-url "https://download.pytorch.org/whl/$cuda_wheel"
"$env_dir/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(f"CUDA unavailable after PyTorch installation: {torch.__version__}, runtime {torch.version.cuda}")
print(f"CUDA check passed: torch {torch.__version__}, runtime {torch.version.cuda}")
PY
constraints="$BA_ROOT/locks/$backend-constraints.txt"
if [[ "$cuda_wheel" == cu126 ]]; then
  # Keep shared pins, replace only CUDA-specific wheel pins for older hosts.
  constraints="$(mktemp)"
  trap 'rm -f "$constraints"' EXIT
  sed -E '/^(cuda-|nvidia-|torch==|torchvision==)/d' "$BA_ROOT/locks/$backend-constraints.txt" > "$constraints"
  printf 'torch==%s+cu126\ntorchvision==%s+cu126\n' "$torch_version" "$vision_version" >> "$constraints"
fi
if [[ "$backend" == flux ]]; then
  diffusers='diffusers @ git+https://github.com/huggingface/diffusers.git@c943837899b16cbae2f619b8dd4f7bb6f07dd81a'
  uv pip install --python "$env_dir/bin/python" -c "$constraints" -r "$BA_ROOT/locks/flux-requirements.txt" "$diffusers" -e "$BA_ROOT"
else
  uv pip install --python "$env_dir/bin/python" -c "$constraints" -r "$BA_ROOT/locks/qwen-requirements.txt" -e "$BA_ROOT/sources/diffusers-qwen" -e "$BA_ROOT"
fi
"$env_dir/bin/python" "$BA_ROOT/scripts/check_environment.py"
CUDA_VISIBLE_DEVICES='' "$env_dir/bin/python" "$BA_ROOT/scripts/check_invariants.py" "$backend"
"$BA_ROOT/scripts/setup_metrics.sh"
"$BA_ROOT/scripts/setup_face_quality.sh"
if [[ "${2:-}" == --download-weights ]]; then
  "$env_dir/bin/python" "$BA_ROOT/scripts/weights_manifest.py" download --lock "$BA_ROOT/locks/$weight_lock" --weights-dir "$BA_ROOT/weights"
fi
printf 'Environment ready: %s\n' "$env_dir"
