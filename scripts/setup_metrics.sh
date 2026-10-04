#!/usr/bin/env bash
set -euo pipefail
BA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv python install 3.11.13
env_dir="${BA_ENVS_DIR:-$BA_ROOT/envs}/metrics"
constraints="$BA_ROOT/locks/metrics-constraints.txt"
if [[ "$(uname -m)" == aarch64 ]]; then
  # Official ARM CPU wheels have the same release version without the x86 +cpu suffix.
  constraints="$(mktemp)"
  trap 'rm -f "$constraints"' EXIT
  sed -E 's/^((torch|torchvision)==[^+]+)\+cpu$/\1/' "$BA_ROOT/locks/metrics-constraints.txt" > "$constraints"
fi
[[ -x "$env_dir/bin/python" ]] || uv venv --python 3.11.13 "$env_dir"
uv pip install --python "$env_dir/bin/python" torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$env_dir/bin/python" numpy==1.26.4 Cython==3.0.12 setuptools==78.1.0 wheel==0.45.1
uv pip install --python "$env_dir/bin/python" -c "$constraints" --no-build-isolation insightface==0.7.3 onnxruntime==1.21.0 opencv-python-headless==4.11.0.86 Pillow==11.1.0 comet-ml==3.49.7 'clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1' -e "$BA_ROOT"
uv pip check --python "$env_dir/bin/python"
echo "Metrics environment ready: $env_dir"
