#!/usr/bin/env bash
set -euo pipefail
BA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv python install 3.11.13
env_dir="${BA_ENVS_DIR:-$BA_ROOT/envs}/face-quality"
[[ -x "$env_dir/bin/python" ]] || uv venv --python 3.11.13 "$env_dir"
uv pip install --python "$env_dir/bin/python" torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$env_dir/bin/python" numpy==1.26.4 Cython==3.0.12 setuptools==78.1.0 wheel==0.45.1
# PyIQA depends on a different CLIP package; isolate it from the legacy ID/CLIP scorer.
uv pip install --python "$env_dir/bin/python" -c "$BA_ROOT/locks/face-quality-constraints.txt" --no-build-isolation pyiqa==0.1.15 insightface==0.7.3 onnxruntime==1.21.0 Pillow==11.1.0 comet-ml==3.49.7 opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86 -e "$BA_ROOT"
# Both distributions provide cv2. Finish with headless binaries for minimal Vast images.
uv pip install --python "$env_dir/bin/python" --no-deps --reinstall-package opencv-python-headless opencv-python-headless==4.11.0.86
uv pip check --python "$env_dir/bin/python"
echo "Face-quality environment ready: $env_dir"
