#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# Frozen ArcFace ONNX execution in PyTorch; no ONNX Runtime on the training GPU.
uv pip install --python "$root/envs/flux-toolkit/bin/python" onnx==1.23.1
