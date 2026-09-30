#!/usr/bin/env bash
set -euo pipefail
BA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend="${1:-all}"
[[ "$backend" == flux || "$backend" == qwen || "$backend" == all ]] || { echo 'Usage: clone_sources.sh [flux|qwen|all]' >&2; exit 2; }
mkdir -p "$BA_ROOT"/{sources,envs,weights,cache,scratch,runs}
clone_pinned() {
  local remote="$1" name="$2" patch="$3" ref dest
  ref="$(cat "$BA_ROOT/locks/$name.commit")"
  dest="$BA_ROOT/sources/$name"
  if [[ ! -e "$dest" ]]; then
    git clone --filter=blob:none --no-checkout "$remote" "$dest"
    git -C "$dest" fetch --depth 1 origin "$ref"
    git -C "$dest" checkout --detach "$ref"
  fi
  [[ "$(git -C "$dest" rev-parse HEAD)" == "$ref" ]] || { echo "Wrong source revision: $dest" >&2; exit 1; }
  if git -C "$dest" apply --reverse --check "$BA_ROOT/patches/$patch" 2>/dev/null; then
    echo "Pinned source and patch already present: $name"
  else
    git -C "$dest" apply --check "$BA_ROOT/patches/$patch"
    git -C "$dest" apply "$BA_ROOT/patches/$patch"
  fi
}
if [[ "$backend" == flux || "$backend" == all ]]; then
  clone_pinned https://github.com/ostris/ai-toolkit.git ai-toolkit-flux flux2_reference_branch_and_offload.patch
fi
if [[ "$backend" == qwen || "$backend" == all ]]; then
  clone_pinned https://github.com/huggingface/diffusers.git diffusers-qwen qwen21_local_pairs_comet.patch
fi
