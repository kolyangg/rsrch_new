#!/usr/bin/env bash
set -euo pipefail

BA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$BA_ROOT"/{sources,envs,locks,weights,cache,scratch,runs}

clone_pinned() {
  local remote="$1" name="$2" ref="$3" dest
  dest="$BA_ROOT/sources/$name"
  [[ ! -e "$dest" ]] || { printf 'Refusing to overwrite: %s\n' "$dest" >&2; return 1; }
  git clone --filter=blob:none "$remote" "$dest"
  git -C "$dest" fetch origin "$ref"
  git -C "$dest" switch --detach "$ref"
  git -C "$dest" switch -c research/cl39-ba
  git -C "$dest" submodule update --init --recursive
  git -C "$dest" rev-parse HEAD > "$BA_ROOT/locks/$name.commit"
}

clone_pinned https://github.com/huggingface/diffusers.git diffusers-qwen fef717ffb01f407d2637584ed936c16db908587a
clone_pinned https://github.com/ostris/ai-toolkit.git ai-toolkit-flux ecee894ed2b1f3716d9d7326693061ec1a3105bb
git -C "$BA_ROOT/sources/diffusers-qwen" apply --check "$BA_ROOT/patches/qwen21_local_pairs_comet.patch"
git -C "$BA_ROOT/sources/diffusers-qwen" apply "$BA_ROOT/patches/qwen21_local_pairs_comet.patch"
git -C "$BA_ROOT/sources/ai-toolkit-flux" apply --check "$BA_ROOT/patches/flux2_reference_branch_and_offload.patch"
git -C "$BA_ROOT/sources/ai-toolkit-flux" apply "$BA_ROOT/patches/flux2_reference_branch_and_offload.patch"
for spec in 'QwenLM/Qwen-Image-2.1 qwen-release' 'black-forest-labs/flux2 flux2-official'; do
  read -r remote name <<< "$spec"
  dest="$BA_ROOT/sources/$name"
  [[ ! -e "$dest" ]] || { printf 'Refusing to overwrite: %s\n' "$dest" >&2; exit 1; }
  git clone --depth 1 "https://github.com/$remote.git" "$dest"
  git -C "$dest" rev-parse HEAD > "$BA_ROOT/locks/$name.commit"
done
