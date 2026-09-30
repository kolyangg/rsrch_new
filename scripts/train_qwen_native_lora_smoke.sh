#!/usr/bin/env bash
# Upstream paired-LoRA plumbing smoke; BA research trainer is a separate work package.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:?Pass 48 or 80 for the target GPU class}"
case "$PROFILE" in
  48) MIN_MIB=45000 ;;
  80) MIN_MIB=75000 ;;
  *) printf 'Expected GPU profile 48 or 80\n' >&2; exit 2 ;;
esac
GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if (( GPU_MIB < MIN_MIB )); then
  printf 'GPU has %s MiB; %s GB-class smoke requires at least %s MiB\n' "$GPU_MIB" "$PROFILE" "$MIN_MIB" >&2
  exit 1
fi
[[ -f "$ROOT/.env" ]] && { set -a; source "$ROOT/.env"; set +a; }
export COMET_PROJECT_NAME=rsrch_new
export HF_HUB_OFFLINE=1
[[ -f "$ROOT/data/qwen_smoke_dataset/dataset_info.json" ]] || { printf 'Run export_qwen_smoke_dataset.py first\n' >&2; exit 1; }
[[ -f "$ROOT/weights/qwen21/model_index.json" ]] || { printf 'Qwen model weights missing\n' >&2; exit 1; }
[[ -x "$ROOT/envs/qwen21/bin/accelerate" ]] || { printf 'Qwen environment missing\n' >&2; exit 1; }
RUN_NAME="qwen21_native_lora_smoke_${PROFILE}gb_$(date -u +%Y%m%dT%H%M%SZ)"
"$ROOT/envs/qwen21/bin/accelerate" launch --num_processes 1 \
  "$ROOT/sources/diffusers-qwen/examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py" \
  --pretrained_model_name_or_path "$ROOT/weights/qwen21" \
  --dataset_name "$ROOT/data/qwen_smoke_dataset" \
  --cond_image_column cond_image --image_column image --caption_column caption \
  --instance_prompt 'Photograph the person from the reference in the requested scene.' \
  --output_dir "$ROOT/runs/$RUN_NAME" \
  --mixed_precision bf16 --resolution 512 --train_batch_size 1 \
  --gradient_accumulation_steps 8 --gradient_checkpointing \
  --rank 16 --lora_alpha 16 --learning_rate 0.0001 \
  --lr_scheduler constant --lr_warmup_steps 0 --max_train_steps 100 \
  --cache_latents --offload --seed 42 --report_to comet_ml --skip_final_inference
