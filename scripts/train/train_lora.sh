#!/usr/bin/env bash
# LoRA fine-tuning on GSM8K (see README.md for argument details).
set -euo pipefail


export HF_HOME=/root/autodl-tmp/MambaIA/data
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"



MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-0.6B-Base}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/qwen3-0.6b-gsm8k-lora}"

python3 train_peft_gsm8k.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --adapter_type lora \
    --lora_r "${LORA_R:-8}" \
    --lora_alpha "${LORA_ALPHA:-32}" \
    --lora_dropout "${LORA_DROPOUT:-0.05}" \
    --lora_target_modules "${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}" \
    --train_samples "${TRAIN_SAMPLES:--1}" \
    --eval_samples "${EVAL_SAMPLES:-500}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
    --train_batch_size "${TRAIN_BATCH_SIZE:-8}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-8}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
    --learning_rate "${LEARNING_RATE:-2e-5}" \
    --logging_steps "${LOGGING_STEPS:-10}" \
    --eval_steps "${EVAL_STEPS:-100}" \
    --save_steps "${SAVE_STEPS:-100}" \
    --warmup_steps "${WARMUP_STEPS:-20}" \
    ${EXTRA_ARGS:-}
