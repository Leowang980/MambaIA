#!/usr/bin/env bash
# IA3 on GSM8K. Default targets: q_proj,v_proj,down_proj (PEFT qwen3 mapping).
# Few trainable scalars: defaults use mild LR bump, zero weight decay, 3 epochs vs LoRA-style 2e-5/0.01/2ep.
set -euo pipefail

export HF_HOME=/root/autodl-tmp/MambaIA/data
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-0.6B-Base}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/qwen3-0.6b-gsm8k-ia3}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
# Optional: MAX_LENGTH=2048 speeds up training if truncating long CoT is acceptable

python3 train_peft_gsm8k.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --adapter_type ia3 \
    --ia3_target_modules "${IA3_TARGET_MODULES:-q_proj,v_proj,down_proj}" \
    --ia3_feedforward_modules "${IA3_FEEDFORWARD_MODULES:-down_proj}" \
    --train_samples "${TRAIN_SAMPLES:--1}" \
    --eval_samples "${EVAL_SAMPLES:-500}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --train_batch_size "${TRAIN_BATCH_SIZE:-8}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-8}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --logging_steps "${LOGGING_STEPS:-10}" \
    --eval_steps "${EVAL_STEPS:-100}" \
    --save_steps "${SAVE_STEPS:-100}" \
    --warmup_steps "${WARMUP_STEPS}" \
    ${EXTRA_ARGS:-}
