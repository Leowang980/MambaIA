#!/usr/bin/env bash
# Prefix tuning on GSM8K. Training disables gradient_checkpointing and uses eager attention on Qwen3 (avoids SDPA shape errors).
set -euo pipefail

export HF_HOME=/root/autodl-tmp/MambaIA/data
export HF_ENDPOINT=https://hf-mirror.com
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-0.6B-Base}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/qwen3-0.6b-gsm8k-prefix}"

# PREFIX_PROJECTION: true|false (train_peft_gsm8k.py uses parse_bool)
PREFIX_PROJECTION="${PREFIX_PROJECTION:-false}"

python3 train_peft_gsm8k.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --adapter_type prefix_tuning \
    --num_virtual_tokens "${NUM_VIRTUAL_TOKENS:-20}" \
    --prefix_projection "${PREFIX_PROJECTION}" \
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
