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

# Prefix tuning only updates prefix params: use a higher LR than LoRA (similar to prompt tuning).
# prefix_projection=true often optimizes more stably (extra MLP to key/value space).
NUM_VIRTUAL_TOKENS="${NUM_VIRTUAL_TOKENS:-32}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
PREFIX_PROJECTION="${PREFIX_PROJECTION:-true}"

python3 train_peft_gsm8k.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --adapter_type prefix_tuning \
    --num_virtual_tokens "${NUM_VIRTUAL_TOKENS}" \
    --prefix_projection "${PREFIX_PROJECTION}" \
    --train_samples "${TRAIN_SAMPLES:--1}" \
    --eval_samples "${EVAL_SAMPLES:-500}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
    --train_batch_size "${TRAIN_BATCH_SIZE:-6}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-6}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
    --learning_rate "${LEARNING_RATE}" \
    --logging_steps "${LOGGING_STEPS:-10}" \
    --eval_steps "${EVAL_STEPS:-100}" \
    --save_steps "${SAVE_STEPS:-100}" \
    --warmup_steps "${WARMUP_STEPS:-20}" \
    ${EXTRA_ARGS:-}
