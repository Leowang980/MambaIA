#!/usr/bin/env bash
# GSM8K eval: base + LoRA (or other HF PEFT) checkpoint; default path matches train_lora.sh.
set -euo pipefail

export HF_HOME="${HF_HOME:-/root/autodl-tmp/MambaIA/data}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B-Base}"
ADAPTER_PATH="${ADAPTER_PATH:-${ROOT}/outputs/qwen3-0.6b-gsm8k-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/bench}"

python3 evaluate.py \
    --model_type peft \
    --base_model "${BASE_MODEL}" \
    --adapter_path "${ADAPTER_PATH}" \
    --dataset_name "${DATASET_NAME:-openai/gsm8k}" \
    --dataset_config "${DATASET_CONFIG:-main}" \
    --split "${SPLIT:-test}" \
    --num_samples "${NUM_SAMPLES:-100}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-2048}" \
    --device "${DEVICE:-cuda}" \
    --seed "${SEED:-42}" \
    --output_dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS:-}
