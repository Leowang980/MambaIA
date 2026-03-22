#!/usr/bin/env bash
# GSM8K eval: chat-tuned Qwen (non-base prompt format); for comparing instruct models.
set -euo pipefail

export HF_HOME="${HF_HOME:-/root/autodl-tmp/MambaIA/data}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/bench}"

python3 evaluate.py \
    --model_type qwen \
    --qwen_model "${QWEN_MODEL}" \
    --dataset_name "${DATASET_NAME:-openai/gsm8k}" \
    --dataset_config "${DATASET_CONFIG:-main}" \
    --split "${SPLIT:-test}" \
    --num_samples "${NUM_SAMPLES:-100}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-2048}" \
    --device "${DEVICE:-cuda}" \
    --seed "${SEED:-42}" \
    --output_dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS:-}
