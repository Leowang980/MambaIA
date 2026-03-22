#!/usr/bin/env bash
# GSM8K 评测：仅 Base 模型（无 PEFT），与 train 脚本使用的基座默认一致。
set -euo pipefail

export HF_HOME="${HF_HOME:-/root/autodl-tmp/MambaIA/data}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B-Base}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/bench}"

python3 evaluate.py \
    --model_type base \
    --base_model "${BASE_MODEL}" \
    --dataset_name "${DATASET_NAME:-openai/gsm8k}" \
    --dataset_config "${DATASET_CONFIG:-main}" \
    --split "${SPLIT:-test}" \
    --num_samples "${NUM_SAMPLES:-100}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-2048}" \
    --device "${DEVICE:-cuda}" \
    --seed "${SEED:-42}" \
    --output_dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS:-}
