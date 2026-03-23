"""Helpers to read PEFT adapter directories (shared by train/eval scripts)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

ADAPTER_CONFIG_NAME = "adapter_config.json"


def read_adapter_config(adapter_path: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(adapter_path, ADAPTER_CONFIG_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_prefix_tuning_checkpoint(adapter_path: str) -> bool:
    cfg = read_adapter_config(adapter_path)
    return bool(cfg and cfg.get("peft_type") == "PREFIX_TUNING")


def base_causal_lm_kwargs_for_peft_adapter(adapter_path: str) -> Dict[str, Any]:
    """Extra from_pretrained kwargs for the base LM when wrapping with a PEFT adapter."""
    if is_prefix_tuning_checkpoint(adapter_path):
        # Qwen3 + prefix past_key_values + SDPA can yield wrong attention during generate();
        # match train_peft_gsm8k.py prefix_tuning path (eager attention).
        return {"attn_implementation": "eager"}
    return {}
