from typing import Optional

from peft import PeftConfig, get_peft_model

from .bottleneck_adapter import apply_bottleneck_adapter_peft
from .ia3 import build_ia3_config
from .lora import build_lora_config
from .prefix_tuning import build_prefix_tuning_config
from .prompt_tuning import build_prompt_tuning_config
from .types import AdapterBuildConfig


SUPPORTED_ADAPTER_TYPES = ("lora", "prompt_tuning", "prefix_tuning", "ia3", "adapter")


def build_peft_config(
    cfg: AdapterBuildConfig,
    tokenizer_name_or_path: Optional[str] = None,
) -> PeftConfig:
    adapter_type = cfg.adapter_type.lower()
    if adapter_type == "adapter":
        raise ValueError(
            "adapter (bottleneck) is injected via apply_peft_model; build_peft_config does not apply. "
            "Use apply_peft_model(model, cfg, ...) instead."
        )
    if adapter_type == "lora":
        return build_lora_config(cfg)
    if adapter_type == "prompt_tuning":
        if tokenizer_name_or_path is None:
            raise ValueError("tokenizer_name_or_path is required for prompt_tuning.")
        return build_prompt_tuning_config(cfg, tokenizer_name_or_path=tokenizer_name_or_path)
    if adapter_type == "prefix_tuning":
        return build_prefix_tuning_config(cfg)
    if adapter_type == "ia3":
        return build_ia3_config(cfg)
    raise ValueError(
        f"Unsupported adapter_type={cfg.adapter_type}. "
        f"Supported: {', '.join(SUPPORTED_ADAPTER_TYPES)}"
    )


def apply_peft_model(
    model,
    cfg: AdapterBuildConfig,
    tokenizer_name_or_path: Optional[str] = None,
):
    if cfg.adapter_type.lower() == "adapter":
        return apply_bottleneck_adapter_peft(model, cfg)
    peft_config = build_peft_config(cfg, tokenizer_name_or_path=tokenizer_name_or_path)
    return get_peft_model(model, peft_config)
