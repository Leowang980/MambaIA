from peft import PromptTuningConfig, PromptTuningInit, TaskType

from .types import AdapterBuildConfig


def build_prompt_tuning_config(
    cfg: AdapterBuildConfig,
    tokenizer_name_or_path: str,
) -> PromptTuningConfig:
    init_mode = cfg.prompt_tuning_init.upper()
    if init_mode not in {"RANDOM", "TEXT"}:
        raise ValueError("prompt_tuning_init must be RANDOM or TEXT")

    kwargs = {
        "task_type": TaskType.CAUSAL_LM,
        "num_virtual_tokens": cfg.num_virtual_tokens,
        "prompt_tuning_init": PromptTuningInit[init_mode],
    }
    if init_mode == "TEXT":
        kwargs["prompt_tuning_init_text"] = cfg.prompt_tuning_init_text
        kwargs["tokenizer_name_or_path"] = tokenizer_name_or_path
    return PromptTuningConfig(**kwargs)
