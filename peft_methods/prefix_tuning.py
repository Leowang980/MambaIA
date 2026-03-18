from peft import PrefixTuningConfig, TaskType

from .types import AdapterBuildConfig


def build_prefix_tuning_config(cfg: AdapterBuildConfig) -> PrefixTuningConfig:
    return PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=cfg.num_virtual_tokens,
        prefix_projection=cfg.prefix_projection,
    )
