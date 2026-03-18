from peft import IA3Config, TaskType

from .types import AdapterBuildConfig


def build_ia3_config(cfg: AdapterBuildConfig) -> IA3Config:
    return IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg.ia3_target_modules,
        feedforward_modules=cfg.ia3_feedforward_modules,
    )
