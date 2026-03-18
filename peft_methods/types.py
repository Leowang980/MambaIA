from dataclasses import dataclass, field
from typing import List, Literal


AdapterType = Literal["lora", "prompt_tuning", "prefix_tuning", "ia3"]


@dataclass
class AdapterBuildConfig:
    adapter_type: str = "lora"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Prompt Tuning / Prefix Tuning
    num_virtual_tokens: int = 20

    # Prompt Tuning only
    prompt_tuning_init: str = "RANDOM"  # RANDOM or TEXT
    prompt_tuning_init_text: str = "Solve the problem carefully."

    # Prefix Tuning only
    prefix_projection: bool = False

    # IA3
    ia3_target_modules: List[str] = field(
        default_factory=lambda: ["k_proj", "v_proj", "down_proj"]
    )
    ia3_feedforward_modules: List[str] = field(default_factory=lambda: ["down_proj"])
