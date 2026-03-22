"""
经典 Bottleneck Adapter（Houlsby et al. 风格）：在目标 Linear 上并联 down→act→up。
当前 HuggingFace PEFT 已不再内置该类型，因此在项目内实现并与训练/评测脚本对接。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from .types import AdapterBuildConfig


class LinearWithBottleneckAdapter(nn.Module):
    """y = W x + Up(Act(Drop(Down(x))))，基座 Linear 冻结，仅训练 adapter 分支。"""

    def __init__(
        self,
        linear: nn.Linear,
        bottleneck_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "relu",
    ):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False

        dev = linear.weight.device
        dtype = linear.weight.dtype
        self.adapter_down = nn.Linear(linear.in_features, bottleneck_dim, bias=False).to(
            device=dev, dtype=dtype
        )
        self.adapter_up = nn.Linear(bottleneck_dim, linear.out_features, bias=False).to(
            device=dev, dtype=dtype
        )
        self.dropout = nn.Dropout(dropout)
        nl = non_linearity.lower()
        if nl == "gelu":
            self.act = nn.GELU()
        elif nl == "relu":
            self.act = nn.ReLU()
        elif nl == "silu" or nl == "swish":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported non_linearity={non_linearity!r}, use relu|gelu|silu")

        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        h = self.adapter_down(self.dropout(x))
        h = self.act(h)
        return base + self.adapter_up(h)


def _parent_child(model: nn.Module, full_name: str) -> tuple[nn.Module, str]:
    if "." not in full_name:
        return model, full_name
    parent_path, child = full_name.rsplit(".", 1)
    return model.get_submodule(parent_path), child


def inject_bottleneck_adapters(
    model: nn.Module,
    target_modules: List[str],
    bottleneck_dim: int,
    dropout: float,
    non_linearity: str,
) -> nn.Module:
    target_set = set(target_modules)
    to_replace: List[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        suffix = name.split(".")[-1]
        if suffix in target_set:
            to_replace.append((name, module))

    for full_name, linear in to_replace:
        parent, child = _parent_child(model, full_name)
        wrapped = LinearWithBottleneckAdapter(
            linear,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            non_linearity=non_linearity,
        )
        setattr(parent, child, wrapped)
    return model


def freeze_all_unfreeze_adapters(model: nn.Module) -> None:
    for _, p in model.named_parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "adapter_down" in name or "adapter_up" in name:
            p.requires_grad = True


def print_trainable_parameter_stats(model: nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    print(f"trainable params: {trainable:,d} || all params: {total:,d} || trainable%: {pct:.4f}")


def apply_bottleneck_adapter_peft(model: nn.Module, cfg: AdapterBuildConfig) -> nn.Module:
    inject_bottleneck_adapters(
        model,
        target_modules=list(cfg.adapter_target_modules),
        bottleneck_dim=cfg.adapter_bottleneck_dim,
        dropout=cfg.adapter_dropout,
        non_linearity=cfg.adapter_non_linearity,
    )
    freeze_all_unfreeze_adapters(model)
    return model


BOTTLENECK_CONFIG_NAME = "bottleneck_adapter_config.json"
BOTTLENECK_WEIGHTS_SAFE = "bottleneck_adapter.safetensors"
BOTTLENECK_WEIGHTS_BIN = "bottleneck_adapter.bin"


def bottleneck_adapter_checkpoint_dict(cfg: AdapterBuildConfig, base_model_name_or_path: str) -> Dict[str, Any]:
    return {
        "peft_method": "bottleneck_adapter",
        "base_model_name_or_path": base_model_name_or_path,
        "target_modules": list(cfg.adapter_target_modules),
        "bottleneck_dim": cfg.adapter_bottleneck_dim,
        "dropout": cfg.adapter_dropout,
        "non_linearity": cfg.adapter_non_linearity,
    }


def is_bottleneck_adapter_checkpoint(path: str) -> bool:
    return os.path.isfile(os.path.join(path, BOTTLENECK_CONFIG_NAME))


def _adapter_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if ("adapter_down" in k or "adapter_up" in k)}


def save_bottleneck_adapter(
    model: nn.Module,
    output_dir: str,
    cfg: AdapterBuildConfig,
    base_model_name_or_path: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    meta = bottleneck_adapter_checkpoint_dict(cfg, base_model_name_or_path)
    with open(os.path.join(output_dir, BOTTLENECK_CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    sd = _adapter_state_dict(model)
    safe_path = os.path.join(output_dir, BOTTLENECK_WEIGHTS_SAFE)
    bin_path = os.path.join(output_dir, BOTTLENECK_WEIGHTS_BIN)
    try:
        from safetensors.torch import save_file

        save_file(sd, safe_path)
    except Exception:
        torch.save(sd, bin_path)


def load_bottleneck_adapter(
    model: nn.Module,
    adapter_path: str,
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    cfg_path = os.path.join(adapter_path, BOTTLENECK_CONFIG_NAME)
    with open(cfg_path, encoding="utf-8") as f:
        meta = json.load(f)

    inject_bottleneck_adapters(
        model,
        target_modules=list(meta["target_modules"]),
        bottleneck_dim=int(meta["bottleneck_dim"]),
        dropout=float(meta.get("dropout", 0.0)),
        non_linearity=str(meta.get("non_linearity", "relu")),
    )

    safe_path = os.path.join(adapter_path, BOTTLENECK_WEIGHTS_SAFE)
    bin_path = os.path.join(adapter_path, BOTTLENECK_WEIGHTS_BIN)
    if os.path.isfile(safe_path):
        from safetensors.torch import load_file

        weights = load_file(safe_path, device="cpu")
        model.load_state_dict(weights, strict=False)
    elif os.path.isfile(bin_path):
        try:
            weights = torch.load(bin_path, map_location="cpu", weights_only=True)
        except TypeError:
            weights = torch.load(bin_path, map_location="cpu")
        model.load_state_dict(weights, strict=False)
    else:
        raise FileNotFoundError(
            f"No weights in {adapter_path}: expected {BOTTLENECK_WEIGHTS_SAFE} or {BOTTLENECK_WEIGHTS_BIN}"
        )

    if device is not None:
        model.to(device)

    freeze_all_unfreeze_adapters(model)
    return model
