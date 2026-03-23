import argparse
import inspect
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from peft_methods import AdapterBuildConfig, SUPPORTED_ADAPTER_TYPES, apply_peft_model
from peft_methods.bottleneck_adapter import print_trainable_parameter_stats, save_bottleneck_adapter


SYSTEM_PROMPT = "You are a helpful math tutor. Solve the problem step by step."


@dataclass
class ScriptArgs:
    model_name_or_path: str = "Qwen/Qwen3-0.6B-Base"
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    output_dir: str = "./outputs/qwen3-0.6b-template-gsm8k-peft"
    max_length: int = 4096
    train_samples: int = -1
    eval_samples: int = 500
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    warmup_steps: int = 20

    adapter_type: str = "lora"

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    # Prompt / Prefix (prompt tuning on small models: prefer TEXT init and higher LR; see scripts/train/train_prompt.sh)
    num_virtual_tokens: int = 32
    prompt_tuning_init: str = "TEXT"  # RANDOM or TEXT
    prompt_tuning_init_text: str = "Solve the math problem step by step."
    prefix_projection: bool = False

    # IA3
    ia3_target_modules: str = "q_proj,v_proj,down_proj"
    ia3_feedforward_modules: str = "down_proj"

    # Bottleneck Adapter (classic PEFT adapter, not LoRA)
    adapter_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    adapter_bottleneck_dim: int = 64
    adapter_dropout: float = 0.05
    adapter_non_linearity: str = "relu"


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool value: {value}")


def parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> ScriptArgs:
    parser = argparse.ArgumentParser(description="Train PEFT adapters on GSM8K")
    d = ScriptArgs()

    parser.add_argument("--model_name_or_path", type=str, default=d.model_name_or_path)
    parser.add_argument("--dataset_name", type=str, default=d.dataset_name)
    parser.add_argument("--dataset_config", type=str, default=d.dataset_config)
    parser.add_argument("--output_dir", type=str, default=d.output_dir)
    parser.add_argument("--max_length", type=int, default=d.max_length)
    parser.add_argument("--train_samples", type=int, default=d.train_samples)
    parser.add_argument("--eval_samples", type=int, default=d.eval_samples)
    parser.add_argument("--learning_rate", type=float, default=d.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=d.weight_decay)
    parser.add_argument("--max_grad_norm", type=float, default=d.max_grad_norm)
    parser.add_argument("--train_batch_size", type=int, default=d.train_batch_size)
    parser.add_argument("--eval_batch_size", type=int, default=d.eval_batch_size)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=d.gradient_accumulation_steps,
    )
    parser.add_argument("--num_train_epochs", type=int, default=d.num_train_epochs)
    parser.add_argument("--logging_steps", type=int, default=d.logging_steps)
    parser.add_argument("--eval_steps", type=int, default=d.eval_steps)
    parser.add_argument("--save_steps", type=int, default=d.save_steps)
    parser.add_argument("--warmup_steps", type=int, default=d.warmup_steps)

    parser.add_argument(
        "--adapter_type",
        type=str,
        choices=SUPPORTED_ADAPTER_TYPES,
        default=d.adapter_type,
    )

    parser.add_argument("--lora_r", type=int, default=d.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=d.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=d.lora_dropout)
    parser.add_argument("--lora_target_modules", type=str, default=d.lora_target_modules)

    parser.add_argument("--num_virtual_tokens", type=int, default=d.num_virtual_tokens)
    parser.add_argument("--prompt_tuning_init", type=str, default=d.prompt_tuning_init)
    parser.add_argument(
        "--prompt_tuning_init_text",
        type=str,
        default=d.prompt_tuning_init_text,
    )
    parser.add_argument(
        "--prefix_projection",
        type=parse_bool,
        default=d.prefix_projection,
    )

    parser.add_argument("--ia3_target_modules", type=str, default=d.ia3_target_modules)
    parser.add_argument(
        "--ia3_feedforward_modules",
        type=str,
        default=d.ia3_feedforward_modules,
    )

    parser.add_argument(
        "--adapter_target_modules",
        type=str,
        default=d.adapter_target_modules,
    )
    parser.add_argument(
        "--adapter_bottleneck_dim",
        type=int,
        default=d.adapter_bottleneck_dim,
    )
    parser.add_argument("--adapter_dropout", type=float, default=d.adapter_dropout)
    parser.add_argument(
        "--adapter_non_linearity",
        type=str,
        choices=["relu", "gelu", "silu"],
        default=d.adapter_non_linearity,
    )
    return ScriptArgs(**vars(parser.parse_args()))


def to_adapter_build_config(args: ScriptArgs) -> AdapterBuildConfig:
    return AdapterBuildConfig(
        adapter_type=args.adapter_type,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=parse_csv_list(args.lora_target_modules),
        num_virtual_tokens=args.num_virtual_tokens,
        prompt_tuning_init=args.prompt_tuning_init,
        prompt_tuning_init_text=args.prompt_tuning_init_text,
        prefix_projection=args.prefix_projection,
        ia3_target_modules=parse_csv_list(args.ia3_target_modules),
        ia3_feedforward_modules=parse_csv_list(args.ia3_feedforward_modules),
        adapter_target_modules=parse_csv_list(args.adapter_target_modules),
        adapter_bottleneck_dim=args.adapter_bottleneck_dim,
        adapter_dropout=args.adapter_dropout,
        adapter_non_linearity=args.adapter_non_linearity,
    )


def format_prompt_prefix(model_name_or_path: str, question: str, bos_token: str = "") -> str:
    if "Base" in model_name_or_path:
        return (
            f"{bos_token}Question: {question}\n"
            f"Answer: Let's think step by step.\n"
        )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def format_example(
    model_name_or_path: str,
    question: str,
    answer: str,
    bos_token: str = "",
    eos_token: str = "",
) -> str:
    part = answer.split("####")
    cot = part[0].strip()
    final_answer = part[1].strip()

    if "Base" in model_name_or_path:
        return (
            f"{bos_token}Question: {question}\n"
            f"Answer: Let's think step by step.\n"
            f"{cot}\n"
            f"Therefore, the answer is {final_answer}.{eos_token}"
        )

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n{cot}</think>\n\n{final_answer}<|im_end|>"
    )


class DataCollatorForCausalLMWithLossMask:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        batch_max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            batch_max_len = (
                (batch_max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        pad_id = self.tokenizer.pad_token_id

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = batch_max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)

        batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
        batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.long)
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
        return batch


def main():
    args = parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from_pretrained_kw: dict = {
        "torch_dtype": "auto",
        "device_map": "auto",
    }
    # Prefix tuning injects prefixes via past_key_values; Qwen3 + SDPA + gradient checkpointing can
    # mismatch attention dims (e.g. 448 vs 448+num_virtual_tokens). Eager attention + no GC avoids this.
    if args.adapter_type == "prefix_tuning":
        from_pretrained_kw["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **from_pretrained_kw,
    )

    adapter_cfg = to_adapter_build_config(args)
    model = apply_peft_model(
        model,
        adapter_cfg,
        tokenizer_name_or_path=args.model_name_or_path,
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        print_trainable_parameter_stats(model)

    dataset = load_dataset(args.dataset_name, args.dataset_config)
    n_train = (
        len(dataset["train"])
        if args.train_samples < 0
        else min(args.train_samples, len(dataset["train"]))
    )
    n_eval = (
        len(dataset["test"])
        if args.eval_samples < 0
        else min(args.eval_samples, len(dataset["test"]))
    )
    train_ds = dataset["train"].select(range(n_train))
    eval_ds = dataset["test"].select(range(n_eval))
    print(f"Train samples: {n_train}, Eval samples: {n_eval}")

    def preprocess(batch):
        bos = tokenizer.bos_token or ""
        eos = tokenizer.eos_token or ""
        texts = [
            format_example(args.model_name_or_path, q, a, bos_token=bos, eos_token=eos)
            for q, a in zip(batch["question"], batch["answer"])
        ]
        answer_start_chars = [
            len(format_prompt_prefix(args.model_name_or_path, q, bos_token=bos))
            for q in batch["question"]
        ]
        tokenized = tokenizer(
            texts,
            max_length=args.max_length,
            truncation=True,
            truncation_side="right",
            padding=False,
            return_offsets_mapping=True,
        )

        labels_list = []
        for i, offsets in enumerate(tokenized["offset_mapping"]):
            ans_start = answer_start_chars[i]
            labels = []
            for token_ids, (start, end) in zip(tokenized["input_ids"][i], offsets):
                if end <= ans_start:
                    labels.append(-100)
                else:
                    labels.append(token_ids)
            labels_list.append(labels)
        tokenized["labels"] = labels_list
        del tokenized["offset_mapping"]
        return tokenized

    train_ds = train_ds.map(
        preprocess,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train dataset",
    )
    eval_ds = eval_ds.map(
        preprocess,
        batched=True,
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval dataset",
    )

    collator = DataCollatorForCausalLMWithLossMask(tokenizer=tokenizer)

    ta_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "report_to": "none",
        "gradient_checkpointing": True,
        "warmup_steps": args.warmup_steps,
        "lr_scheduler_type": "cosine",
    }

    ta_signature = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in ta_signature:
        ta_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in ta_signature:
        ta_kwargs["eval_strategy"] = "steps"

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            ta_kwargs["bf16"] = True
        else:
            ta_kwargs["fp16"] = True

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if "dataloader_pin_memory" in ta_signature:
            ta_kwargs["dataloader_pin_memory"] = False

    # Bottleneck adapter wraps the full model; Trainer.save_model would save full weights; save adapters only at end.
    if args.adapter_type == "adapter":
        ta_kwargs["save_strategy"] = "no"

    if args.adapter_type == "prefix_tuning":
        ta_kwargs["gradient_checkpointing"] = False
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        print(
            "prefix_tuning: gradient_checkpointing disabled (incompatible with Qwen3 prefix KV + checkpointing); "
            "VRAM use increases — reduce batch size if needed.",
            flush=True,
        )

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()
    if args.adapter_type == "adapter":
        save_bottleneck_adapter(
            model,
            args.output_dir,
            adapter_cfg,
            args.model_name_or_path,
        )
    else:
        trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
