import argparse
import inspect
import random
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


SYSTEM_PROMPT = "You are a helpful math tutor. Solve the problem step by step."


@dataclass
class ScriptArgs:
    model_name_or_path: str = "Qwen/Qwen3-0.6B-Base"
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    output_dir: str = "./outputs/qwen3-0.6b-gsm8k-lora"
    max_length: int = 4096
    train_samples: int = -1  # -1 表示使用全部训练集
    eval_samples: int = 500  # 验证集样本数，-1 表示全部
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 2
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    warmup_steps: int = 20
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


def parse_args() -> ScriptArgs:
    parser = argparse.ArgumentParser()
    defaults = ScriptArgs()
    for field_name, field_value in defaults.__dict__.items():
        parser.add_argument(
            f"--{field_name}",
            type=type(field_value),
            default=field_value,
        )
    ns = parser.parse_args()
    return ScriptArgs(**vars(ns))


def format_prompt_prefix(model_name_or_path: str, question: str, bos_token: str = "") -> str:
    """返回 answer 之前的 prompt 部分，用于计算 loss mask 边界。"""
    if "Base" in model_name_or_path:
        return f"{bos_token}{question}\n\n"
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
    if "Base" in model_name_or_path:
        return f"{bos_token}{question}\n\n{answer}{eos_token}"
    part = answer.split("####")
    think = part[0]
    answer = part[1]
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n{think}</think>\n\n{answer}<|im_end|>"
    )


class DataCollatorForCausalLMWithLossMask:
    """自定义 collator：保留 labels 中的 -100 作为 loss mask，仅对 padding 做 padding。"""

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

    # 固定随机种子，保证训练可复现
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(args.dataset_name, args.dataset_config)
    n_train = len(dataset["train"]) if args.train_samples < 0 else min(args.train_samples, len(dataset["train"]))
    n_eval = len(dataset["test"]) if args.eval_samples < 0 else min(args.eval_samples, len(dataset["test"]))
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
        # 计算每个样本中 answer 起始的字符位置，用于 loss mask
        answer_start_chars = [
            len(format_prompt_prefix(args.model_name_or_path, q, bos_token=bos))
            for q in batch["question"]
        ]
        tokenized = tokenizer(
            texts,
            max_length=args.max_length,
            truncation=True,
            truncation_side="left",  # 保留 answer+EOS，截断 question，确保模型学到 EOS
            padding=False,
            return_offsets_mapping=True,
        )
        # 构建 labels：prompt 部分为 -100（不计算 loss），answer 部分为 input_ids
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

    # Prefer bf16 when available; otherwise fallback to fp16 on CUDA.
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            ta_kwargs["bf16"] = True
        else:
            ta_kwargs["fp16"] = True

    # MPS backend currently does not benefit from pinned memory.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if "dataloader_pin_memory" in ta_signature:
            ta_kwargs["dataloader_pin_memory"] = False

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
