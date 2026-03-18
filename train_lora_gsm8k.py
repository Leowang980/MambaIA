import argparse
import inspect
from dataclasses import dataclass

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
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
    max_length: int = 1024
    train_samples: int = 2000
    eval_samples: int = 200
    learning_rate: float = 2e-4
    train_batch_size: int = 2
    eval_batch_size: int = 2
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


def format_example(question: str, answer: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )


def main():
    args = parse_args()

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
    train_ds = dataset["train"].select(range(min(args.train_samples, len(dataset["train"]))))
    eval_ds = dataset["test"].select(range(min(args.eval_samples, len(dataset["test"]))))

    def preprocess(batch):
        texts = [format_example(q, a) for q, a in zip(batch["question"], batch["answer"])]
        tokenized = tokenizer(
            texts,
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )
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

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    ta_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
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
