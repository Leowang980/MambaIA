import argparse
import json
import os
import re
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = "You are a helpful math tutor. Solve the problem step by step."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate single model on GSM8K")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["qwen", "base", "lora"],
        default="lora",
        help="qwen=Qwen3-0.6B, base=Qwen3-0.6B-Base, lora=Base+LoRA",
    )
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--lora_path", type=str, default="./outputs/qwen3-0.6b-gsm8k-lora")
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_config", type=str, default="main")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/MambaIA/outputs/bench",
    )
    return parser.parse_args()


def build_prompt(question: str, model_type: str, tokenizer: AutoTokenizer) -> str:
    if model_type in ("base", "lora"):
        bos = tokenizer.bos_token or ""
        return f"{bos}{question}\n\n"
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def extract_label_answer(answer_text: str) -> Optional[str]:
    # GSM8K official answer format usually ends with "#### 42"
    m = re.search(r"####\s*([^\n]+)", answer_text)
    if not m:
        return None
    return normalize_number_string(m.group(1))


def normalize_number_string(text: str) -> Optional[str]:
    if text is None:
        return None
    text = text.strip()
    text = text.replace(",", "")
    # Pick the last signed/unsigned integer or float in the text.
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    if not matches:
        return None
    value = matches[-1]
    # Normalize 42.0 -> 42
    if "." in value:
        try:
            f = float(value)
            if f.is_integer():
                return str(int(f))
            return str(f)
        except ValueError:
            return value
    return value


def extract_pred_answer(generated_text: str) -> Optional[str]:
    return normalize_number_string(generated_text)


def write_json_atomic(output_json: str, payload: Dict[str, object]) -> None:
    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    tmp_path = f"{output_json}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_json)


@torch.inference_mode()
def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int,
    model_type: str = "qwen",
) -> str:
    prompt = build_prompt(question, model_type, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,  # 抑制 #### X 重复
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: List[Dict[str, str]],
    max_new_tokens: int,
    model_type: str,
    report_meta: Dict[str, object],
    output_json: str,
) -> Dict[str, object]:
    correct = 0
    details: List[Dict[str, object]] = []
    total = len(samples)

    for idx, sample in enumerate(samples):
        question = sample["question"]
        gold_raw = sample["answer"]
        gold = extract_label_answer(gold_raw)

        pred_text = generate_answer(
            model, tokenizer, question, max_new_tokens=max_new_tokens, model_type=model_type
        )
        pred = extract_pred_answer(pred_text)
        is_correct = (gold is not None) and (pred == gold)
        if is_correct:
            correct += 1

        details.append(
            {
                "index": idx,
                "question": question,
                "gold": gold,
                "gold_raw": gold_raw,
                "pred": pred,
                "correct": is_correct,
                "pred_text": pred_text,
            }
        )
        print(f"[{idx + 1}/{total}] gold={gold} pred={pred} correct={is_correct}")

        completed = idx + 1
        running_acc = correct / completed if completed > 0 else 0.0
        running_report = {
            **report_meta,
            "progress": {"completed": completed, "total": total},
            "accuracy": running_acc,
            "details": details,
        }
        write_json_atomic(output_json, running_report)

    accuracy = correct / total if total > 0 else 0.0
    final_report = {
        **report_meta,
        "progress": {"completed": total, "total": total},
        "accuracy": accuracy,
        "details": details,
    }
    write_json_atomic(output_json, final_report)
    return {"accuracy": accuracy, "details": details}


def ensure_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name_or_path: str, device: torch.device) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map=None,
    )
    return model.to(device)


def load_lora_model(
    base_model_name: str,
    lora_path: str,
    device: torch.device,
) -> AutoModelForCausalLM:
    base_for_lora = load_model(base_model_name, device=device)
    lora_model = PeftModel.from_pretrained(base_for_lora, lora_path)
    return lora_model.to(device)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, fallback to CPU.")
        runtime_device = torch.device("cpu")
    else:
        runtime_device = torch.device(args.device)
    print(f"Runtime device: {runtime_device}")

    print("Loading dataset...")
    ds = load_dataset(args.dataset_name, args.dataset_config)[args.split]
    num_samples = min(args.num_samples, len(ds))
    samples = [ds[i] for i in range(num_samples)]

    if args.model_type == "qwen":
        print(f"Loading model: {args.qwen_model}")
        tokenizer = ensure_tokenizer(args.qwen_model)
        model = load_model(args.qwen_model, device=runtime_device)
        model_path = args.qwen_model
    elif args.model_type == "base":
        print(f"Loading model: {args.base_model}")
        tokenizer = ensure_tokenizer(args.base_model)
        model = load_model(args.base_model, device=runtime_device)
        model_path = args.base_model
    else:
        print(f"Loading base: {args.base_model}, LoRA: {args.lora_path}")
        tokenizer = ensure_tokenizer(args.base_model)
        model = load_lora_model(args.base_model, args.lora_path, device=runtime_device)
        model_path = args.lora_path

    report_meta = {
        "model_type": args.model_type,
        "model_path": model_path,
        "dataset": f"{args.dataset_name}/{args.dataset_config}",
        "split": args.split,
        "num_samples": num_samples,
        "max_new_tokens": args.max_new_tokens,
    }

    print("Evaluating...")
    output_json = os.path.join(args.output_dir, f"eval_gsm8k_{args.model_type}.json")
    eval_result = evaluate_model(
        model,
        tokenizer,
        samples,
        max_new_tokens=args.max_new_tokens,
        model_type=args.model_type,
        report_meta=report_meta,
        output_json=output_json,
    )
    accuracy = eval_result["accuracy"]

    print("\n===== RESULT =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved report to: {output_json}")


if __name__ == "__main__":
    main()
