import argparse
from threading import Thread
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from peft_methods.bottleneck_adapter import is_bottleneck_adapter_checkpoint, load_bottleneck_adapter


SYSTEM_PROMPT = "You are a helpful assistant. Please answer clearly and briefly."
DEFAULT_QUESTIONS = [
    "1+1 等于多少？请只给结果。",
    "如果小明有 10 个苹果，吃掉 3 个，还剩多少个？",
    "请用一句话解释什么是机器学习。",
    "把这句话翻译成英文：今天天气很好。",
    "如果 x + 3 = 10，那么 x 等于多少？",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple streaming test script for Qwen/PEFT models")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["qwen", "base", "peft", "custom"],
        default="peft",
        help="Which model preset to use.",
    )
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--adapter_path", type=str, default="./outputs/qwen3-0.6b-gsm8k-peft")
    parser.add_argument("--lora_path", type=str, default="", help="Deprecated alias of --adapter_path")
    parser.add_argument("--custom_model", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--questions",
        type=str,
        nargs="*",
        default=None,
        help="Custom question list. If empty, use built-in simple questions.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode after predefined questions.",
    )
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> str:
    if args.model_type == "qwen":
        return args.qwen_model
    if args.model_type == "base":
        return args.base_model
    if args.model_type == "custom":
        if not args.custom_model.strip():
            raise ValueError("When model_type=custom, --custom_model cannot be empty.")
        return args.custom_model.strip()
    return args.base_model


def build_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_model_and_tokenizer(args: argparse.Namespace):
    if args.lora_path:
        args.adapter_path = args.lora_path
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU.")
        runtime_device = torch.device("cpu")
    else:
        runtime_device = torch.device(args.device)
    print(f"Using device: {runtime_device}")

    model_path = resolve_model_path(args)
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=None,
    ).to(runtime_device)

    if args.model_type == "peft":
        print(f"Loading PEFT adapter from: {args.adapter_path}")
        if is_bottleneck_adapter_checkpoint(args.adapter_path):
            model = load_bottleneck_adapter(model, args.adapter_path, device=runtime_device)
        else:
            model = PeftModel.from_pretrained(model, args.adapter_path).to(runtime_device)

    model.eval()
    return model, tokenizer


def stream_generate_answer(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = question
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    chunks: List[str] = []
    for new_text in streamer:
        print(new_text, end="", flush=True)
        chunks.append(new_text)

    thread.join()
    print()
    return "".join(chunks).strip()


def run_fixed_questions(args: argparse.Namespace, model, tokenizer) -> None:
    questions = args.questions if args.questions else DEFAULT_QUESTIONS
    print(f"\nStart fixed-question test, total {len(questions)} questions.")
    for idx, q in enumerate(questions, start=1):
        print("\n" + "=" * 80)
        print(f"[Q{idx}] {q}")
        print("[A] ", end="", flush=True)
        _ = stream_generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=q,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


def run_interactive(args: argparse.Namespace, model, tokenizer) -> None:
    print("\n进入交互模式，输入 exit 或 quit 结束。")
    while True:
        user_q = input("\n你: ").strip()
        if user_q.lower() in {"exit", "quit"}:
            print("已退出交互模式。")
            break
        if not user_q:
            continue
        print("模型: ", end="", flush=True)
        _ = stream_generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=user_q,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args)
    run_fixed_questions(args, model, tokenizer)
    if args.interactive:
        run_interactive(args, model, tokenizer)


if __name__ == "__main__":
    main()
