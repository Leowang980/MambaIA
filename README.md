# Qwen3-0.6B LoRA Fine-tuning (PEFT + Transformers)

This project fine-tunes `Qwen/Qwen3-0.6B-Base` with LoRA using `PEFT + Transformers`.

## Benchmark Choice

This setup uses **GSM8K (`openai/gsm8k`)**:
- It is widely used in LLM papers, especially for reasoning and math evaluation/fine-tuning.
- It has a clean format and is easy to load via Hugging Face `datasets`.
- It is practical for experiments with a 0.6B-scale model.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train_lora_gsm8k.py \
  --model_name_or_path Qwen/Qwen3-0.6B-Base \
  --output_dir ./outputs/qwen3-0.6b-gsm8k-lora \
  --train_samples 2000 \
  --eval_samples 200 \
  --num_train_epochs 2 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 8
```

## Main Script Arguments

- `--train_samples`: Number of training samples (default: 2000)
- `--eval_samples`: Number of evaluation samples (default: 200)
- `--max_length`: Maximum sequence length (default: 1024)
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--lora_dropout`: LoRA dropout (default: 0.05)

## Outputs

After training, `--output_dir` will contain:
- LoRA adapter weights
- Tokenizer files
