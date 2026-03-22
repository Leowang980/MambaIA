# Qwen3-0.6B PEFT Fine-tuning (Transformers)

This project fine-tunes `Qwen/Qwen3-0.6B-Base` with PEFT using `Transformers`.
Supported adapter methods:
- LoRA
- Prompt Tuning
- Prefix Tuning
- IA3
- Bottleneck Adapter (classic PEFT adapter via `--adapter_type adapter`, implemented in this repo; not shipped in Hugging Face PEFT anymore)

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
python train_peft_gsm8k.py \
  --model_name_or_path Qwen/Qwen3-0.6B-Base \
  --output_dir ./outputs/qwen3-0.6b-gsm8k-peft \
  --adapter_type lora \
  --train_samples 2000 \
  --eval_samples 200 \
  --num_train_epochs 2 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 8
```

### Prompt Tuning

On small models, **RANDOM init + the same learning rate as LoRA (e.g. 2e-5)** often fails or yields garbage/empty generations. Prefer **TEXT init**, **more virtual tokens**, and **learning_rate around 1e-4 to 1e-3** (only soft prompts are trained). See `./scripts/train/train_prompt.sh`.

```bash
python train_peft_gsm8k.py \
  --adapter_type prompt_tuning \
  --num_virtual_tokens 48 \
  --prompt_tuning_init TEXT \
  --prompt_tuning_init_text "Solve the math problem step by step." \
  --learning_rate 3e-4
```

### Prefix Tuning

On Qwen3, Prefix Tuning combined with **gradient checkpointing + SDPA** can trigger attention shape errors. This repo **disables gradient checkpointing** and uses **eager attention** for `prefix_tuning` (higher VRAM; reduce batch if needed).

```bash
python train_peft_gsm8k.py \
  --adapter_type prefix_tuning \
  --num_virtual_tokens 32 \
  --prefix_projection true
```

### IA3

```bash
python train_peft_gsm8k.py \
  --adapter_type ia3 \
  --ia3_target_modules k_proj,v_proj,down_proj \
  --ia3_feedforward_modules down_proj
```

### Bottleneck Adapter (classic PEFT adapter)

Weights are saved under `output_dir` as `bottleneck_adapter_config.json` and `bottleneck_adapter.safetensors` (or `.bin`). `evaluate.py` / `test_model.py` with `--model_type peft` auto-detect this layout.

```bash
python train_peft_gsm8k.py \
  --adapter_type adapter \
  --adapter_bottleneck_dim 64 \
  --adapter_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --adapter_non_linearity relu
```

Or use `./scripts/train/train_ada.sh`.

> `train_lora_gsm8k.py` is kept as a backward-compatible entry and now forwards to `train_peft_gsm8k.py`.

## Main Script Arguments

- `--train_samples`: Number of training samples (default: 2000)
- `--eval_samples`: Number of evaluation samples (default: 200)
- `--max_length`: Maximum sequence length (default: 1024)
- `--adapter_type`: `lora | prompt_tuning | prefix_tuning | ia3 | adapter`
- `--lora_r`, `--lora_alpha`, `--lora_dropout`: LoRA params
- `--num_virtual_tokens`: Prompt/Prefix virtual tokens
- `--prompt_tuning_init`, `--prompt_tuning_init_text`: Prompt Tuning params
- `--prefix_projection`: Prefix Tuning projection switch
- `--ia3_target_modules`, `--ia3_feedforward_modules`: IA3 target modules
- `--adapter_target_modules`, `--adapter_bottleneck_dim`, `--adapter_dropout`, `--adapter_non_linearity`: Bottleneck Adapter params

## Outputs

After training, `--output_dir` will contain:
- PEFT adapter weights
- Tokenizer files
