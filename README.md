# Qwen3-0.6B PEFT Fine-tuning (Transformers)

This project fine-tunes `Qwen/Qwen3-0.6B-Base` with PEFT using `Transformers`.
Supported adapter methods:
- LoRA
- Prompt Tuning
- Prefix Tuning
- IA3
- Bottleneck Adapter（经典 Adapter PEFT，`--adapter_type adapter`，由本项目实现；HuggingFace PEFT 已不再内置）

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

```bash
python train_peft_gsm8k.py \
  --adapter_type prompt_tuning \
  --num_virtual_tokens 32 \
  --prompt_tuning_init RANDOM
```

### Prefix Tuning

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

### Bottleneck Adapter（经典 PEFT Adapter）

权重保存在 `output_dir` 下的 `bottleneck_adapter_config.json` 与 `bottleneck_adapter.safetensors`（或 `.bin`）。`evaluate.py` / `test_model.py` 在 `--model_type peft` 时会自动识别该目录。

```bash
python train_peft_gsm8k.py \
  --adapter_type adapter \
  --adapter_bottleneck_dim 64 \
  --adapter_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --adapter_non_linearity relu
```

或使用 `./scripts/train_ada.sh`。

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
- `--adapter_target_modules`, `--adapter_bottleneck_dim`, `--adapter_dropout`, `--adapter_non_linearity`: Bottleneck Adapter 参数

## Outputs

After training, `--output_dir` will contain:
- PEFT adapter weights
- Tokenizer files
