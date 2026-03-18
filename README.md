# Qwen3-0.6B LoRA Fine-tuning (PEFT + Transformers)

本项目使用 `PEFT + Transformers` 对 `Qwen/Qwen3-0.6B-Base` 做 LoRA 微调。

## Benchmark 选择

这里选择 **GSM8K (`openai/gsm8k`)**：
- 在大模型论文中非常常见（尤其是推理/数学能力评测与微调实验）。
- 数据集规范、可直接通过 Hugging Face `datasets` 加载。
- 对 0.6B 级别模型也有较好的实验可行性。

## 安装

```bash
pip install -r requirements.txt
```

## 训练

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

## 主要脚本参数

- `--train_samples`: 训练样本数（默认 2000）
- `--eval_samples`: 验证样本数（默认 200）
- `--max_length`: 最大序列长度（默认 1024）
- `--lora_r`: LoRA rank（默认 16）
- `--lora_alpha`: LoRA alpha（默认 32）
- `--lora_dropout`: LoRA dropout（默认 0.05）

## 输出

训练完成后会在 `--output_dir` 下保存：
- LoRA adapter 权重
- tokenizer 配置
