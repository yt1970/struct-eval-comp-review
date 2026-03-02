#!/bin/bash

# V10 MLX Training Script (Legendary Version - Inspired by AgentBench-comp)
# Using Gradient Checkpointing to enable 2048 context on 32GB Mac

MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
DATA_PATH="data/mlx_v10"
# 実験ごとにフォルダを分ける思想を採用
EXP_ID=$(date +%Y%m%d_%H%M%S)
ADAPTER_PATH="adapters/v10_mlx_exp_${EXP_ID}"

echo "🚀 Starting MLX Training for V10 (EXP: $EXP_ID)..."
echo "🔥 Applied: --grad-checkpoint --max-seq-length 2048 --lora-layers 16"

.venv/bin/python -m mlx_lm.lora \
  --model "$MODEL_ID" \
  --train \
  --data "$DATA_PATH" \
  --batch-size 4 \
  --iters 1000 \
  --save-every 100 \
  --steps-per-report 10 \
  --steps-per-eval 100 \
  --learning-rate 1e-5 \
  --adapter-path "$ADAPTER_PATH" \
  --max-seq-length 2048 \
  --grad-checkpoint \
  --lora-layers 16

echo "✅ Training Completed! Adapters are in $ADAPTER_PATH"
