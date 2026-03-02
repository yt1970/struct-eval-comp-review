#!/bin/bash

# V10 MLX Training Script (Updated for stability)
# Using HuggingFace ID style to ensure correct config loading

MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
DATA_PATH="data/mlx_v10"
ADAPTER_PATH="adapters/v10_mlx"

echo "🚀 Starting MLX Training for V10 (Model ID: $MODEL_ID)..."
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
  --adapter-path "$ADAPTER_PATH"

echo "✅ Training Completed! Adapters are in $ADAPTER_PATH"
