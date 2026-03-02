import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

# --- 設定 (Silence DPO v3 - 徹底矯正) ---
# Previous timestamps:
# SFT: 20260209_0709
# DPO v2: 20260210_Silence_v2 (60 steps, failed to silence)

TIMESTAMP = "20260209_0709" 
TIMESTAMP_DPO = "20260210_Silence_v3_Aggressive"

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = f"adapters/adapter_legacy_sft_{TIMESTAMP}"
# Use the v2 dataset which definitely has EOS tokens
DATA_PATH = "data/dpo_synthetic_dataset_v2.jsonl" 
OUTPUT_DIR = f"outputs/dpo_legacy_silence_{TIMESTAMP_DPO}"
ADAPTER_OUT_DIR = f"adapters/adapter_legacy_silence_{TIMESTAMP_DPO}"

# --- Tuned Hyperparameters for Aggressive Correction ---
# v2 was 60 steps, 5e-7 LR -> Too weak.
# v3 strategy: Higher LR, More Steps.
MAX_STEPS = 60    # 60 steps (v2と同じ。確実に終わるステップ数)
LEARNING_RATE = 2e-6 # 4x LR of v2 (Aggressive update - 短期決戦)
BATCH_SIZE = 1 # 復活
# Batch Accum = 8 # コメントとして残すか削除
GRAD_ACCUM = 8     # Increase accumulated batch size for more stable updates (Effective BS=8)
BETA = 0.3         # Strength of KL penalty. Higher beta = stick closer to ref? 
                   # No, standard DPO: Loss = -log sigmoid( beta * (log(pi_theta/pi_ref)) )
                   # We want to deviate from the "chatty" reference.
                   # Actually, lower beta increases the strength of the preference optimization relative to the KL constraint?
                   # Let's keep beta default 0.1 or slightly higher 0.2 to avoid mode collapse, 
                   # but rely on LR/Steps to drive the change. 
                   # Let's stick to 0.1 to allow maximum movement away from "chatty".

# Wait, if V2 (60 steps) took 2 hours, 120 steps will take 4 hours.
# User accepted "300 steps" thinking it's 1 hour, based on my wrong estimation.
# I will set 120 steps but high LR (2e-6) to make it count.
# 120 steps * 8 grad_accum = 960 samples seen. (Dataset size ~4000) -> 1/4 epoch.
# This should be enough to learn the format if the signal is strong.

def main():
    print(f"🔥 徹底矯正DPO学習(v3)を開始します... ID: {TIMESTAMP_DPO}")
    print(f"Settings: Steps={MAX_STEPS}, LR={LEARNING_RATE}, Eff.Batch={BATCH_SIZE*GRAD_ACCUM}")
    
    # データセットのロード
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    
    # モデルとトークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # ベースモデルロード
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": "mps"}
    )
    
    # SFT済みアダプタをロードして学習対象にする
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
    
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        save_strategy="steps", # こまめに保存
        save_steps=20,         # 20ステップごとにチェックポイント
        remove_unused_columns=False,
        bf16=False,
        report_to="none",
        beta=0.1 # Allow flexible deviation
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # Reference is the current SFT model (chatty)
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer, 
    )
    
    print("🚀 DPO v3 学習スタート！(Aggressive Mode)")
    trainer.train()
    
    print(f"💾 矯正済みアダプタを保存中... -> {ADAPTER_OUT_DIR}")
    trainer.save_model(ADAPTER_OUT_DIR)
    print("✅ 学習完了！")

if __name__ == "__main__":
    main()
