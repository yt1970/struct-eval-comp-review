import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from trl import DPOTrainer, DPOConfig

# --- 設定 (Silence DPO v2) ---
TIMESTAMP = "20260209_0709" # Legacy SFT
TIMESTAMP_DPO = "20260210_Silence_v2" # 新しいタイムスタンプ

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = f"adapters/adapter_legacy_sft_{TIMESTAMP}"
DATA_PATH = "data/dpo_synthetic_dataset_v2.jsonl" # v2データセット
OUTPUT_DIR = f"outputs/dpo_legacy_silence_{TIMESTAMP_DPO}"
ADAPTER_OUT_DIR = f"adapters/adapter_legacy_silence_{TIMESTAMP_DPO}"

MAX_STEPS = 60 # 前回100stepsで2.5hだったので、60stepsに短縮（約1.5h目標）
LEARNING_RATE = 5e-7 
BATCH_SIZE = 1
GRAD_ACCUM = 4

def main():
    print(f"🤫 沈黙のDPO学習 v2 を開始します... ID: {TIMESTAMP_DPO}")
    
    # データセットのロード
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    
    # モデルとトークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    # Qwenの場合pad=eosになっていることが多いが念のため確認
    # また、EOSトークンをデータセットに含めているので、Special Tokenとして認識されるはず
    
    # ベースモデルロード
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": "mps"}
    )
    
    # SFT済みアダプタをロードして継続学習
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
    
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        save_strategy="no",
        remove_unused_columns=False,
        bf16=False,
        report_to="none",
        beta=0.1
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None, 
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer, # 修正済み
    )
    
    print("🚀 DPO学習 (v2) スタート！")
    trainer.train()
    
    print(f"💾 沈黙のアダプタ(v2)を保存中... -> {ADAPTER_OUT_DIR}")
    trainer.save_model(ADAPTER_OUT_DIR)
    print("✅ 学習完了！")

if __name__ == "__main__":
    main()
