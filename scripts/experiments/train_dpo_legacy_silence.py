import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from trl import DPOTrainer, DPOConfig

# --- 設定 (Silence DPO) ---
TIMESTAMP = "20260209_0709" # Legacy SFTの時刻
TIMESTAMP_DPO = "20260209_Silence"

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = f"adapters/adapter_legacy_sft_{TIMESTAMP}"
DATA_PATH = "data/dpo_synthetic_dataset.jsonl"
OUTPUT_DIR = f"outputs/dpo_legacy_silence_{TIMESTAMP_DPO}"
ADAPTER_OUT_DIR = f"adapters/adapter_legacy_silence_{TIMESTAMP_DPO}"

MAX_STEPS = 100 
LEARNING_RATE = 5e-7 # 慎重に
BATCH_SIZE = 1
GRAD_ACCUM = 4

def main():
    print(f"🤫 沈黙のDPO学習を開始します... ID: {TIMESTAMP_DPO}")
    
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
    
    # SFT済みアダプタをロードしてマージ（または継続学習）
    # 今回はLoRAアダプタをロードした状態で、さらにその上に学習を重ねる（またはLoRA自体を更新）
    # PeftModelでロードすると、TrainerがLoRAとして認識して更新してくれるはず
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
    
    # DPO用設定
    # Refモデルは通常Baseモデルだが、今回はSFTとの乖離を抑えるため同一モデルを使用（暗黙的に）
    # または明示的にref_model=Noneとすると、現在のモデルのコピーがrefになる
    
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
        ref_model=None, # 自動的にmodelのコピーが作成される
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    print("🚀 DPO学習スタート！")
    trainer.train()
    
    print(f"💾 沈黙のアダプタを保存中... -> {ADAPTER_OUT_DIR}")
    trainer.save_model(ADAPTER_OUT_DIR)
    print("✅ 学習完了！")

if __name__ == "__main__":
    main()
