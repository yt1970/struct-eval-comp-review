import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import DPOConfig, DPOTrainer

# === DPO v7: 外科的スタイル矯正 ===
#
# v6 との違い:
#   1. layers_to_transform: 後半9層(27-35)のみ学習 → 知識を保護
#   2. target_modules: q_proj, v_proj のみ → 変更範囲を最小化
#   3. r=8 (半減) → 変更容量を制限
#   4. beta=0.5 (高) → 参照モデルからの乖離を抑制
#   5. データ: Rejectedに指示文風テキストなし → 安全

TIMESTAMP = "20260213_Surgical_v7"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "adapters/adapter_legacy_sft_20260209_0709"
DPO_DATA_PATH = "data/dpo_v7_surgical.jsonl"
OUTPUT_DIR = f"outputs/dpo_v7_{TIMESTAMP}"
FINAL_ADAPTER_DIR = "adapters/adapter_dpo_v7_surgical"

# ハイパーパラメータ（外科的設定）
NUM_STEPS = 50        # 300件 / (1 * 4) = 75 → 50ステップ（2/3周）で早期終了
LEARNING_RATE = 5e-7  # 精密調整
BATCH_SIZE = 1
GRAD_ACCUM = 4
BETA = 0.5            # 保守的: 参照モデルから離れすぎない

def main():
    print(f"🏥 DPO v7 外科的スタイル矯正 開始: {TIMESTAMP}")
    print(f"   戦略: 後半9層のAttention(Q,V)のみ学習, Beta={BETA}")
    
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. ベースモデル + SFT アダプタマージ
    print(f"   Loading Base Model: {BASE_MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"   Merging SFT Adapter: {SFT_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()
    print("   ✅ Base Model ready (Legacy SFT merged).")
    
    # 3. データセット
    dataset = load_dataset("json", data_files=DPO_DATA_PATH, split="train")
    
    def format_dpo(sample):
        return {
            "prompt": f"### 指示\n{sample['prompt']}\n\n### 応答\n",
            "chosen": sample["chosen"] + tokenizer.eos_token,
            "rejected": sample["rejected"] + tokenizer.eos_token,
        }
    dataset = dataset.map(format_dpo)
    print(f"   Formatted {len(dataset)} pairs.")
    
    # 4. LoRA Config — 外科的設定
    peft_config = LoraConfig(
        r=8,                      # v6: 16 → 変更容量を半減
        lora_alpha=16,            # v6: 32 → rank比維持
        target_modules=["q_proj", "v_proj"],  # v6: q,k,v,o → 最小限
        layers_to_transform=list(range(27, 36)),  # ★核心: 後半9層のみ
        task_type="CAUSAL_LM",
        lora_dropout=0.1,         # v6: 0.05 → 正則化強化
        bias="none",
    )
    
    print(f"   LoRA設定: r={peft_config.r}, modules={peft_config.target_modules}")
    print(f"   ★ layers_to_transform={peft_config.layers_to_transform}")
    
    # 5. DPO Config
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        max_steps=NUM_STEPS,
        save_strategy="no",
        remove_unused_columns=False,
        bf16=False,
        report_to="none",
        beta=BETA,                # v6: 0.1 → 保守的
        logging_steps=10,
    )
    
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    print(f"🚀 DPO v7 学習開始 (Steps={NUM_STEPS}, LR={LEARNING_RATE}, Beta={BETA})")
    trainer.train()
    
    print(f"💾 Saving DPO v7 Adapter → {FINAL_ADAPTER_DIR}")
    trainer.save_model(FINAL_ADAPTER_DIR)
    print("🎉 DPO v7 外科的スタイル矯正 完了!")

if __name__ == "__main__":
    main()
