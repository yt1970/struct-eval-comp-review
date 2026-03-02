import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import DPOConfig, DPOTrainer

# === DPO v7.1: 攻めた外科的スタイル矯正 ===
#
# v7 からの変更:
#   Beta: 0.5 -> 0.2 (ブレーキを緩める)
#   LR:   5e-7 -> 2e-6 (変化速度を上げる)
#   layers_to_transform は維持 (知識保護は継続)

TIMESTAMP = "20260213_Surgical_v7_1"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "adapters/adapter_legacy_sft_20260209_0709"
DPO_DATA_PATH = "data/dpo_v7_surgical.jsonl"
OUTPUT_DIR = f"outputs/dpo_v7_1_{TIMESTAMP}"
FINAL_ADAPTER_DIR = "adapters/adapter_dpo_v7_1_surgical"

# v7.1 ハイパーパラメータ（攻めた設定）
NUM_STEPS = 50
LEARNING_RATE = 2e-6   # v7: 5e-7 -> 4倍に引き上げ
BATCH_SIZE = 1
GRAD_ACCUM = 4
BETA = 0.2             # v7: 0.5 -> ブレーキを緩める

def main():
    print(f"🏥 DPO v7.1 攻めた外科的矯正 開始: {TIMESTAMP}")
    print(f"   変更点: Beta={BETA} (v7: 0.5), LR={LEARNING_RATE} (v7: 5e-7)")
    print(f"   維持: layers_to_transform=[27-35], r=8, q_proj+v_proj only")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"   Loading Base Model: {BASE_MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"   Merging SFT Adapter: {SFT_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()
    print("   Base Model ready (Legacy SFT merged).")
    
    dataset = load_dataset("json", data_files=DPO_DATA_PATH, split="train")
    
    def format_dpo(sample):
        return {
            "prompt": "### 指示\n" + sample["prompt"] + "\n\n### 応答\n",
            "chosen": sample["chosen"] + tokenizer.eos_token,
            "rejected": sample["rejected"] + tokenizer.eos_token,
        }
    dataset = dataset.map(format_dpo)
    print(f"   Formatted {len(dataset)} pairs.")
    
    # LoRA Config: v7 と同じ（知識保護は維持）
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=list(range(27, 36)),  # 後半9層のみ
        task_type="CAUSAL_LM",
        lora_dropout=0.1,
        bias="none",
    )
    
    print(f"   LoRA: r={peft_config.r}, layers={peft_config.layers_to_transform}")
    
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
        beta=BETA,
        logging_steps=10,
    )
    
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    print(f"🚀 DPO v7.1 学習開始 (Steps={NUM_STEPS}, LR={LEARNING_RATE}, Beta={BETA})")
    trainer.train()
    
    print(f"💾 Saving Adapter -> {FINAL_ADAPTER_DIR}")
    trainer.save_model(FINAL_ADAPTER_DIR)
    print("🎉 DPO v7.1 完了!")

if __name__ == "__main__":
    main()
