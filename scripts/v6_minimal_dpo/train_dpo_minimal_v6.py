
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import DPOConfig, DPOTrainer

# --- Configuration (DPO v6: The Minimal Silence) ---
TIMESTAMP = "20260212_Minimal_v6"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "adapters/adapter_legacy_sft_20260209_0709" 
DPO_DATA_PATH = "data/dpo_minimal_v6.jsonl"
OUTPUT_DIR = f"outputs/dpo_minimal_v6_{TIMESTAMP}"
FINAL_ADAPTER_DIR = f"adapters/adapter_dpo_minimal_v6"

# Hyperparameters (さらっと学習するための設定)
NUM_STEPS = 75        # 300件 / (Batch 1 * Accum 4) = 75ステップ（約1周分）
LEARNING_RATE = 1e-6  
BATCH_SIZE = 1
GRAD_ACCUM = 4
BETA = 0.1

def main():
    print(f"🤐 Starting Minimal DPO Training (v6): {TIMESTAMP}")
    
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load Model & Merge SFT Adapter
    # SFT済みモデルを「ベース」として、その差分でDPOを学習させるためマージします
    print(f"   Loading Base Model: {BASE_MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"   Blending SFT Adapter: {SFT_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()
    print("   ✅ Base Model prepared (Legacy SFT merged).")

    # 3. Load & Format Dataset
    dataset = load_dataset("json", data_files=DPO_DATA_PATH, split="train")
    
    def format_dpo(sample):
        return {
            "prompt": f"### 指示\n{sample['prompt']}\n\n### 応答\n",
            "chosen": sample["chosen"] + tokenizer.eos_token,
            "rejected": sample["rejected"] + tokenizer.eos_token,
        }
    dataset = dataset.map(format_dpo)
    print(f"   Formatted {len(dataset)} pairs.")

    # 4. LoRA Config for DPO
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
    )

    # 5. DPO Config & Trainer
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        max_steps=NUM_STEPS,
        save_strategy="no", # 1周だけなので保存は最後のみ
        remove_unused_columns=False,
        bf16=False,
        report_to="none",
        beta=BETA
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("🚀 Running Minimal DPO...")
    trainer.train()

    print(f"💾 Saving Final DPO Adapter to {FINAL_ADAPTER_DIR}...")
    trainer.save_model(FINAL_ADAPTER_DIR)
    print("🎉 DPO v6 Minimal Complete!")

if __name__ == "__main__":
    main()
