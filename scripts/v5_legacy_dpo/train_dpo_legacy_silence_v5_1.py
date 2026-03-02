
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import DPOConfig, DPOTrainer

# --- Configuration ---
# DPO v5.1: The Silence of the Legend (Retry with Clean Data)
TIMESTAMP = "20260211_Silence_v5_1_Clean"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Critical: Use SFT v2 Adapter as Base (SFT Rebirth)
ADAPTER_PATH = "adapters/adapter_legacy_sft_20260209_0709" 

# Generated Hybrid Dataset (VERIFIED CLEAN)
DATA_PATH = "data/dpo_synthetic_dataset_v5_20260211_1100.jsonl"

OUTPUT_DIR = f"outputs/dpo_legacy_silence_{TIMESTAMP}"
ADAPTER_OUT_DIR = f"adapters/adapter_legacy_silence_{TIMESTAMP}"

# Aggressive Hyperparameters (Short & High-LR)
MAX_STEPS = 60
LEARNING_RATE = 2e-6
BATCH_SIZE = 1
GRAD_ACCUM = 8
BETA = 0.1

def main():
    print(f"🤐 Starting DPO v5.1 Training: {TIMESTAMP}")
    print(f"   Base Adapter: {ADAPTER_PATH} (SFT v2)")
    print(f"   Dataset: {DATA_PATH} (Cleaned Hybrid)")
    print(f"   Steps: {MAX_STEPS}, LR: {LEARNING_RATE}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model + SFT Adapter
    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load SFT Adapter onto Base Model
    print(f"Loading SFT Adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
    model.print_trainable_parameters() 
    
    print("🔄 Merging SFT Adapter into Base Model for DPO...")
    model = model.merge_and_unload()
    print("✅ Merge Complete. Model is now SFT v2 (Dense).")

    # 3. Prepare Dataset
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    def format_dpo(sample):
        # Already cleaned prompt/chosen/rejected in generation script.
        # Just format for Trainer.
        # Note: Do not add extra chat templates if prompt is already raw text?
        # DPOTrainer handles formatting if tokenizer is passed.
        # BUT we need to ensure EOS is appended to chosen/rejected.
        
        # Simple string concatenation for standard CausalLM DPO
        return {
            "prompt": f"### 指示\n{sample['prompt']}\n\n### 応答\n",
            "chosen": sample["chosen"] + tokenizer.eos_token,
            "rejected": sample["rejected"] + tokenizer.eos_token,
        }

    dataset = dataset.map(format_dpo)
    
    # 4. Config & Train
    # New Adapter for DPO
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
    )

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        save_strategy="steps",
        save_steps=20,
        remove_unused_columns=False,
        bf16=False,
        report_to="none",
        beta=BETA
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None, # Implicitly uses model copy as ref
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config, # Train NEW adapter
    )

    print("🚀 Starting DPO Training...")
    trainer.train()

    print(f"💾 Saving DPO v5.1 Adapter to {ADAPTER_OUT_DIR}...")
    trainer.save_model(ADAPTER_OUT_DIR)
    print("🎉 DPO v5.1 Complete!")

if __name__ == "__main__":
    main()
