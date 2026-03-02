import os
# Avoid AttributeError: module 'huggingface_hub.constants' has no attribute 'HF_HUB_ENABLE_HF_TRANSFER'
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
from unsloth import FastLanguageModel, PatchDPOTrainer
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import TrainingArguments
from peft import PeftModel

# ==============================================================================
# V13 DPO Training Script: "The Silent Brain"
# Strategy: Hidden CoT DPO on 0.75 SFT Base
# ==============================================================================

# 0. Patch for DPO (Must be before importing DPOTrainer)
PatchDPOTrainer()

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507" # The original Qwen base
SFT_ADAPTER_ID = "satoyutaka/LLM2026_SFT_0_again" # The 0.75 SFT Adapter
V13_DIR = "/content/drive/MyDrive/LLM2026/main_competition/DPO/exp_20260228_v13_silent_dpo"
TRAIN_DATA = "/content/drive/MyDrive/LLM2026/main_competition/DPO_data/v13_silent_dpo_train.jsonl"
EVAL_DATA = "/content/drive/MyDrive/LLM2026/main_competition/DPO_data/v13_silent_dpo_eval.jsonl"
OUTPUT_DIR = os.path.join(V13_DIR, "checkpoint")

# Hyperparameters
MAX_SEQ_LENGTH = 1024 # Buffer for CoT in 'rejected' samples
LORA_R = 16
LORA_ALPHA = 16
LEARNING_RATE = 2e-6 # Stable tuning for 0.75 base (now merged)
NUM_EPOCHS = 3
BETA = 0.1 # Standard DPO beta

def main():
    print(f"🚀 Initializing V13 DPO Training (CPU Pre-Merge Strategy)...")
    print(f"🔹 Base Model: {BASE_MODEL_ID}")
    print(f"🔹 SFT Adapter: {SFT_ADAPTER_ID}")

    # --- 1. SET UP SWAP SPACE & PRE-MERGE TO AVOID COLAB LIMITS ---
    import gc
    import subprocess
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Add 16GB of Swap space so Colab doesn't crash on 12.7GB system RAM limits
    print("🛠️ Setting up 16GB Swap space on Colab disk to prevent CPU OOM...")
    subprocess.run(["fallocate", "-l", "16G", "/swapfile"], check=False)
    subprocess.run(["chmod", "600", "/swapfile"], check=False)
    subprocess.run(["mkswap", "/swapfile"], check=False)
    subprocess.run(["swapon", "/swapfile"], check=False)

    print("🧠 Step 1: Pre-merging SFT adapter on CPU to save GPU VRAM...")
    
    # Load base tokenizer and save it later
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    # Load 16-bit model on CPU with low memory usage enabled
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="cpu",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    # Merge the 0.75 adapter purely in system RAM
    print(f"🔗 Merging SFT Adapter: {SFT_ADAPTER_ID} into base...")
    base_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_ID)
    base_model = base_model.merge_and_unload()
    
    
    # Save physically to Colab disk
    TEMP_MERGE_DIR = "/content/v13_merged_sft_temp"
    base_model.save_pretrained(TEMP_MERGE_DIR)
    tokenizer.save_pretrained(TEMP_MERGE_DIR)
    print("✅ Pre-merge complete! Saved model and tokenizer to local disk.")
    
    # Destroy objects and reclaim system RAM
    del base_model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- 2. LOAD MERGED MODEL ON GPU WITH UNSLOTH 4-BIT ---
    print("\n📦 Step 2: Loading merged model onto GPU in 4-bit for training...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = TEMP_MERGE_DIR,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True, # Crucial: Extremely VRAM lightweight for 7-module LoRA
    )

    # 3. Add NEW LoRA Adapters for DPO
    print("➕ Adding new LoRA layers for DPO training (All 7 modules)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_R,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = LORA_ALPHA,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 4. Load Dataset
    print("📊 Loading local datasets...")
    train_dataset = load_dataset("json", data_files=TRAIN_DATA, split="train")
    eval_dataset = load_dataset("json", data_files=EVAL_DATA, split="train")

    # 5. Training Arguments
    training_args = DPOConfig(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        learning_rate = LEARNING_RATE,
        num_train_epochs = NUM_EPOCHS,
        beta = BETA,
        logging_steps = 10,
        eval_strategy = "steps",
        eval_steps = 50,
        save_strategy = "no",
        lr_scheduler_type = "linear",
        warmup_steps = 10,
        weight_decay = 0.05,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        optim = "adamw_8bit",
        max_length = MAX_SEQ_LENGTH,
        max_prompt_length = 512,
        report_to = "none",
    )

    # 6. Initialize Trainer
    dpo_trainer = DPOTrainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        processing_class = tokenizer,
    )

    # 7. Start Training
    print("🔥 Starting V13 DPO Training Loop...")
    dpo_trainer.train()

    # 8. Save the Adapter
    final_adapter_dir = os.path.join(V13_DIR, "v13_silent_dpo_adapter")
    print(f"💾 Saving LoRA adapter to: {final_adapter_dir}")
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    print("\n" + "="*50)
    print("✅ V13 DPO TRAINING COMPLETED!")
    print(f"Please move '{final_adapter_dir}' to your merge script.")
    print("="*50)

if __name__ == "__main__":
    main()
