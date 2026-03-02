import os
import torch
from datasets import load_dataset
# Unsloth is highly recommended for Colab T4/L4 as it reduces VRAM usage and speeds up training.
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOConfig, DPOTrainer

# ==============================================================================
# V13 DPO Training Script (Cloud/Colab L4 Version): "The Silent Brain"
# Strategy: Hidden CoT DPO on 0.75 SFT Base (Full 7-Modules, Memory Optimized)
# ==============================================================================

# 1. Configuration
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507" 
SFT_ADAPTER_ID = "satoyutaka/LLM2026_SFT_0_again" 
# Use standard relative/absolute paths expected in Colab
TRAIN_DATA = "data/v13_silent_dpo_train.jsonl"
EVAL_DATA = "data/v13_silent_dpo_eval.jsonl"
OUTPUT_DIR = "checkpoint_v13_colab"

# Hyperparameters
MAX_SEQ_LENGTH = 1024 
LORA_R = 16
LORA_ALPHA = 16
LEARNING_RATE = 2e-6
NUM_EPOCHS = 3
BETA = 0.1

def main():
    # Patch for DPO (Must be called before initializing Trainer when using Unsloth)
    PatchDPOTrainer()

    print(f"🚀 Initializing V13 DPO Training (Colab T4/L4 Strategy)...")

    # 2. Load Base Model & Tokenizer with 4-bit quantization (Saves massive VRAM)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL_ID,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None, # Auto detection
        load_in_4bit = True, # Crucial for T4/L4 GPUs
    )

    # 3. Load SFT Adapter and Merge
    print(f"🔗 Loading and Merging SFT Adapter: {SFT_ADAPTER_ID}...")
    model.load_adapter(SFT_ADAPTER_ID)
    # 4. Add NEW LoRA Adapters for DPO
    print("➕ Adding new LoRA layers for DPO training (All 7 Modules)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_R,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = LORA_ALPHA,
        lora_dropout = 0, # Unsloth optimizes dropout=0
        bias = "none",
        use_gradient_checkpointing = "unsloth", # This alone saves ~30% VRAM!
        random_state = 3407,
    )

    # 5. Load Dataset
    print("📊 Loading datasets...")
    train_dataset = load_dataset("json", data_files=TRAIN_DATA, split="train")
    eval_dataset = load_dataset("json", data_files=EVAL_DATA, split="train")

    # 6. Training Arguments (Updated for latest TRL compatibility)
    training_args = DPOConfig(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        learning_rate = LEARNING_RATE,
        num_train_epochs = NUM_EPOCHS,
        beta = BETA,
        logging_steps = 10,
        eval_strategy = "steps", # Fixed from legacy evaluation_strategy
        eval_steps = 50,
        save_strategy = "no",
        lr_scheduler_type = "cosine",
        warmup_steps = 10, # Fixed from deprecated warmup_ratio
        max_grad_norm = 1.0, # Safety net against NaN
        weight_decay = 0.05,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        optim = "adamw_8bit",
        max_length = MAX_SEQ_LENGTH, # Fixed for latest TRL
        max_prompt_length = 512,
        report_to = "none",
    )

    # 7. Initialize Trainer
    dpo_trainer = DPOTrainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        processing_class = tokenizer, # Fixed from legacy tokenizer argument
    )

    # 8. Start Training
    print("🔥 Starting V13 DPO Training Loop...")
    dpo_trainer.train()

    # 9. Save the Adapter
    final_adapter_dir = "v13_silent_dpo_adapter_colab"
    print(f"💾 Saving LoRA adapter to: {final_adapter_dir}")
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    print("\n" + "="*50)
    print("✅ V13 DPO TRAINING COMPLETED!")
    print("="*50)

if __name__ == "__main__":
    main()
