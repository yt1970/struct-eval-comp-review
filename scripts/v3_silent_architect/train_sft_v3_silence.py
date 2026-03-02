
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# --- Configuration (SFT v3: The Silent Architect) ---
TIMESTAMP = "20260211_Silence_v3"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Generated Silence Dataset
DATA_PATH = "data/sft_train_data_v3_silence_20260211_1511.jsonl"

OUTPUT_DIR = f"outputs/train_sft_v3_{TIMESTAMP}"
ADAPTER_OUT_DIR = f"adapters/adapter_sft_v3_{TIMESTAMP}"

# Training Hyperparameters
# User requested "ensure enough training volume"
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2          # 4500 samples * 2 epochs = Sufficient to overwrite behavior
BATCH_SIZE = 1          # Per device
GRAD_ACCUM = 8          # Effective batch size = 8
MAX_SEQ_LENGTH = 1024   # Handle long JSONs

def main():
    print(f"🔇 Starting SFT v3 Training (Silence Pre-training): {TIMESTAMP}")
    print(f"   Dataset: {DATA_PATH}")
    print(f"   Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")
    
    # 1. Load Dataset
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"   Loaded {len(dataset)} samples.")

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Base Model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto" # or {"": "mps"} if specific
    )
    
    # 4. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], # Reverted to Legacy setting as per user request!
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
    )
    
    # 5. Training Args
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        dataset_text_field="text",
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        remove_unused_columns=False,
        bf16=False, # Use float16 on Mac/MPS to be safe or bf16 if supported
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer, 
    )

    print("🚀 Starting SFT v3 Training...")
    trainer.train()

    print(f"💾 Saving SFT v3 Adapter to {ADAPTER_OUT_DIR}...")
    trainer.save_model(ADAPTER_OUT_DIR)
    print("🎉 SFT v3 Complete!")

if __name__ == "__main__":
    main()
