import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOConfig, DPOTrainer

# ==============================================================================
# V13 DPO Training Script (Local MPS Version): "The Silent Brain"
# ==============================================================================

# 1. Configuration (Local Paths)
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_ID = "satoyutaka/LLM2026_SFT_0_again" # 0.75 SFT Adapter
V13_DIR = "/Users/yutako/dev/struct-eval-comp/experiments/exp_20260228_v13_silent_dpo"
TRAIN_DATA = os.path.join(V13_DIR, "data/v13_silent_dpo_train.jsonl")
EVAL_DATA = os.path.join(V13_DIR, "data/v13_silent_dpo_eval.jsonl")
OUTPUT_DIR = os.path.join(V13_DIR, "checkpoint_local")

# Hyperparameters
MAX_SEQ_LENGTH = 1024
LEARNING_RATE = 2e-6
NUM_EPOCHS = 3
BETA = 0.1

def main():
    # Set device to MPS for Mac
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Running V13 DPO on: {device}")

    # 2. Load Base Model & Tokenizer
    print(f"🔹 Loading Base Model: {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load in float16 for Mac MPS compatibility and memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map=None,
    ).to(device)
    
    # Enable Gradient Checkpointing for memory safety on MPS
    model.config.use_cache = False 
    model.gradient_checkpointing_enable()

    # 3. Load SFT Adapter and Merge
    print(f"🔗 Loading SFT Adapter: {SFT_ADAPTER_ID}...")
    # NOTE: from_pretrained here might trigger the same PEFT issues if we don't handle them
    # But on a clean local env with latest peft, it should be fine.
    try:
        model = PeftModel.from_pretrained(model, SFT_ADAPTER_ID)
        print("🔗 Merging into base model...")
        model = model.merge_and_unload()
        print("✅ SFT Adapter merged.")
    except Exception as e:
        print(f"⚠️ Warning during merge: {e}")
        print("If PEFT config error occurs, we may need to manually update the adapter config.")
        raise e

    # 4. Set up Peft for DPO
    print("➕ Preparing new LoRA for DPO training...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # 5. Load Dataset
    print("📊 Loading local datasets...")
    train_dataset = load_dataset("json", data_files=TRAIN_DATA, split="train")
    eval_dataset = load_dataset("json", data_files=EVAL_DATA, split="train")

    # 6. Training Arguments (MPS Optimized)
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, # Batch size 1 + Accumulation is safer on Mac
        gradient_accumulation_steps=8,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        beta=BETA,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="no",
        lr_scheduler_type="cosine",
        warmup_steps=10, 
        max_grad_norm=1.0, # Prevent NaN loss on MPS
        gradient_checkpointing=True, # Enable checkpointing in DPOConfig
        fp16=True, 
        bf16=False,
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=512,
        remove_unused_columns=False,
        report_to="none",
    )

    # 7. Initialize Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # 8. Start Training
    print("🔥 Starting V13 DPO local training loop...")
    dpo_trainer.train()

    # 9. Save final Adapter
    final_dir = os.path.join(V13_DIR, "v13_silent_dpo_adapter_local")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"✅ Training completed! Saved to: {final_dir}")

if __name__ == "__main__":
    main()
