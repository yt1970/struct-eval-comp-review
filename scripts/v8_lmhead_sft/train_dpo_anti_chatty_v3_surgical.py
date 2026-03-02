import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# --- CONFIG ---
TIMESTAMP = "20260215_1822"
VERSION = "V3_Surgical_Strike"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
# 元の "賢い" SFT v8.5 をベースにする
SFT_ADAPTER_PATH = "adapters/adapter_sft_v8_5_anti_hallucination_step100"
DPO_DATA_PATH = f"data/dpo_anti_chatty_{VERSION}.jsonl"
OUTPUT_DIR = f"outputs/dpo_anti_chatty_{VERSION}"
FINAL_ADAPTER_DIR = f"adapters/adapter_dpo_anti_chatty_{VERSION}"

# Hyperparameters (Aggressive & Quick)
BETA = 0.05           # 強制矯正
LEARNING_RATE = 5e-6  # 強めの学習率
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_STEPS = 30        # 短期決戦
MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 768

def main():
    print("=" * 60)
    print(f"🚀 Anti-Chatty DPO Phase ({VERSION})")
    print(f"   Mode: Surgical Strike with 10-step Checkpoints")
    print(f"   Base: SFT v8.5 (Back to Original smart model)")
    print(f"   LR: {LEARNING_RATE}, Beta: {BETA}, Steps: {MAX_STEPS}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n📦 Loading model (FP32 safety mode)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float32, device_map="mps"
    )
    
    # SFT v8.5 をマージしてベースにする
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()
    print("   SFT v8.5 intelligence merged into base!")

    # LoRA config (全層アタック)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset("json", data_files=DPO_DATA_PATH, split="train")
    print(f"\n📊 Dataset: {len(dataset)} records")

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant_with_warmup",
        max_steps=MAX_STEPS,
        warmup_steps=5,
        logging_steps=2,    # 2ステップごとにログ
        save_strategy="steps",
        save_steps=10,      # 10ステップごとに保存 (Step 10, 20, 30)
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        beta=BETA,
        bf16=False,
        fp16=False,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\n🔥 DPO Surgical Strike start...")
    trainer.train()
    print("\n✅ Training complete!")

    # 最終保存
    trainer.model.save_pretrained(FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)
    print(f"💾 Final Surgical Adapter -> {FINAL_ADAPTER_DIR}")

if __name__ == "__main__":
    main()
