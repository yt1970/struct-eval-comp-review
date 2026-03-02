import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# --- CONFIG ---
VERSION = "V3_1_Deep_Silence"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "adapters/adapter_sft_v8_5_anti_hallucination_step100"
DPO_DATA_PATH = f"data/dpo_anti_chatty_{VERSION}.jsonl"
OUTPUT_DIR = f"outputs/dpo_anti_chatty_{VERSION}"
FINAL_ADAPTER_DIR = f"adapters/adapter_dpo_anti_chatty_{VERSION}"

# Hyperparameters (More Aggressive for V3.1)
BETA = 0.05           
LEARNING_RATE = 1e-5  # V3 の 2 倍に強化
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_STEPS = 100       # 500件のデータをほぼ一周させる
MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 768

def main():
    print("=" * 60)
    print(f"🚀 Anti-Chatty DPO Phase ({VERSION})")
    print(f"   Mode: Deep Silence (High LR, Concentrated Data)")
    print(f"   LR: {LEARNING_RATE}, Beta: {BETA}, Steps: {MAX_STEPS}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n📦 Loading model (FP32)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float32, device_map="mps"
    )
    
    # SFT v8.5 をベースに
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()
    print("   SFT v8.5 merged as base.")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset("json", data_files=DPO_DATA_PATH, split="train")
    print(f"📊 Concentrated Dataset: {len(dataset)} records")

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="steps",
        save_steps=20,      # 20, 40, 60, 80, 100 で保存
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

    print("\n🔥 Deep Silence Training start...")
    trainer.train()
    print("\n✅ Training complete!")

    trainer.model.save_pretrained(FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)
    print(f"💾 Final Adapter -> {FINAL_ADAPTER_DIR}")

if __name__ == "__main__":
    main()
