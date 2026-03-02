import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# --- CONFIG ---
TIMESTAMP = "20260215_1822"
VERSION = "V2_AllLinear_Safe"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "adapters/adapter_sft_v8_5_anti_hallucination_step100"
DPO_DATA_PATH = f"data/dpo_anti_chatty_{TIMESTAMP}.jsonl"
OUTPUT_DIR = f"outputs/dpo_anti_chatty_{VERSION}"
FINAL_ADAPTER_DIR = f"adapters/adapter_dpo_anti_chatty_{VERSION}"

# Hyperparameters (Safe & Intensive)
BETA = 0.15           # 乖離制約を強め、知能低下を防止
LEARNING_RATE = 1e-6  # 極めて安全な学習率
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_STEPS = 100       # じっくり矯正 (約1.5時間見込み)
MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 768

def main():
    print("=" * 60)
    print(f"🚀 Anti-Chatty DPO Intensive Phase ({VERSION})")
    print(f"   Base: SFT v8.5 (Step 100)")
    print(f"   All-Linear Targeting (q,k,v,o,gate,up,down)")
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

    # LoRA config (All-Linear層をすべて学習対象に)
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
        lr_scheduler_type="cosine",
        max_steps=MAX_STEPS,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        beta=BETA,
        bf16=False,
        fp16=False, # MPS + DPO での NaN 回避のため False
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

    print("\n🔥 DPO Intensive Training start...")
    trainer.train()
    print("\n✅ Training complete!")

    # 保存
    trainer.model.save_pretrained(FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)
    print(f"💾 Final Intensive DPO adapter -> {FINAL_ADAPTER_DIR}")

if __name__ == "__main__":
    main()
