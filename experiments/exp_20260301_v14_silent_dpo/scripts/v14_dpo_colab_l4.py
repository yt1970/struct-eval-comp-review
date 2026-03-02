# --- 安定版インストール (HuggingFace標準) ---
# !pip install -U trl peft transformers accelerate bitsandbytes datasets
# ---------------------------------------------

import os
import torch
import gc
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments
)
from peft import PeftModel, LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer
from huggingface_hub import login

try:
    from google.colab import drive
    from google.colab import userdata
    from google.colab import runtime
    drive.mount('/content/drive')
    hf_token = userdata.get('HF_TOKEN')
    login(hf_token)
except ImportError:
    runtime = None
    print("Not running in Colab environment.")

WORK_DIR = '/content/drive/MyDrive/LLM2026/main_competition'
V14_DIR = os.path.join(WORK_DIR, "DPO/exp_20260301_v14_silent_dpo")
NEW_MODEL = os.path.join(V14_DIR, "v14_silent_dpo_adapter")
OUTPUT_DIR = os.path.join(V14_DIR, "checkpoint")
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_ID = "satoyutaka/LLM2026_SFT_0_again"
DATASET_PATH = os.path.join(WORK_DIR, "data/v14_silent_dpo_train.jsonl")
HF_UPLOAD_REPO = "satoyutaka/LLM2026_v14_silent_dpo_adapter"

def main():
    print("🚀 Initiating V14 DPO Training (GPU-Only Merge Strategy for T4)...")

    # --- 1. Load Base + Adapter and Merge ON GPU (To avoid CPU RAM Crash) ---
    print(f"📦 Loading {BASE_MODEL_ID} into GPU (16-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load directly to GPU (8GB VRAM)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"🔗 Merging SFT Adapter: {SFT_ADAPTER_ID} within VRAM...")
    # Merge adapter immediately without saving to disk to save CPU RAM
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_ID)
    model = model.merge_and_unload()
    print("✅ Merge complete on GPU. System RAM is safe.")
    
    gc.collect()
    torch.cuda.empty_cache()

    # --- 2. Add New LoRA Adapters for DPO ---
    print("➕ Adding new LoRA layers for DPO training (All 7 modules)...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- 3. DPO Training Setup (Using TRL 0.12.0+ API) ---
    print("📊 Loading V14 Structural Silence dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # In TRL 0.12.0+, many arguments moved from Trainer to DPOConfig
    # max_prompt_length was REMOVED. Use only max_length.
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, # T4 Safety
        gradient_accumulation_steps=8, 
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=20, # 10% of 200 steps
        max_steps=200, 
        logging_steps=10,
        save_strategy="no",
        fp16=True, # Optimized for T4
        report_to="none",
        remove_unused_columns=False,
        beta=0.1,
        max_length=1024,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None, # PEFT DPO handles this automatically
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("🔥 Starting V14 DPO Loop...")
    dpo_trainer.train()

    # --- 4. Save & Upload Adapter ---
    print(f"💾 Saving adapter to {NEW_MODEL}...")
    os.makedirs(NEW_MODEL, exist_ok=True)
    dpo_trainer.model.save_pretrained(NEW_MODEL)
    tokenizer.save_pretrained(NEW_MODEL)
    
    print(f"🚀 Uploading to HF: {HF_UPLOAD_REPO}...")
    try:
        dpo_trainer.model.push_to_hub(HF_UPLOAD_REPO)
        tokenizer.push_to_hub(HF_UPLOAD_REPO)
        print("✅ Upload successful!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

    if runtime is not None:
        print("💡 Terminating Colab runtime...")
        runtime.unassign()

if __name__ == "__main__":
    main()
