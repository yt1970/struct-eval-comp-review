import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig

# --- 設定 ---
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
# 今のDPO版をベースに、さらに構造を矯正
SFT_ADAPTER_PATH = "./adapters/adapter_dpo_fix_10674" 
DPO_DATA_PATH = "data/train_data/dpo_flat_structure_v2.jsonl"
OUTPUT_DIR = f"outputs/train_dpo_flat_{os.getpid()}"
ADAPTER_OUT_DIR = f"adapters/adapter_dpo_flat_final"

# 構造矯正のための超短距離DPO
MAX_STEPS = 40 
LEARNING_RATE = 2e-6 # 極めて慎重に
BATCH_SIZE = 1
GRAD_ACCUM = 4

def main():
    dataset = load_dataset("json", data_files=DPO_DATA_PATH, split="train")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": "mps"}
    )
    
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_PATH, is_trainable=True)
    
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        logging_steps=5,
        save_steps=MAX_STEPS,
        save_total_limit=1,
        remove_unused_columns=False,
        bf16=False,
        gradient_checkpointing=True,
        eval_strategy="no",
        report_to="none",
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
    )

    print("🚀 最終構造矯正DPO開始...")
    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    dpo_trainer.train()

    print(f"💾 最終アダプタ保存中... -> {ADAPTER_OUT_DIR}")
    dpo_trainer.model.save_pretrained(ADAPTER_OUT_DIR)
    print("✅ 全ての工程が完了しました！")

if __name__ == "__main__":
    main()
