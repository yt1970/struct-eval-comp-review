"""
SFT v8.1: lm_head LoRA のみ（attention 凍結）
- v8 で modules_to_save による lm_head フル学習で崩壊 → LoRA で微調整に変更
- lm_head に rank=4 の LoRA のみ適用、attention 層は一切触らない
- データ: sft_v8_lmhead.jsonl (300件, DPO v7 chosen由来)
- Epoch ごとにチェックポイント保存
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

# === Config ===
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "adapters/adapter_legacy_sft_20260209_0709"
SFT_DATA_PATH = "data/sft_v8_lmhead.jsonl"
OUTPUT_DIR = "outputs/sft_v8_1_lmhead_lora"
ADAPTER_BASE = "adapters/adapter_sft_v8_1_lmhead_lora"

NUM_EPOCHS = 3
LEARNING_RATE = 5e-5   # LoRA は小さいので LR 高めでOK
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_SEQ_LEN = 2048
LORA_RANK = 4          # 最小限の変更: 4096×4 + 4×151936 ≈ 62万パラメータ

def main():
    print("=" * 60)
    print("SFT v8.1: lm_head LoRA のみ (attention 凍結)")
    print(f"  LoRA rank={LORA_RANK}, LR={LEARNING_RATE}")
    print("=" * 60)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Base model + SFT v2 merge
    print("\n📦 Loading base model + SFT v2 adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()
    print("   SFT v2 merged!")

    # 3. LoRA: lm_head のみ
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=8,          # 2r
        target_modules=["lm_head"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(f"\n🔧 LoRA: lm_head のみ, rank={LORA_RANK}, alpha=8")
    print(f"   学習パラメータ: 約{(4096*LORA_RANK + LORA_RANK*151936)//1000}K")

    # 4. Dataset
    dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
    print(f"\n📊 Dataset: {len(dataset)} records")

    steps_per_epoch = len(dataset) // (BATCH_SIZE * GRAD_ACCUM)
    print(f"   Steps/epoch: {steps_per_epoch}, Total: {steps_per_epoch * NUM_EPOCHS}")

    # 5. Training config
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        max_length=MAX_SEQ_LEN,
        bf16=False,
        fp16=False,  # MPS 対応
        gradient_checkpointing=False,
        report_to="none",
        seed=42,
    )

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 7. Train
    print("\n🚀 Training start...")
    trainer.train()
    print("\n✅ Training complete!")

    # 8. Save final adapter
    final_path = f"{ADAPTER_BASE}_epoch3"
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"💾 Final adapter -> {final_path}")

    # 9. Copy epoch 1, 2 checkpoints
    for epoch in [1, 2]:
        ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{steps_per_epoch * epoch}")
        dest_dir = f"{ADAPTER_BASE}_epoch{epoch}"
        if os.path.exists(ckpt_dir):
            import shutil
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            shutil.copytree(ckpt_dir, dest_dir)
            print(f"💾 Epoch {epoch} adapter -> {dest_dir}")
        else:
            print(f"⚠️  Checkpoint not found: {ckpt_dir}")

    print("\n🎉 SFT v8.1 完了!")

if __name__ == "__main__":
    main()
