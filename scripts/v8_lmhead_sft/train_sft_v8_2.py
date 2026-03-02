"""
SFT v8.2: lm_head LoRA rank=16（attention 凍結）
- v8.1 (rank=4) では Markdown 除去が不十分 → rank を 4→16 に上げて再挑戦
- パラメータ数: 4096×16 + 16×151936 ≈ 約250万パラメータ
- v8 (modules_to_save, 6.2億) と v8.1 (rank=4, 62万) の中間を狙う
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
OUTPUT_DIR = "outputs/sft_v8_2_lmhead_lora"
ADAPTER_BASE = "adapters/adapter_sft_v8_2_lmhead_lora"

NUM_EPOCHS = 3
LEARNING_RATE = 3e-5   # rank 大きいので v8.1 (5e-5) より少し下げる
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_SEQ_LEN = 2048
LORA_RANK = 16         # v8.1 の 4 → 16 に拡大
LORA_ALPHA = 32        # 2r

def main():
    print("=" * 60)
    print("SFT v8.2: lm_head LoRA rank=16 (attention 凍結)")
    print(f"  LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}, LR={LEARNING_RATE}")
    params = 4096 * LORA_RANK + LORA_RANK * 151936
    print(f"  学習パラメータ: 約{params//1000}K ({params/624000:.1f}x v8.1)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n📦 Loading base model + SFT v2 adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()
    print("   SFT v2 merged!")

    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["lm_head"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
    print(f"\n📊 Dataset: {len(dataset)} records")

    steps_per_epoch = len(dataset) // (BATCH_SIZE * GRAD_ACCUM)
    print(f"   Steps/epoch: {steps_per_epoch}, Total: {steps_per_epoch * NUM_EPOCHS}")

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
        fp16=False,
        gradient_checkpointing=False,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\n🚀 Training start...")
    trainer.train()
    print("\n✅ Training complete!")

    final_path = f"{ADAPTER_BASE}_epoch3"
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"💾 Final adapter -> {final_path}")

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

    print("\n🎉 SFT v8.2 完了!")

if __name__ == "__main__":
    main()
