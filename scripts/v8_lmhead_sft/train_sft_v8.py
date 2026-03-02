"""
SFT v8: lm_head + 後半層 Attention LoRA
- ベース: SFT v2 マージ済み (0.751 スコア)
- 学習対象: lm_head + 後半9層の q_proj, v_proj
- Epoch ごとにチェックポイント保存
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTConfig, SFTTrainer

# === Config ===
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "adapters/adapter_legacy_sft_20260209_0709"
SFT_DATA_PATH = "data/sft_v8_lmhead.jsonl"
OUTPUT_DIR = "outputs/sft_v8_lmhead"
ADAPTER_BASE = "adapters/adapter_sft_v8_lmhead"

NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_SEQ_LEN = 2048

def main():
    print("=" * 50)
    print("SFT v8: lm_head + Attention 外科的スタイル矯正")
    print("=" * 50)

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load base model + SFT v2 adapter -> merge
    print("\n📦 Loading base model + SFT v2 adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()
    print("   SFT v2 merged!")

    # 3. Configure LoRA for lm_head + attention
    # lm_head は layers_to_transform の対象外なので自動的に全体に適用される
    # q_proj, v_proj は後半9層のみ
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=list(range(27, 36)),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head"],  # lm_head は LoRA ではなく直接学習
    )
    print(f"\n🔧 LoRA Config:")
    print(f"   target_modules: q_proj, v_proj (layers 27-35)")
    print(f"   modules_to_save: lm_head (直接学習)")
    print(f"   r={peft_config.r}, alpha={peft_config.lora_alpha}")

    # 4. Load dataset
    dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
    print(f"\n📊 Dataset: {len(dataset)} records")

    # 5. Calculate steps per epoch for checkpoint saving
    effective_batch = BATCH_SIZE * GRAD_ACCUM
    steps_per_epoch = len(dataset) // effective_batch
    total_steps = steps_per_epoch * NUM_EPOCHS
    print(f"   Steps/epoch: {steps_per_epoch}, Total: {total_steps}")

    # 6. Training config
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",  # Epoch ごとにチェックポイント保存
        save_total_limit=3,
        max_length=MAX_SEQ_LEN,
        bf16=False,
        fp16=False,  # MPS では GradScaler が FP16 勾配を扱えないため無効化
        gradient_checkpointing=False,
        report_to="none",
        seed=42,
    )

    # 7. Create trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 8. Train!
    print("\n🚀 Training start...")
    trainer.train()
    print("\n✅ Training complete!")

    # 9. Save final adapter
    final_path = f"{ADAPTER_BASE}_epoch3"
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"💾 Final adapter -> {final_path}")

    # 10. Also copy epoch 1 and 2 checkpoints to named directories
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

    print("\n🎉 SFT v8 完了!")

if __name__ == "__main__":
    main()
