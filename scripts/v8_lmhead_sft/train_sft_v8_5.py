"""
SFT v8.5: Integrated Anti-Hallucination SFT
- Dataset: data/sft_v8_5_silent.jsonl (1,000 records with entity anonymization)
- LoRA Target: q_proj, v_proj (Attention/Reasoning) + lm_head (Style/Tone)
- LoRA Params: rank=64, alpha=16 (alpha を低く抑えてハルシネーション（暗記）を抑制)
- Epoch: 1 (約250ステップ。ジェミサンの「100ステップ」という助言を考慮し、短期間で学習)
"""
import os
# Mac MPS のメモリ制限を緩和（システム全体が不安定になるリスクはあるが OOM 回避を優先）
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

# === Config ===
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "adapters/adapter_legacy_sft_20260209_0709"
SFT_DATA_PATH = "data/sft_v8_5_silent.jsonl"
OUTPUT_DIR = "outputs/sft_v8_5_anti_hallucination"
ADAPTER_BASE = "adapters/adapter_sft_v8_5_anti_hallucination"

NUM_EPOCHS = 1
LEARNING_RATE = 3e-5
BATCH_SIZE = 1
GRAD_ACCUM = 8          # 4 -> 8 に増やしてステップを安定化
MAX_SEQ_LEN = 1024      # 2048 -> 1024 に縮小（メモリ節約）
LORA_RANK = 64
LORA_ALPHA = 16

def main():
    print("=" * 60)
    print("SFT v8.5: Integrated Anti-Hallucination SFT")
    print(f"  Target Modules: q_proj, v_proj, lm_head")
    print(f"  LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}, LR={LEARNING_RATE}")
    print(f"  Epochs={NUM_EPOCHS}, Data={SFT_DATA_PATH}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n📦 Loading base model + SFT v2 adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    # v2 の知識をベースに、スタイルだけを v8.5 で上書きする
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()
    print("   SFT v2 merged!")

    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "lm_head"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
    print(f"\n📊 Dataset: {len(dataset)} records")

    steps_per_epoch = len(dataset) // (BATCH_SIZE * GRAD_ACCUM)
    print(f"   Steps/epoch: {steps_per_epoch}")

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,         # 50歩ごとに保存して 100歩付近も選べるようにする
        save_total_limit=10,
        max_length=MAX_SEQ_LEN,
        bf16=False,
        fp16=True,
        gradient_checkpointing=True, # メモリ節約に必須
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

    # 最終モデルの保存
    final_path = f"{ADAPTER_BASE}_final"
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"💾 Final adapter -> {final_path}")

    # ジェミサン推奨の「100ステップ」付近のチェックポイントを抽出
    # 50, 100, 150, 200, 250(final) が保存されるはず
    for step in [50, 100, 150, 200]:
        ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{step}")
        dest_dir = f"{ADAPTER_BASE}_step{step}"
        if os.path.exists(ckpt_dir):
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            shutil.copytree(ckpt_dir, dest_dir)
            print(f"💾 Step {step} adapter -> {dest_dir}")

    print("\n🎉 SFT v8.5 学習完了!")

if __name__ == "__main__":
    main()
