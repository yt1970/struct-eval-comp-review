"""
V12: DPO3と完全同一条件で0.75モデルをDPO学習
- DPO3の成功を0.75ベースで再現する
- 0.73→0.77 (DPO3) の実績を 0.75→0.79+ に
"""
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

# === DPO3と完全同一の設定 ===
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# 0.75モデル (SFT 4000件, q/v only)
SFT_075_ADAPTER = "adapters/adapter_legacy_sft_20260209_0709"

# DPO3と同じデータ（Google Driveから持ってきたもの）
DPO_DATA_PATH = "/Users/yutako/Downloads/dpo_train.jsonl"

# 出力
OUTPUT_DIR = "outputs/v12_dpo_on_075"
FINAL_MERGED_DIR = "models/v12_075_dpo_merged"

# DPO3と同じハイパーパラメータ
LEARNING_RATE = 5e-6       # DPO3と同じ (失敗版は5e-7で低すぎた)
NUM_TRAIN_EPOCHS = 3       # DPO3と同じ
BATCH_SIZE = 1
GRAD_ACCUM = 8             # DPO3と同じ
BETA = 0.1                 # DPO3と同じ
MAX_SEQ_LENGTH = 1024      # DPO3と同じ
MAX_PROMPT_LENGTH = 512    # DPO3と同じ
WARMUP_RATIO = 0.1         # DPO3と同じ

def main():
    print("=" * 60)
    print("🔥 V12: DPO3完全再現 on 0.75ベース")
    print("   0.73→0.77 の実績を 0.75→0.79+ に")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. データロード
    print("\n[1/5] DPOデータロード中...")
    dataset = load_dataset("json", data_files=DPO_DATA_PATH, split="train")
    # DPO3と同じく train/eval を分割
    split = dataset.train_test_split(test_size=0.1, seed=3407)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # 2. ベースモデル + 0.75 SFTアダプタをロード & マージ
    print("\n[2/5] ベースモデル + 0.75 SFTアダプタをロード中...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)

    # 0.75 SFTをマージして土台にする
    print(f"  0.75 SFTアダプタをマージ: {SFT_075_ADAPTER}")
    model = PeftModel.from_pretrained(base_model, SFT_075_ADAPTER)
    model = model.merge_and_unload()
    print("  ✅ 0.75 SFTマージ完了")

    # 3. DPO3と同じLoRA構成で新しいLoRA層を追加
    # DPO3: target_modules = q, k, v, o, gate, up, down (全7モジュール)
    print("\n[3/5] DPO3と同じLoRA構成を追加中...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. DPO学習
    print("\n[4/5] DPO学習開始...")
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=1,
        optim="adamw_torch",  # Mac MPSではadamw_8bitは未対応
        beta=BETA,
        report_to="none",
        remove_unused_columns=False,
        save_strategy="no",
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # DPO3と同じ (メモリ節約)
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("🚀 DPO学習スタート！")
    trainer.train()

    # 5. マージして完全モデルとして保存
    print("\n[5/5] 完全マージモデルとして保存中...")
    merged_model = trainer.model.merge_and_unload()
    os.makedirs(FINAL_MERGED_DIR, exist_ok=True)
    merged_model.save_pretrained(FINAL_MERGED_DIR)
    tokenizer.save_pretrained(FINAL_MERGED_DIR)

    # tokenizer_config.json をベースモデル版で上書き (Colab互換)
    import shutil, glob
    base_tc = glob.glob("models/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/tokenizer_config.json")
    if base_tc:
        shutil.copy(base_tc[0], os.path.join(FINAL_MERGED_DIR, "tokenizer_config.json"))

    print("\n" + "=" * 60)
    print("🎉 V12 完了！")
    print(f"   保存先: {FINAL_MERGED_DIR}")
    print("   次: HFにアップロード → Colabで推論")
    print("=" * 60)

if __name__ == "__main__":
    main()
