import os
import torch
import glob
import logging
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from datetime import datetime
import pytz

def get_timestamp_jst():
    jst = pytz.timezone('Asia/Tokyo')
    return datetime.now(jst).strftime("%Y%m%d_%H%M")

# --- 設定 ---
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATA_PATH = "data/train_data/dpo_pure_format.jsonl"
TIMESTAMP = get_timestamp_jst()
OUTPUT_DIR = f"./outputs/dpo_final_{TIMESTAMP}"
FINAL_ADAPTER_DIR = f"./adapters/adapter_dpo_final_{TIMESTAMP}"
LOG_FILE = f"{OUTPUT_DIR}/dpo_training.log"

SAMPLE_SIZE = 300 

# --- 最新のSFTアダプタを自動探索 ---
adapter_dirs = glob.glob("./adapters/adapter_final_*")
if not adapter_dirs:
    print("❌ SFT済みのアダプタが見つかりません！ adapters/ フォルダを確認してください。")
    exit(1)
    
LATEST_SFT_ADAPTER = max(adapter_dirs, key=os.path.getmtime)
print(f"✅ 最新のSFTアダプタを使用します: {LATEST_SFT_ADAPTER}")

# --- ロギング設定 ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

# 1. データセットの準備
def prepare_dpo_data():
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    if len(dataset) > SAMPLE_SIZE:
        print(f"🔥 データセットを {len(dataset)} から {SAMPLE_SIZE} 件にランダムサンプリングします...")
        dataset = dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))
    return dataset

train_dataset = prepare_dpo_data()

# 2. モデルとトークナイザー
tokenizer = AutoTokenizer.from_pretrained(LATEST_SFT_ADAPTER)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ベースモデルロード
# メモリ節約のために gradient_checkpointing を有効化する
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map={"": "mps"}
)

# SFTアダプタを合体
model = PeftModel.from_pretrained(base_model, LATEST_SFT_ADAPTER, is_trainable=True)

# 3. DPO設定
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    max_steps=100,
    logging_steps=5,
    
    # --- 修正: 保存設定を追加 ---
    save_strategy="steps",
    save_steps=10,          # 10ステップごとに保存
    save_total_limit=2,     # 最新2つだけ残す
    
    beta=0.1,
    fp16=True,
    report_to="none",
    max_length=2048,
    max_prompt_length=1024,
    gradient_checkpointing=True, # メモリ節約
    remove_unused_columns=False, # Datasetの列名エラー防止
)

# 4. Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None, # PEFTの場合自動処理
    args=dpo_config,
    train_dataset=train_dataset,
    processing_class=tokenizer, # v0.8.0以降の引数名に対応
)

print("\n🚀 DPO学習（最終仕上げ）を開始します！")
print("   - 目標: Markdownと余計な思考の完全削除")
print(f"   - ステップ数: {dpo_config.max_steps}")
print(f"   - 保存頻度: {dpo_config.save_steps} ステップごと")
print("-" * 50)

trainer.train()

# 5. 最終保存
print(f"\n📦 DPO完了モデルを保存中: {FINAL_ADAPTER_DIR}")
trainer.model.save_pretrained(FINAL_ADAPTER_DIR)
tokenizer.save_pretrained(FINAL_ADAPTER_DIR)
print("✅ 全工程完了！お疲れ様でした。")
