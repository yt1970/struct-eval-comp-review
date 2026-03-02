import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from datetime import datetime
import pytz

def get_timestamp_jst():
    jst = pytz.timezone('Asia/Tokyo')
    return datetime.now(jst).strftime("%Y%m%d_%H%M")

# --- 設定 ---
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
TRAIN_DATA_PATH = "data/golden_train.jsonl"
VAL_DATA_PATH = "data/golden_val.jsonl"
TIMESTAMP = get_timestamp_jst()
# 実行ごとに独立したフォルダを作成
OUTPUT_DIR = f"./outputs/train_{TIMESTAMP}"
FINAL_ADAPTER_DIR = f"./adapters/adapter_{TIMESTAMP}"
LOG_FILE = f"{OUTPUT_DIR}/training.log"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./adapters", exist_ok=True)

# --- ロギング設定 ---
import logging
import sys

# 既にハンドラがある場合はクリア
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"🚀 【本番SFT】を開始します（450 steps / 朝4時終了予定）...")
logger.info(f"📁 出力先: {OUTPUT_DIR}")
logger.info(f"📝 ログファイル: {LOG_FILE}")

# 1. モデルとトークナイザーのロード
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map={"": "mps"}
)

# 2. データの準備
def load_and_format_dataset(path):
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        exit(1)
    ds = load_dataset("json", data_files=path, split="train")
    
    def clean_and_format(examples):
        texts = []
        for messages in examples["messages"]:
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(formatted_text)
        return {"text": texts}
    
    return ds.map(clean_and_format, batched=True)

train_dataset = load_and_format_dataset(TRAIN_DATA_PATH)
val_dataset = load_and_format_dataset(VAL_DATA_PATH)

# 3. LoRA設定
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. 本番学習設定（朝4時フィニッシュ・プラン 修正版）
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_steps=300,               # 95s/itに基づき、300回で朝4時前に終わるよう調整
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=5,
    
    eval_strategy="steps",
    eval_steps=50,
    
    save_strategy="steps",
    save_steps=100,              # 100, 200, 300 で保存
    fp16=True,
    report_to="none",
    dataset_text_field="text",
    max_length=1024,
)

# 5. Trainerの実行
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
)

logger.info("🔥 徹夜学習キックオフ！ 0.8突破へ向けて...")
trainer.train()

# 6. 保存
logger.info(f"📦 最終アダプタを保存中: {FINAL_ADAPTER_DIR}")
trainer.model.save_pretrained(FINAL_ADAPTER_DIR)
tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

logger.info("✅ 全工程完了！お疲れ様でした。最高の朝を。")

