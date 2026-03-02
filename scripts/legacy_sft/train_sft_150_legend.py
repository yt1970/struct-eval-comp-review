import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from datetime import datetime
import pytz
import logging
import sys

# --- ゆーちゃ仕様設定 (SFT v2 - 伝説再現版) ---
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
TRAIN_DATA_PATH = "data/train_data/train_sft_v2.jsonl"

# タイムスタンプ
def get_timestamp_jst():
    jst = pytz.timezone('Asia/Tokyo')
    return datetime.now(jst).strftime("%Y%m%d_%H%M")

TIMESTAMP = get_timestamp_jst()
OUTPUT_DIR = f"./outputs/train_sft_legend_{TIMESTAMP}" 
FINAL_ADAPTER_DIR = f"./adapters/adapter_sft_legend_{TIMESTAMP}"
LOG_FILE = f"{OUTPUT_DIR}/training.log"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./adapters", exist_ok=True)

# --- ロギング設定 ---
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

logger.info(f"🚀 【伝説再現版 SFT】を開始します！プロンプト形式を ### 指示 に合わせました！")

# 1. モデルとトークナイザーのロード
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map={"": "mps"}
)

# 2. データの準備（伝説のプロンプト形式）
def load_and_format_dataset(path):
    ds = load_dataset("json", data_files=path, split="train")
    
    def format_legend(sample):
        # messages から instruction と output を抽出
        messages = sample["messages"]
        instruction = ""
        output = ""
        for m in messages:
            if m["role"] == "user":
                instruction = m["content"]
            elif m["role"] == "assistant":
                output = m["content"]
        
        # 伝説のフォーマットを再現
        text = f"### 指示\n{instruction}\n\n### 応答\n{output}"
        return {"text": text}
    
    return ds.map(format_legend, remove_columns=ds.column_names)

train_dataset = load_and_format_dataset(TRAIN_DATA_PATH)
logger.info(f"Train: {len(train_dataset)} samples")

# 3. LoRA設定（伝説通り q_proj, v_proj のみ）
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. 学習設定 (伝説に基づき150ステップ・評価なし)
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_steps=150,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=15,
    logging_steps=10,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=30,  # より早くテストできるよう30に変更
    report_to="none",
    dataset_text_field="text",
    max_length=1024,
)

# 5. Trainerの実行
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
)

logger.info("🔥 リベンジ・マッチ開始！絶対に 0.77 を超えましょう！")
trainer.train()

# 6. 保存
logger.info(f"📦 伝説のアダプタを保存中: {FINAL_ADAPTER_DIR}")
trainer.model.save_pretrained(FINAL_ADAPTER_DIR)
tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

logger.info("✅ コンプリート！伝説が蘇りました！")
