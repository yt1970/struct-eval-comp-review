
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

# --- ゆーちゃ仕様設定 (SFT v2) ---
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
TRAIN_DATA_PATH = "data/train_data/train_sft_v2.jsonl"
VAL_DATA_PATH = "data/train_data/valid_sft_v2.jsonl"

# タイムスタンプ
def get_timestamp_jst():
    jst = pytz.timezone('Asia/Tokyo')
    return datetime.now(jst).strftime("%Y%m%d_%H%M")

TIMESTAMP = get_timestamp_jst()
OUTPUT_DIR = f"./outputs/train_sft_v2_{TIMESTAMP}" 
FINAL_ADAPTER_DIR = f"./adapters/adapter_sft_v2_{TIMESTAMP}"
LOG_FILE = f"{OUTPUT_DIR}/training.log"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./adapters", exist_ok=True)

# --- ロギング設定 ---
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

logger.info(f"🚀 【SFT v2: 爆速150ステップ】を開始します！")
logger.info(f"📁 データ: {TRAIN_DATA_PATH}")
logger.info(f"📝 ログ: {LOG_FILE}")

# 1. モデルとトークナイザーのロード
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map={"": "mps"} if torch.backends.mps.is_available() else "auto"
)

# 2. データの準備
def load_and_format_dataset(path):
    if not os.path.exists(path):
        logger.error(f"File not found: {path} - 以前のステップで生成されましたか？")
        exit(1)
        
    ds = load_dataset("json", data_files=path, split="train")
    
    def clean_and_format(examples):
        texts = []
        for messages in examples["messages"]:
            # apply_chat_templateですでに整形済みテキストを作る
            # 学習時は add_generation_prompt=False でOK（EOSまで学習させるため）
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(formatted_text)
        return {"text": texts}
    
    return ds.map(clean_and_format, batched=True)

train_dataset = load_and_format_dataset(TRAIN_DATA_PATH)
val_dataset = load_and_format_dataset(VAL_DATA_PATH)

logger.info(f"Train: {len(train_dataset)} samples / Valid: {len(val_dataset)} samples")

# 3. LoRA設定
peft_config = LoraConfig(
    r=16, # 少し軽量化
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. 学習設定 (150ステップ)
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_steps=150,               # 激短！
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, # 実質バッチサイズ16
    learning_rate=1e-4,          # 安定重視
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=5,
    
    eval_strategy="steps",
    eval_steps=20,               # 頻繁にチェック
    
    save_strategy="steps",
    save_steps=20,               # 頻繁に保存（ベストモデルを逃さない）
    
    fp16=False,                  # Mac(MPS)ではbf16推奨だが、安全策でfp16無効化or自動
    bf16=False,                  # MPSではTrueにすると動く場合もあるが、一旦安全策
    
    report_to="none",
    dataset_text_field="text",
    max_length=2048,             # 十分な長さを確保
)

# 5. Trainerの実行
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
)

logger.info("🔥 学習スタート！約15分で終わります！")
trainer.train()

# 6. 保存
logger.info(f"📦 最終アダプタを保存中: {FINAL_ADAPTER_DIR}")
trainer.model.save_pretrained(FINAL_ADAPTER_DIR)
tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

logger.info("✅ コンプリート！お疲れ様でした！")
