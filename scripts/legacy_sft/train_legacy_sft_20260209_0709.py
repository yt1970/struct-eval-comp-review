
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# --- 設定 (Legacy Rebirth - User Provided Logic) ---
TIMESTAMP = "20260209_0709"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
TRAIN_DATA_PATH = f"data/train_data_legacy_{TIMESTAMP}/train.jsonl"
VALID_DATA_PATH = f"data/train_data_legacy_{TIMESTAMP}/valid.jsonl"
OUTPUT_DIR = f"outputs/train_legacy_sft_{TIMESTAMP}"
ADAPTER_OUT_DIR = f"adapters/adapter_legacy_sft_{TIMESTAMP}"

MAX_STEPS = 30  
LEARNING_RATE = 2e-4
BATCH_SIZE = 1 # ユーザーコードに合わせる
GRAD_ACCUM = 4

def main():
    print(f"🏛️ 伝説の再現（Legacy SFT）を開始します... ID: {TIMESTAMP}")
    
    # データセットのロード
    # split="train" で読み込む（辞書型でなくDatasetオブジェクトとして扱うため）
    # trainとvalidを混ぜるのではなく、シンプルにtrainのみで学習した可能性が高い（ユーザーコードにvalidの記述がないため）
    # しかし、計画通りvalidも一応ロードはしておくが、Trainerにはtrainのみ渡す形にする（当時の再現重視）
    dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")

    # ユーザー提供のフォーマット関数
    def format_func(sample):
        # input/output キーが存在する場合の対応
        instruction = sample.get('instruction', '') or sample.get('input', '')
        output = sample.get('output', '')
        return {"text": f"### 指示\n{instruction}\n\n### 応答\n{output}"}

    # マッピング適用
    dataset = dataset.map(format_func)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # ベースモデル
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": "mps"}
    )
    
    # LoRA構成
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    
    # 学習設定（ユーザー提供コードベース）
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        dataset_text_field="text", # ここで指定！
        save_strategy="no",
        remove_unused_columns=False,
        bf16=False,
        report_to="none"
    )

    print("🚀 学習スタート！")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()

    print(f"💾 伝説のアダプタを保存中... -> {ADAPTER_OUT_DIR}")
    trainer.model.save_pretrained(ADAPTER_OUT_DIR)
    print("✅ 再現完了！")

if __name__ == "__main__":
    main()
