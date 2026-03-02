import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 設定 ---
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
# 学習が終わったら最新のアダプタパスに書き換えてください
ADAPTER_PATH = "./adapters/adapter_sft_legend_20260207_1929" 
EVAL_DATA_PATH = "./data/public_150.json"

# テストしたいタスクID (間違いやすかった難問をピックアップ)
TARGET_TASK_IDS = [
    "p_2204d42637c2e3df784a57a3", # Text to TOML
    "p_b153305f9a1ebe575cde3261", # JSON to XML
    "p_ced5b041360b01bd91f38fd2", # CSV to JSON (Complex)
]

def main():
    if not ADAPTER_PATH:
        print("❌ ADAPTER_PATH を指定してください！ (例: outputs/train_sft_legend_YYYYMMDD_HHMM/checkpoint-30)")
        return

    print(f"📥 評価データ読み込み中...")
    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # ターゲットのみ抽出
    samples = [item for item in eval_data if item["task_id"] in TARGET_TASK_IDS]

    print(f"🔄 ベースモデル読み込み中: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": "mps"}
    )

    print(f"🔗 伝説アダプタ適用中: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    print("\n🚀 伝説形式プロンプトで推論開始！")
    for item in samples:
        query = item.get("query", "")
        task_name = item.get("task_name", "Unknown")
        
        # 伝説のプロンプト形式を再現！
        prompt = f"### 指示\n{query}\n\n### 応答\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("mps")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1, # 構造化データなので低めに
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        print(f"\n{'='*60}")
        print(f"🏷️  Task: {task_name} (ID: {item['task_id']})")
        print(f"🤖 Response:")
        print("-" * 40)
        print(response.strip())
        print("-" * 40)

if __name__ == "__main__":
    main()
