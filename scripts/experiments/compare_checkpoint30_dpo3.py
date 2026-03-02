import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 設定 ---
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "./outputs/train_sft_v2_light_20260207_1550/checkpoint-30"
EVAL_DATA_PATH = "./data/public_150.json"
DPO3_PATH = "./inference_0_DPO3(0.77064).json"

# 間違いが多そうなタスク（難しいやつ）
TARGET_TASKS = ["Text to TOML", "JSON to XML", "YAML to XML", "XML to CSV"]
SAMPLE_PER_TASK = 1

print(f"📥 データ読み込み中...")
with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

with open(DPO3_PATH, "r", encoding="utf-8") as f:
    dpo3_data = json.load(f)

# DPO3の結果をtask_idでマッピング
dpo3_map = {item["task_id"]: item["generation"] for item in dpo3_data}

# ターゲットタスクのサンプルを取得
samples = []
for task_name in TARGET_TASKS:
    for item in eval_data:
        if item.get("task_name") == task_name:
            samples.append(item)
            break  # 1件だけ

print(f"📊 {len(samples)}件のサンプルをテスト")

# モデルロード
print(f"🔄 ベースモデル読み込み中: {BASE_MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map={"": "mps"}
)

# アダプタ適用
print(f"🔗 アダプタ適用中: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

print("✅ モデル準備完了！\n")

for i, item in enumerate(samples):
    query = item.get("query", "")
    task_id = item.get("task_id", f"sample_{i}")
    task_name = item.get("task_name", "Unknown")
    
    print(f"\n{'='*70}")
    print(f"📝 Sample {i+1}/{len(samples)}")
    print(f"🏷️  Task: {task_name}")
    print(f"🆔 Task ID: {task_id}")
    print(f"📋 Query (先頭200文字):")
    print(query[:200] + "..." if len(query) > 200 else query)
    
    # プロンプト作成
    messages = [{"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    
    # 推論
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # DPO3の結果
    dpo3_response = dpo3_map.get(task_id, "N/A")
    
    print(f"\n🔴 DPO3 (0.77スコア) の出力 (先頭300文字):")
    print("-" * 50)
    print(dpo3_response[:300] if len(dpo3_response) > 300 else dpo3_response)
    print("-" * 50)
    
    print(f"\n🟢 Checkpoint-30 の出力 (先頭300文字):")
    print("-" * 50)
    print(response[:300] if len(response) > 300 else response)
    print("-" * 50)

print(f"\n✅ 比較テスト完了！")
