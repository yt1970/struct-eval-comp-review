import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 設定 ---
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "./outputs/train_sft_v2_light_20260207_1550/checkpoint-30"
EVAL_DATA_PATH = "./data/public_150.json"

# サンプル数（最初は5件）
SAMPLE_COUNT = 5

print(f"📥 評価データ読み込み中: {EVAL_DATA_PATH}")
with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

print(f"📊 全{len(eval_data)}件中、{SAMPLE_COUNT}件をテスト")

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

print("✅ モデル準備完了！")

for i, item in enumerate(eval_data[:SAMPLE_COUNT]):
    query = item.get("query", "")
    task_id = item.get("task_id", f"sample_{i}")
    task_name = item.get("task_name", "Unknown")
    
    print(f"\n{'='*60}")
    print(f"📝 Sample {i+1}/{SAMPLE_COUNT}")
    print(f"🏷️  Task: {task_name}")
    print(f"📋 Query (先頭150文字):")
    print(query[:150] + "..." if len(query) > 150 else query)
    
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
    
    print(f"\n🤖 Response:")
    print("-" * 40)
    print(response[:500] if len(response) > 500 else response)
    print("-" * 40)

print(f"\n✅ テスト完了！")
