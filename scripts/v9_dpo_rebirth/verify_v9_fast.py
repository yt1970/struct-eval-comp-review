import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "adapters/adapter_dpo_V9_Rebirth"
DATA_PATH = "data/public_150_full_eval_v8_5.json"

def main():
    print("🚀 V9 (DPO Rebirth) Fast Debug Test (Top 3 Tasks)")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Dataset {DATA_PATH} not found.")
        return

    with open(DATA_PATH, 'r') as f:
        tasks = json.load(f)
    print(f"📁 Loaded dataset with {len(tasks)} tasks.")
    # 先頭の3件ではなく、なるべく複雑なタスク(System/Markdown要求など)をサンプリング
    # 例として インデックス50〜53周辺などの3件を抜粋
    tasks = tasks[50:53]
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    print("📦 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"💉 Applying V9 DPO adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("✅ Model ready\n")
    
    for i, task in enumerate(tasks):
        source = task.get("source_text", "")
        query_raw = task.get("query", "")
        
        # テンプレートに合わせたプロンプト構築
        query = query_raw.replace("[DATA_BLOCK]", source)
        prompt_content = f"{source}\n\nTask:\n{query}" if source and source not in query else query

        messages = [
            {"role": "user", "content": prompt_content}
        ]
        
        # 推論時と同じ apply_chat_template フォーマットを使用
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        print("=" * 80)
        print(f"🔹 Task {i+1} : {task.get('task_name', 'Unknown')} 🔹")
        print("【Input Text (Snippet)】")
        # 入力が長い場合は先頭のみ表示
        print(prompt_content[:300] + ("..." if len(prompt_content) > 300 else ""))
        print("-" * 80)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.01, # 決定論的に
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 不要なプロンプト部分をカットして生成部分のみデコード
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        full_tokens = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        print("【Response Preview】")
        print(response)
        
        has_md = "```" in response
        has_chat = any(x in response.lower()[:50] for x in ["sure", "here", "certainly", "approach", "output", "ok", "はい、"])
        has_eos = tokenizer.eos_token in full_tokens or "<|im_end|>" in full_tokens
        
        print("\n--- Validation Checks ---")
        print("  Markdown: " + ("❌ FOUND (bad)" if has_md else "✅ CLEAN (good)"))
        print("  Chatty:   " + ("❌ FOUND (bad)" if has_chat else "✅ CLEAN (good)"))
        print("  EOS stop: " + ("✅ YES (good)" if has_eos else "❌ NO (bad)"))
        print("\n")

if __name__ == "__main__":
    main()
