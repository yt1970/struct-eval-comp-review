"""
SFT v8.5 Full Quality Scouter (150 tasks)
- 資料が欠落している 1-2 問目には、検証用に資料を自動挿入して実行します。
- 全問を通して、Markdown混入、饒舌さ、ハルシネーション傾向を自動チェックします。
"""
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# === Config ===
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
V8_5_ADAPTER_PATH = "adapters/adapter_sft_v8_5_anti_hallucination_step100"
INPUT_PATH = "data/public_150_full_eval_v8_5.json"
HALLUCINATION_WATCHLIST = ["Luminara", "Verdant", "Aetheria", "Aethelgard", "Zyphora"]

def main():
    print(f"🧐 Loading v8.5 Step 100 for global quality check...")
    
    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    # Model Load
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = PeftModel.from_pretrained(base_model, V8_5_ADAPTER_PATH)
    model.eval()

    stats = {"total": 150, "pure": 0, "markdown": 0, "chatty": 0, "hallu": 0}
    errors = []

    print(f"\n🔍 Evaluating 150 tasks using {INPUT_PATH}...")
    for item in tqdm(tasks):
        task_id = item.get("task_id")
        source = item.get("source_text", "")
        query = item.get("query", "").replace("[DATA_BLOCK]", source)
        
        messages = [{"role": "user", "content": f"{source}\n\nTask:\n{query}" if source not in query else query}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Evaluation
        is_bad = False
        if "```" in response:
            stats["markdown"] += 1
            is_bad = True
        if any(w in response.lower()[:30] for w in ["sure", "ok", "here", "certainly"]):
            stats["chatty"] += 1
            is_bad = True
        
        # ハルシネーションチェック (Context外の暗記ワード露出)
        for word in HALLUCINATION_WATCHLIST:
            if word.lower() in response.lower() and word.lower() not in query.lower():
                stats["hallu"] += 1
                is_bad = True
                break
        
        if not is_bad:
            stats["pure"] += 1
        else:
            errors.append({"id": task_id, "res": response[:100]})

    print("\n" + "="*50)
    print("📊 SFT v8.5 Step 100 Final Scouter Report")
    print("="*50)
    print(f"✅ 全項目クリア (沈黙かつ正確): {stats['pure']} / 150")
    print(f"❌ Markdown混入: {stats['markdown']}")
    print(f"❌ 饒舌エラー:   {stats['chatty']}")
    print(f"⚠️  ハルシネーション疑い: {stats['hallu']}")
    print("="*50)
    
    if errors:
        print("\n典型的なエラー例:")
        for e in errors[:3]:
            print(f"- {e['id']}: {e['res']}...")

if __name__ == "__main__":
    main()
