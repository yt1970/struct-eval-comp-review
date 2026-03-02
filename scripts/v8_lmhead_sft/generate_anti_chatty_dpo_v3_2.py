import json
import random
from transformers import AutoTokenizer

# --- CONFIG ---
VERSION = "V3_2_Pure_JSON"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
# JSONが多いことが確認できたファイルに変更
SOURCE_DATA_FILES = [
    "data/hf_datasets/structured-5k-mix-sft.jsonl",
    "data/hf_datasets/structured-3k-mix-sft.jsonl",
    "data/hf_datasets/sft_1-1.jsonl"
]
OUTPUT_FILE = f"data/dpo_anti_chatty_{VERSION}.jsonl"

def get_chatty_rejected(output_text):
    patterns = [
        f"Sure! Here is the converted JSON data:\n\n```json\n{output_text}\n```",
        f"I have converted the input to JSON format for you.\n\n```json\n{output_text}\n```",
        f"```json\n{output_text}\n```",
        f"Here is the JSON representation:\n\n{output_text}\n\nNote: All values are preserved."
    ]
    return random.choice(patterns)

def is_pure_json(text):
    t = text.strip()
    return (t.startswith("{") or t.startswith("[")) and ("<root>" not in t) and ("<items>" not in t)

def main():
    print(f"🚀 Generating PURE JSON DPO dataset ({VERSION})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    all_json_tasks = []
    
    for file_path in SOURCE_DATA_FILES:
        if not os.path.exists(file_path):
            continue
        print(f"   Reading {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                o = ""
                if "messages" in item and len(item["messages"]) >= 2:
                    o = item["messages"][1]["content"]
                elif "output" in item:
                    o = item["output"]
                
                if is_pure_json(o):
                    all_json_tasks.append(item)
    
    print(f"   Total Pure JSON Tasks found: {len(all_json_tasks)}")
    
    random.seed(42)
    random.shuffle(all_json_tasks)
    target_tasks = all_json_tasks[:200]

    dpo_items = []
    for item in target_tasks: 
        if "messages" in item:
            query = item["messages"][0]["content"]
            answer = item["messages"][1]["content"]
        else:
            query = item.get("instruction", item.get("query", ""))
            answer = item.get("output", "")

        prompt = tokenizer.apply_chat_template([{"role": "user", "content": query}], tokenize=False, add_generation_prompt=True)
        clean_chosen = answer.strip()
        
        dpo_items.append({
            "prompt": prompt,
            "chosen": clean_chosen + tokenizer.eos_token,
            "rejected": get_chatty_rejected(clean_chosen)
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in dpo_items:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(dpo_items)} PURE JSON DPO pairs -> {OUTPUT_FILE}")

import os
if __name__ == "__main__":
    main()
