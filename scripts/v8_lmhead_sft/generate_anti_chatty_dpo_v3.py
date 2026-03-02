import json
import random
import os
from transformers import AutoTokenizer

# --- CONFIG ---
TIMESTAMP = "20260215_1822"
VERSION = "V3_Surgical_Strike"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SOURCE_DATA = "data/hf_datasets/structured-hard-sft-4k.jsonl" 
OUTPUT_FILE = f"data/dpo_anti_chatty_{VERSION}.jsonl"

def get_real_failure_pattern(task_type):
    """
    検証ログ(V2)で実際に観測された「お喋りパターン」を機械的に返す。
    LLMは使用しない。
    """
    # 実際に出力された "Here's the..." シリーズ
    patterns = [
        # Pattern 1: シンプルなMarkdown
        "```json\n",
        
        # Pattern 2: 丁寧な前置き + Markdown
        f"Here's the converted {task_type} data in JSON format:\n\n```json\n",
        
        # Pattern 3: 具体的な指示への応答
        f"Sure! I have converted the provided {task_type} data into a JSON object.\n\n```json\n",
        
        # Pattern 4: 結果の提示
        f"Here is the {task_type} data converted to JSON:\n\n```json\n",
        
        # Pattern 5: 変換しました報告
        f"I have converted the {task_type} code to JSON code as requested.\n\n```json\n"
    ]
    return random.choice(patterns)

def main():
    print(f"🚀 Generating Anti-Chatty DPO dataset ({VERSION})...")
    print("   Method: Rule-based Noise Injection (No LLM used)")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    with open(SOURCE_DATA, "r", encoding="utf-8") as f:
        source_lines = [json.loads(line) for line in f]
    
    dpo_items = []
    
    # 特に CSV/Text 変換タスクを重点的に抽出
    priority_tasks = [item for item in source_lines if "csv" in str(item).lower() or "convert" in str(item).lower()]
    print(f"   Focusing on {len(priority_tasks)} conversion tasks...")

    for item in priority_tasks: 
        if "messages" in item:
            query = item["messages"][0]["content"]
            chosen_answer = item["messages"][1]["content"]
        else:
            query = item.get("instruction", item.get("query", ""))
            chosen_answer = item.get("output", "")

        if not query or not chosen_answer:
            continue

        # ChatML 形式適用
        messages = [{"role": "user", "content": query}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Rejected 生成 (機械的な文字列結合)
        task_type = "CSV" if "csv" in query.lower() else "Text"
        noise_prefix = get_real_failure_pattern(task_type)
        
        # Markdown閉じタグもセットで付与
        rejected_answer = noise_prefix + chosen_answer + "\n```"
        
        # Chosen 生成 (EOS付与)
        chosen_with_eos = chosen_answer + tokenizer.eos_token
        
        dpo_items.append({
            "prompt": prompt,
            "chosen": chosen_with_eos,
            "rejected": rejected_answer
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in dpo_items:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(dpo_items)} DPO pairs -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
