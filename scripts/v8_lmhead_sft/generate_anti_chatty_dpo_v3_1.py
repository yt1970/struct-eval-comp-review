import json
import random
import os
from transformers import AutoTokenizer

# --- CONFIG ---
VERSION = "V3_1_Deep_Silence"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SOURCE_DATA = "data/hf_datasets/structured-hard-sft-4k.jsonl" 
OUTPUT_FILE = f"data/dpo_anti_chatty_{VERSION}.jsonl"

def get_real_failure_pattern(task_type, output_text):
    """
    検証ログで観測された「お喋り」の全バリエーションを網羅。
    Markdown、挨拶、末尾の解説。
    """
    patterns = [
        # パターン1: 挨拶 + Markdown
        f"Sure! Here is the {task_type} data converted to JSON format:\n\n```json\n{output_text}\n```",
        # パターン2: 丁寧な説明 + Markdown + 解説
        f"I have converted the provided {task_type} into a JSON object as requested. All fields have been preserved.\n\n```json\n{output_text}\n```\n\nThis JSON preserves the structure of your original data.",
        # パターン3: いきなりMarkdown (これもアウト)
        f"```json\n{output_text}\n```",
        # パターン4: シンプルな前置き
        f"Here is the converted JSON:\n\n{output_text}"
    ]
    return random.choice(patterns)

def main():
    print(f"🚀 Generating CONCENTRATED Anti-Chatty DPO dataset ({VERSION})...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    with open(SOURCE_DATA, "r", encoding="utf-8") as f:
        source_lines = [json.loads(line) for line in f]
    
    dpo_items = []
    
    # 【最重要】特にお喋りになりやすい「変換系(csv, convert, transform)」だけに絞る
    target_tasks = []
    for item in source_lines:
        msg = str(item).lower()
        if any(kw in msg for kw in ["csv", "convert ", "transform", "format", "json"]):
            target_tasks.append(item)
    
    print(f"   Original: {len(source_lines)} -> Filtered: {len(target_tasks)}")
    
    # 質の高い 500 件に濃縮（多すぎると薄まるため）
    random.shuffle(target_tasks)
    target_tasks = target_tasks[:500]

    for item in target_tasks: 
        if "messages" in item:
            query = item["messages"][0]["content"]
            chosen_answer = item["messages"][1]["content"]
        else:
            query = item.get("instruction", item.get("query", ""))
            chosen_answer = item.get("output", "")

        if not query or not chosen_answer:
            continue

        # ChatML 形式適用
        # プロンプトは「素（シンプル）」なままにする
        # これにより「普通のプロンプトで沈黙する重み」を学習させる
        messages = [{"role": "user", "content": query}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Rejected: お喋りフレーズをたっぷり盛り込む
        task_type = "CSV" if "csv" in query.lower() else "Data"
        rejected_answer = get_real_failure_pattern(task_type, chosen_answer)
        
        # Chosen: 究極の「生データ」。挨拶もMarkdownも一切なし
        # 余計な空白や改行を削り、純粋な JSON 構造から始めさせる
        clean_chosen = chosen_answer.strip()
        
        # EOSを付けて「ここで終われ」と教え込む
        chosen_with_eos = clean_chosen + tokenizer.eos_token
        
        dpo_items.append({
            "prompt": prompt,
            "chosen": chosen_with_eos,
            "rejected": rejected_answer
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in dpo_items:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(dpo_items)} DENSE DPO pairs -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
