import json
import random
import os
from transformers import AutoTokenizer

# --- CONFIG ---
TIMESTAMP = "20260215_1822"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SOURCE_DATA = "data/hf_datasets/structured-hard-sft-4k.jsonl" # 運営配布 2-3
OUTPUT_FILE = f"data/dpo_anti_chatty_{TIMESTAMP}.jsonl"

def generate_noise(output_text, task_type="JSON"):
    """
    検証で見られたモデルの失敗パターンを機械的に再現する
    """
    patterns = [
        # パターンA: Markdown混入
        f"```json\n{output_text}\n```",
        # パターンB: 饒舌な前置き
        f"Here is the conversion result in {task_type} format:\n\n{output_text}",
        # パターンC: ハイブリッド (最も多い失敗)
        f"Sure! I have converted the data into {task_type} according to your requirements.\n\n```json\n{output_text}\n```",
        # パターンD: 注釈付き
        f"{output_text}\n\nNote: All fields have been preserved."
    ]
    return random.choice(patterns)

def main():
    print(f"🚀 Generating Anti-Chatty DPO dataset ({TIMESTAMP})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    with open(SOURCE_DATA, "r", encoding="utf-8") as f:
        source_lines = [json.loads(line) for line in f]
    
    # 指示形式によるサンプリング (失敗が多かった CSV/Text 変換系を重点的に)
    dpo_items = []
    
    # 運営配布データ 2-3 (structured-hard-sft-4k) は messages リスト形式ではない場合があるため調整
    # (実際の形式を確認しながら)
    for i, item in enumerate(source_lines[:500]): # ひとまず先頭500件からサンプリング
        # データ形式の統一 (Qwen形式の messages を想定、なければ構築)
        if "messages" in item:
            query = item["messages"][0]["content"]
            chosen_answer = item["messages"][1]["content"]
        else:
            query = item.get("instruction", item.get("query", ""))
            chosen_answer = item.get("output", "")

        if not query or not chosen_answer:
            continue

        # 本番環境と 1 文字もズレないように ChatML 形式を適用
        messages = [{"role": "user", "content": query}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Rejected の生成 (ノイズ注入)
        task_type = "JSON" if "json" in query.lower() else ("CSV" if "csv" in query.lower() else "XML")
        rejected_answer = generate_noise(chosen_answer, task_type)
        
        # EOS 制御: Chosen には EOS を付与し、Rejected には付与しない
        # (tokenizer.eos_token を明示的に追加)
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
