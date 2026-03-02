import json
import re
import os
import random

# ==========================================
# V11 Surgical DPO Data Generator
# ==========================================
# 目的: 0.77モデル（DPO3）のお喋り癖を完璧に外科手術する
# ルール遵守: public_150.json は不使用。hf_datasets のみを使用。
# ==========================================

VERSION = "V11_Surgical"
SOURCE_DATA = "data/hf_datasets/structured-hard-sft-4k.jsonl"
OUTPUT_FILE = "data/dpo_v11_surgical.jsonl"

def clean_structural_content(content):
    """
    あらゆるお喋り、Markdown、前置き、後書きを剥ぎ取って「生データ」にする。
    """
    # 1. 典型的な前置きを削る (Case-insensitive)
    content = re.sub(r'^(Sure!|Certainly|Here is|Below is|I have|As requested)[^\n]*\n+', '', content, flags=re.IGNORECASE)
    
    # 2. Markdownコードブロック (```json ... ```) を剥ぎ取る
    content = re.sub(r'```[a-zA-Z]*\n', '', content)
    content = re.sub(r'\n```$', '', content)
    content = re.sub(r'^```', '', content)
    content = re.sub(r'```$', '', content)
    
    return content.strip()

def create_rejected_pattern(content):
    """
    0.77モデルがやりがちな「お節介」を人工的に合成する。
    """
    structural = clean_structural_content(content)
    
    patterns = [
        # パターン1: 挨拶 + Markdown
        f"Sure! Here is the converted data in JSON format:\n\n```json\n{structural}\n```",
        # パターン2: 丁寧な説明 + 生データ
        f"I have processed the request and formatted the output as requested.\n\n{structural}",
        # パターン3: いきなり Markdown
        f"```json\n{structural}\n```",
        # パターン4: シンプルな前置き
        f"Here is the result:\n\n{structural}",
        # パターン5: 解説付き Markdown
        f"Based on the input provided, here is the structured output.\n\n```\n{structural}\n```\n\nThis format follows all your constraints."
    ]
    return random.choice(patterns)

def main():
    print(f"🚀 V11 Surgical DPO データ生成開始: {VERSION}")
    
    if not os.path.exists(SOURCE_DATA):
        print(f"❌ Source data not found: {SOURCE_DATA}")
        return

    with open(SOURCE_DATA, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    print(f"   読み込み完了: {len(lines)} 件")

    dpo_items = []
    
    # 特にお喋りが発生しやすい変換・抽出タスクを優先的に抽出
    random.shuffle(lines)
    
    target_count = 1000 # 外科手術には1000件程度が適正（多すぎると知能が枯れる）
    
    for item in lines:
        if len(dpo_items) >= target_count:
            break
            
        messages = item.get("messages", [])
        if len(messages) < 2:
            continue
            
        prompt_content = messages[0]["content"]
        assistant_content = messages[1]["content"]
        
        # Chosen: 徹底的にクレンジングされた沈黙のデータ
        chosen = clean_structural_content(assistant_content)
        if not chosen:
            continue
            
        # Rejected: 汚染されたお喋り回答
        rejected = create_rejected_pattern(assistant_content)
        
        # DPOペアに追加
        dpo_items.append({
            "prompt": prompt_content,
            "chosen": chosen,
            "rejected": rejected
        })

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in dpo_items:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ 生成完了: {len(dpo_items)} 件 -> {OUTPUT_FILE}")
    print(f"   ※ChosenからMarkdownが消え、Rejectedにお喋りが注入されました。")

if __name__ == "__main__":
    main()
