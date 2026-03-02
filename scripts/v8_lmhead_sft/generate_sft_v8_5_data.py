
import json
import re
import os
import random

# === SFT v8.5: ハルシネーション防止 & 沈黙強化 データ生成 ===

SOURCE_FILE = "data/hf_datasets/structured-hard-sft-4k.jsonl"
OUTPUT_FILE = "data/sft_v8_5_silent.jsonl"
TARGET_SIZE = 1000

def anonymize_and_clean(text):
    """
    1. Markdown/おしゃべり除去
    2. 固有名詞らしきものをプレースホルダに置換 (ハルシネーション防止)
    """
    # --- 1. Markdown/おしゃべり除去 (v8.4 方針) ---
    text = re.sub(r'```\w*\n?', '', text)
    text = text.replace("```", "")
    text = re.sub(r"Approach:.*?Output:\n", "", text, flags=re.DOTALL)
    
    chatty_patterns = [
        r"^Sure!.*?:\s*\n?", r"^Here is the.*?:\s*\n?", r"^Certainly!.*?:\s*\n?",
        r"^I can help.*?:\s*\n?", r"^Below is the.*?:\s*\n?", r"^The (?:following|converted|resulting).*?:\s*\n?",
    ]
    for p in chatty_patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)
        
    tail_patterns = [
        r"\n\s*Note:.*$", r"\n\s*I hope this helps.*$", r"\n\s*Let me know.*$", r"\n\s*Feel free.*$",
    ]
    for p in tail_patterns:
        text = re.sub(p, "", text, flags=re.DOTALL)

    # --- 2. 固有名詞置換 (ハルシネーション防止) ---
    # 単純な置換: 引用符内の特定文字列などを [ENTITY_N] に変える
    # (本当はもっと賢くやるべきだが、モデルに教えたいのは『型』なのでこれで十分)
    def replace_entities(t):
        entities = []
        def repl(m):
            val = m.group(1)
            if len(val) > 3 and not val.isdigit():
                if val not in entities: entities.append(val)
                return f'"{[f"VAL_{entities.index(val)}"]}"' # 簡易的な置換
            return f'"{val}"'
        
        # JSON/YAML/XMLの値らしき部分を置換
        t = re.sub(r'"([^"]+)"', repl, t)
        t = re.sub(r'>([^<]{4,})<', lambda m: f'>PLACEHOLDER_{len(m.group(1))}<', t)
        return t

    # 完全に置換しすぎると構文が壊れるので、今回は『沈黙』を最優先し、
    # 固有名詞による知識汚染を避けるため、Chosen は元データから『おしゃべり』を抜いたものを使用。
    # ただし、特定の固有名詞 ("Luminara"等) が含まれている場合は置換を試みる。
    hallucination_triggers = ["Luminara", "Verdant", "Aetheria", "Aethelgard", "Zyphora", "Xerophos"]
    for word in hallucination_triggers:
        text = text.replace(word, "REDACTED_ENTITY")
        text = text.replace(word.lower(), "redacted_entity")

    return text.strip()

def main():
    print(f"🎬 SFT v8.5 データ生成中... 目標: {TARGET_SIZE}件")
    
    with open(SOURCE_FILE, 'r') as f:
        all_lines = f.readlines()
    
    random.seed(85)
    random.shuffle(all_lines)
    
    data = []
    for line in all_lines:
        if len(data) >= TARGET_SIZE: break
        
        item = json.loads(line)
        user_msg = next((m['content'] for m in item['messages'] if m['role'] == 'user'), None)
        asst_msg = next((m['content'] for m in item['messages'] if m['role'] == 'assistant'), None)
        
        if not user_msg or not asst_msg: continue
        
        # v8.5 方針：形式はクリーン、かつ特定単語への依存を排除
        chosen = anonymize_and_clean(asst_msg)
        if len(chosen) < 30: continue
        
        data.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": chosen}
            ]
        })

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, 'w') as out:
        for item in data:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ 生成完了: {len(data)} 件 -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
