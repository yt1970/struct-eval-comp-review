import json
import re
import os
import random

def clean_assistant_content(content):
    # "Output:" 以降を抽出。もしなければ全体を返す。
    # 複数のパターン（Output:\n, Output: , # Output など）に対応
    patterns = [
        r'Output:\n([\s\S]*)',
        r'Output: ([\s\S]*)',
        r'### Output\n([\s\S]*)'
    ]
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            code = match.group(1).strip()
            # Markdownのコードブロック記号 (```json ... ```) があれば中身だけ抜き取る
            code = re.sub(r'^```[a-zA-Z]*\n', '', code)
            code = re.sub(r'\n```$', '', code)
            return code.strip()
    
    # パターンにマッチしない場合は、Markdownのコードブロックがあればそれを使う
    code_match = re.search(r'```[a-zA-Z]*\n([\s\S]*)\n```', content)
    if code_match:
        return code_match.group(1).strip()
        
    return content.strip()

def process_file(input_path):
    items = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
            except:
                continue
            messages = data.get('messages', [])
            if len(messages) < 2:
                continue
            
            user_content = ""
            assistant_content = ""
            
            for m in messages:
                if m['role'] == 'user':
                    user_content = m['content']
                elif m['role'] == 'assistant' or m['role'] == 'assistant':
                    assistant_content = m['content']
            
            if user_content and assistant_content:
                clean_code = clean_assistant_content(assistant_content)
                # もし何らかの理由で空になったらスキップ
                if not clean_code:
                    continue
                    
                items.append({
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": clean_code}
                    ]
                })
    return items

def main():
    datasets = [
        "data/hf_datasets/sft_1-1.jsonl",
        "data/hf_datasets/structured-5k-mix-sft.jsonl",
        "data/hf_datasets/structured-hard-sft-4k.jsonl"
    ]
    
    all_items = []
    for ds in datasets:
        print(f"📦 Processing {ds}...")
        all_items.extend(process_file(ds))
    
    print(f"✅ Total items collected: {len(all_items)}")
    
    # シャッフルして分割
    random.seed(42)
    random.shuffle(all_items)
    
    train_split = int(len(all_items) * 0.95)
    train_items = all_items[:train_split]
    valid_items = all_items[train_split:]
    
    os.makedirs("data/mlx_v10", exist_ok=True)
    
    with open("data/mlx_v10/train.jsonl", "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open("data/mlx_v10/valid.jsonl", "w", encoding="utf-8") as f:
        for item in valid_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"🚀 Data ready in data/mlx_v10/ (Train: {len(train_items)}, Valid: {len(valid_items)})")

if __name__ == "__main__":
    main()
