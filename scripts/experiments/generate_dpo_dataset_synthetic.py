import json
import random
from typing import List, Dict

INPUT_FILE = "data/hf_datasets/structured-hard-sft-4k.jsonl"
OUTPUT_FILE = "data/dpo_synthetic_dataset.jsonl"
SKIP_LINES = 100 # SFTで使用した行数（これはスキップする）

# --- 設定 ---
NOISE_TEMPLATES = [
    # パターンA: 標準的な会話形式 + Markdown
    "Sure! Here is the JSON:\n```json\n{json_content}\n```",
    "Here is the requested JSON format:\n```json\n{json_content}\n```",
    "Certainly! Below is the JSON representation based on your description:\n```json\n{json_content}\n```",
    
    # パターンB: Markdown + 末尾の注釈
    "```json\n{json_content}\n```\nNote: This JSON structure follows all the constraints you provided.",
    "```json\n{json_content}\n```\n\nI hope this helps!",
    "```json\n{json_content}\n```\n✅ Validation passed.",
    
    # パターンC: 会話形式のみ
    "Here is the data you asked for:\n{json_content}",
    "This is the JSON output:\n{json_content}",
    
    # パターンD: Markdownのみ (非常によくあるケース)
    "```json\n{json_content}\n```"
]

def load_data(filepath, skip_lines=0):
    data = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError:
                print(f"行 {i} をスキップ: 無効なJSONです")
    return data

def extract_prompt_response(item):
    """
    'messages' リストからユーザーの指示（プロンプト）とアシスタントの応答を抽出します。
    パースに成功した場合は (prompt, response) のタプルを、失敗した場合は None を返します。
    """
    messages = item.get("messages", [])
    prompt = None
    response = None
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        
        if role == "user":
            prompt = content
        elif role == "assistant":
            response = content
            
    if prompt and response:
        return prompt, response
    return None

def create_rejected(json_content):
    """
    ランダムにノイズテンプレートを選択し、純粋なJSONコンテンツに適用してRejectedデータを作成します。
    """
    template = random.choice(NOISE_TEMPLATES)
    rejected_content = template.format(json_content=json_content)
    return rejected_content

def main():
    print(f"{INPUT_FILE} からデータを読み込んでいます（最初の {SKIP_LINES} 行はスキップ）...")
    raw_data = load_data(INPUT_FILE, skip_lines=SKIP_LINES)
    
    dpo_dataset = []
    
    print(f"{len(raw_data)} 件のデータを処理中...")
    
    for item in raw_data:
        extracted = extract_prompt_response(item)
        if not extracted:
            continue
            
        prompt, pure_json = extracted
        
        # 'pure_json' が Chosen（正解：綺麗なJSON）となります
        chosen = pure_json
        
        # 合成した Rejected（不正解：ノイズ付きJSON）を作成
        rejected = create_rejected(pure_json)
        
        dpo_entry = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
        dpo_dataset.append(dpo_entry)
        
    print(f"{len(dpo_dataset)} 件のDPOペアを生成しました。")
    
    # JSONL形式で保存
    print(f"{OUTPUT_FILE} に保存しています...")
    with open(OUTPUT_FILE, 'w') as f:
        for entry in dpo_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print("完了しました！ ✅")
    
    # 最初の3件を表示して確認
    print("\n--- 生成例（最初の3件） ---")
    for i in range(min(3, len(dpo_dataset))):
        print(f"\n[例 {i+1}]")
        print(f"PROMPT (先頭50文字): {dpo_dataset[i]['prompt'][:50]}...")
        print(f"CHOSEN (先頭50文字): {dpo_dataset[i]['chosen'][:50]}...")
        print(f"REJECTED (先頭50文字): {dpo_dataset[i]['rejected'][:50]}...")

if __name__ == "__main__":
    main()
