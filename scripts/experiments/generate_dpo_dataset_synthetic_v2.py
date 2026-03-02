import json
import random
from typing import List, Dict

INPUT_FILE = "data/hf_datasets/structured-hard-sft-4k.jsonl"
OUTPUT_FILE = "data/dpo_synthetic_dataset_v2.jsonl"
SKIP_LINES = 100 # SFTで使用した行数（これはスキップする）

# --- EOSトークン設定 ---
EOS_TOKEN = "<|im_end|>" # QwenのEOS

# --- 設定 (Rejectedパターン強化: Markdown残存や過剰な丁寧さへの対策) ---
NOISE_TEMPLATES = [
    # パターンA: Markdownのみ (30%) - これが残存しやすいので確率を上げる
    "```json\n{json_content}\n```",
    "```json\n{json_content}\n```",
    "```json\n{json_content}\n```",
    
    # パターンB: 標準的な会話形式 + Markdown (20%)
    "Sure! Here is the JSON:\n```json\n{json_content}\n```",
    "Certainly! Below is the JSON representation based on your description:\n```json\n{json_content}\n```",
    
    # パターンC: Markdown + 末尾の注釈 (20%) - 蛇足解説への対策
    "```json\n{json_content}\n```\nNote: This JSON structure follows all the constraints you provided.",
    "```json\n{json_content}\n```\n✅ Validation passed.",
    
    # パターンD: 会話形式のみ (20%)
    "Here is the data you asked for:\n{json_content}",
    "This is the JSON output:\n{json_content}",
    
    # パターンE: Markdownなし、蛇足解説のみ (10%)
    "{json_content}\nNote: I generated this based on the query.",
    "{json_content}\nHopefully this is correct."
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
    Markdownパターンを強化しています。
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
        
        # Chosen（正解）: 純粋なJSON + EOSトークン
        # EOSを明示的に付与することで、「ここまでで終わり」と学習させる
        # ※ Trainer側で別途EOSが付与される場合、重複する可能性もあるが、
        # Qwenのチャットテンプレートによっては EOS が body に含まれないこともあるので、
        # 文字列として結合しておくのが安全策（"Silence" を徹底するため）。
        chosen = pure_json + EOS_TOKEN
        
        # Rejected（不正解）: ノイズ付きJSON + EOSトークン (または無し)
        # ここでは「ノイズがあるのがダメ」と教えたいので、EOSは付けておく（文法違反ではない）。
        # 純粋なJSONとノイズ付きJSONの対比構造を作る。
        rejected = create_rejected(pure_json) + EOS_TOKEN
        
        dpo_entry = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
        dpo_dataset.append(dpo_entry)
        
    print(f"{len(dpo_dataset)} 件のDPOペアを生成しました（v2）。")
    
    # JSONL形式で保存
    print(f"{OUTPUT_FILE} に保存しています...")
    with open(OUTPUT_FILE, 'w') as f:
        for entry in dpo_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print("完了しました！ ✅")
    
    print("\n--- 生成例（最初の3件） ---")
    for i in range(min(3, len(dpo_dataset))):
        print(f"\n[例 {i+1}]")
        print(f"PROMPT (先頭50文字): {dpo_dataset[i]['prompt'][:50]}...")
        # EOSが見えるように末尾を表示
        print(f"CHOSEN (末尾20文字): ...{dpo_dataset[i]['chosen'][-20:]}")
        print(f"REJECTED (先頭50文字): {dpo_dataset[i]['rejected'][:50]}...")

if __name__ == "__main__":
    main()
