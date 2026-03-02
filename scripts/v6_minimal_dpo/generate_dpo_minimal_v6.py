import json
import re
import os
import random

# ソース: SFT v3 で使用され、CoT混入の原因となった元データ (structured-hard-sft-4k)
# ここには多様なフォーマット(JSON/XML/CSV/YAML)が含まれているため、これを利用して修正データを作成します。
SOURCE_FILE = "data/hf_datasets/structured-hard-sft-4k.jsonl"
OUTPUT_FILE = "data/dpo_minimal_v6.jsonl"
TARGET_SIZE = 300  # 最小限の学習で済ませるため、データセットサイズを絞る

def clean_cot(text):
    """
    思考過程（CoT）や余計な挨拶文を除去し、純粋な回答のみを抽出する関数。
    これが「Chosen（正解）」のデータになります。
    """
    # 1. Markdownのコードブロック記法 (```json 等) を削除
    text = re.sub(r'```\w*', '', text) 
    text = text.replace("```", "")

    # 2. SFT v3で混入していた「Approach: ... Output:」の思考ブロックを削除
    pattern = r"Approach:.*?Output:\n"
    if re.search(pattern, text, re.DOTALL):
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    
    # 3. よくあるチャットの挨拶文（Sure!, Here is...等）を削除
    chatty_patterns = [
        r"^Sure!.*?:",
        r"^Here is the JSON.*?:",
        r"^Certainly!.*?:",
        r"^I can help.*?:",
        r"^Below is the.*?:",
    ]
    for pattern in chatty_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # 4. 末尾の "Note:" セクションがあれば削除
    if "Note:" in text:
        text = text.split("Note:")[0]

    return text.strip()

def make_chatty(text):
    """
    あえておしゃべりな（Chattyな）回答を人工的に生成する関数。
    これが「Rejected（不正解）」のデータになります。
    """
    prefixes = [
        "Sure! Here is the data you requested:\n",
        "Certainly! I have converted the following for you:\n",
        "Here is the result:\n",
        # MarkdownもRejectedに含める！
        "Here is the code:\n```json\n",
        # SFT v3 で問題になった思考パターンも再現してRejectedにする
        "Approach:\n1. Analyze the request.\n2. Convert the format.\n3. Output result.\n\nOutput:\n",
    ]
    prefix = random.choice(prefixes)
    
    # prefixがMarkdown開始を含む場合、末尾に閉じタグを追加する
    if "```" in prefix:
        return f"{prefix}{text}\n```"
    else:
        return f"{prefix}{text}"

def main():
    print(f"Minimal DPO データセット (v6) を生成中... 目標数: {TARGET_SIZE}件")
    print(f"方針: 元データの思考過程を削除したものを「正解(Chosen)」、元の思考過程付き(または人工的に付与)を「不正解(Rejected)」としてペアを作成")
    
    data = []
    
    try:
        with open(SOURCE_FILE, 'r') as f:
            all_lines = f.readlines()
            # ランダムにシャッフルして、特定のタスクタイプに偏らないようにする
            random.shuffle(all_lines)
            
            for line in all_lines:
                if len(data) >= TARGET_SIZE:
                    break
                
                try:
                    item = json.loads(line)
                    if 'messages' in item:
                        user_msg = next((m['content'] for m in item['messages'] if m['role'] == 'user'), None)
                        asst_msg = next((m['content'] for m in item['messages'] if m['role'] == 'assistant'), None)
                        
                        if user_msg and asst_msg:
                            original_text = asst_msg
                            
                            # 1. まず元の回答をクリーニングして「沈黙（Silent）」版を作る -> Chosen
                            cleaned_text = clean_cot(original_text)
                            
                            # 2. Rejected（不正解）を作る
                            is_original_dirty = (len(cleaned_text) < len(original_text) - 10) 
                            
                            if is_original_dirty:
                                # 元が汚い -> そのままRejected
                                chosen = cleaned_text
                                rejected = original_text
                            else:
                                # 元がきれい -> 人工的に汚してRejected
                                chosen = cleaned_text
                                rejected = make_chatty(cleaned_text)
                            
                            # 短すぎる回答は除外
                            if len(chosen) < 10:
                                continue
                                
                            data.append({
                                "prompt": user_msg,
                                "chosen": chosen,
                                "rejected": rejected
                            })
                except json.JSONDecodeError:
                    continue

        # 保存
        with open(OUTPUT_FILE, 'w') as out:
            for item in data:
                out.write(json.dumps(item) + "\n")
        
        print(f"完了: {len(data)} 件のDPOペアを生成しました -> {OUTPUT_FILE}")
        
    except FileNotFoundError:
        print(f"エラー: ソースファイル {SOURCE_FILE} が見つかりません。")

if __name__ == "__main__":
    main()
