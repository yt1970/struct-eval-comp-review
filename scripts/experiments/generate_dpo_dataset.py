import json
import re
import os

def clean_output(text):
    """
    LLMを使わずに、正規表現と文字列操作だけで不純物を完全に除去する。
    """
    # 1. 「Output:」という文字列があれば、それ以降を抽出（CoTの思考過程を捨てる）
    if "Output:" in text:
        text = text.split("Output:")[-1]
    
    # 2. Markdownのコードブロックタグ（```json, ```yaml等）を削除
    text = re.sub(r"```[a-z]*\n", "", text)
    text = text.replace("```", "")
    
    # 3. 最初に出現する記号 { or < or [ or 文字 より前にあるお喋りをカット
    # (JSON, XML, YAML, TOML すべてに対応するため、より慎重に)
    # 実際には clean_output の冒頭で余計なものは削られているはずだが、念のため。
    
    return text.strip()

def extract_user_query(prompt):
    """
    ChatML形式のプロンプトからユーザーの質問文だけを抜き出す。
    """
    match = re.search(r"<\|im_start\|>user\n(.*?)\n<\|im_end\|>", prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return prompt.strip()

def main():
    input_path = "/Users/yutako/dev/struct-eval-comp/data/hf_datasets/dpo-dataset-qwen-cot.jsonl"
    output_path = "/Users/yutako/dev/struct-eval-comp/data/train_data/dpo_pure_format.jsonl"

    if not os.path.exists(input_path):
        print(f"❌ 入力ファイルが見つかりません: {input_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dpo_data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                
                query = extract_user_query(item["prompt"])
                
                # Chosen: ここが「理想の回答（思考過程なし、Markdownなし）」
                chosen_clean = clean_output(item["chosen"])
                
                # Rejected: ここが「悪い回答（Markdown入り、あるいは冗長なもの）」
                # 元の rejected をそのまま使うことで、モデルがやりがちなミスを「ダメな例」として教える
                rejected_messy = item["rejected"]
                
                # フィルタリング: Chosen が空でないことを確認
                if chosen_clean and query:
                    dpo_data.append({
                        "prompt": query,
                        "chosen": chosen_clean,
                        "rejected": rejected_messy
                    })
            except Exception as e:
                continue

    # 書き出し
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dpo_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ DPOデータの生成が完了しました（全 {len(dpo_data)} 件）")
    print(f"📍 ファイル: {output_path}")

if __name__ == "__main__":
    main()
