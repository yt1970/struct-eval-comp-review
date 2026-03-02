"""
SFT v8 データ生成: dpo_v7_surgical.jsonl の chosen 列を SFT 形式に変換

データの由来:
  structured-hard-sft-4k.jsonl (HuggingFace 構造化データ 4k タスク)
    ↓ scripts/v7_surgical_dpo/generate_dpo_v7.py
    ↓ - clean_output(): Markdown/CoT/おしゃべり除去 → chosen
    ↓ - make_rejected(): chosen にラッパー追加 → rejected
    ↓ - seed=42, 短すぎる回答(<20文字)スキップ
  dpo_v7_surgical.jsonl (301件, DPO v7/v7.1 で使用)
    ↓ このスクリプト (generate_sft_v8_data.py)
    ↓ - prompt + chosen のみ抽出、rejected は不要 (SFT なので)
  sft_v8_lmhead.jsonl (301件, SFT v8 で使用)
"""
import json

INPUT_PATH = "data/dpo_v7_surgical.jsonl"
OUTPUT_PATH = "data/sft_v8_lmhead.jsonl"

def main():
    records = []
    with open(INPUT_PATH, 'r') as f:
        for line in f:
            item = json.loads(line)
            record = {
                "messages": [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["chosen"]}
                ]
            }
            records.append(record)

    with open(OUTPUT_PATH, 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated {len(records)} SFT records -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
