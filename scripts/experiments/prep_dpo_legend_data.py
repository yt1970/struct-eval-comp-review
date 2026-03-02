import json
import re
import random
import torch
from transformers import pipeline
from tqdm import tqdm

# --- 設定 ---
SOURCE_FILE = "data/hf_datasets/dpo-dataset-qwen-cot.jsonl"
OUTPUT_FILE = "data/train_data/dpo_structure_fix_v1.jsonl"
BERT_MODEL = "bert-base-multilingual-cased"

def clean_output(text):
    """
    'Output:\n' 以降を抽出し、マークダウン装飾を除去するプログラム
    """
    if "Output:" in text:
        text = text.split("Output:")[1]
    
    # マークダウンブロックの除去
    text = re.sub(r"```[a-z]*\n?", "", text)
    text = text.replace("```", "")
    return text.strip()

def corrupt_nesting(text):
    """
    フラットなキーを勝手に階層化する『改悪』プログラム
    (例: mission_1_name -> mission_1: { name: ... })
    ※簡易的な正規表現置換でミスを模倣
    """
    # JSONの場合の置換
    text = re.sub(r'"([a-z0-9]+)_([a-z0-9]+)":', r'"\1": {"\2":', text)
    return text

def add_hallucination(text):
    """
    値を 'Aurora' や '12.5' に捏造する『改悪』プログラム
    """
    # 文字列の値を Aurora に、数値を 12.5 にランダム置換
    text = re.sub(r'": "[^"]+"', '": "Aurora"', text)
    text = re.sub(r'": \d+(\.\d+)?', '": 12.5', text)
    return text

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading BERT for augmentation on {device}...")
    fill_mask = pipeline("fill-mask", model=BERT_MODEL, device=device)

    print(f"Reading source: {SOURCE_FILE}")
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    dpo_data = []

    for item in tqdm(lines[:1000]): # 1000件程度で十分効果があります
        # 1. 指示文の抽出（im_start等のタグを除去）
        prompt_raw = item["prompt"]
        user_match = re.search(r"<\|im_start\|>user\n(.*?)\n<\|im_end\|>", prompt_raw, re.DOTALL)
        if not user_match: continue
        instruction = user_match.group(1).strip()

        # 2. Chosen (クリーンな正解)
        chosen_clean = clean_output(item["chosen"])

        # 3. Rejected (ミス再現)
        # ランダムに『改悪』パターンを選択
        fail_type = random.choice(["nesting", "hallucination", "markdown"])
        if fail_type == "nesting":
            rejected = corrupt_nesting(chosen_clean)
        elif fail_type == "hallucination":
            rejected = add_hallucination(chosen_clean)
        else:
            # DPOデータ自体の rejected (解説付きで汚いもの) を使用
            rejected = item["rejected"]

        # 4. BERTによる指示文拡張 (25%の確率で実施)
        if random.random() < 0.25 and len(instruction.split()) > 5:
            try:
                words = instruction.split()
                idx = random.randint(0, len(words)-1)
                masked_prompt = " ".join(words[:idx] + ["[MASK]"] + words[idx+1:])
                # promptの長さ制限
                preds = fill_mask(masked_prompt[:512])
                instruction = masked_prompt.replace("[MASK]", preds[0]["token_str"])
            except: pass

        # 5. DPOフォーマット（伝説形式）
        dpo_entry = {
            "instruction": f"### 指示\n{instruction}\n\n### 応答\n",
            "chosen": chosen_clean,
            "rejected": rejected
        }
        dpo_data.append(dpo_entry)

    print(f"Saving {len(dpo_data)} DPO pairs to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in dpo_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("Success! 🚀")

if __name__ == "__main__":
    main()
