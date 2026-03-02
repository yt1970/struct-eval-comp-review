import json
import os
import random
import torch
from transformers import pipeline
from tqdm import tqdm

# --- 設定 ---
SOURCE_FILE = "data/hf_datasets/structured-hard-sft-4k.jsonl"
OUTPUT_DIR = "data/train_data"
TRAIN_TOTAL = 500  # 元データの使用件数（拡張により最終的に3倍になります）
VALID_TOTAL = 50   # 検証データの件数
AUGMENT_PER_ITEM = 2  # 1件につき何パターンのBERT言い換えを作るか
MODEL_NAME = "bert-base-multilingual-cased" # 日本語・英語混在に対応

def clean_content(text):
    """
    回答の冒頭にある『Here is the XML:』などの不要な前置きを削除し、
    <root> または { から始まるクリーンな状態にします。
    """
    xml_start = text.find("<root>")
    json_start = text.find("{")
    start_idx = -1
    if xml_start != -1 and json_start != -1:
        start_idx = min(xml_start, json_start)
    elif xml_start != -1: start_idx = xml_start
    elif json_start != -1: start_idx = json_start
    
    if start_idx != -1:
        return text[start_idx:].strip()
    return text.strip()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"Loading BERT pipeline on {device}...")
    fill_mask = pipeline("fill-mask", model=MODEL_NAME, device=device)

    print(f"Reading source data: {SOURCE_FILE}")
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        all_lines = [json.loads(line) for line in f if line.strip()]

    # データのシャッフルと抽出
    random.shuffle(all_lines)
    train_subset = all_lines[:TRAIN_TOTAL]
    valid_subset = all_lines[TRAIN_TOTAL : TRAIN_TOTAL + VALID_TOTAL]

    # --- 学習データの作成と拡張 ---
    final_train_data = []
    print("Augmenting Training Data...")
    for item in tqdm(train_subset):
        # 1. オリジナルのクリーニング
        item["messages"][1]["content"] = clean_content(item["messages"][1]["content"])
        final_train_data.append(item)

        # 2. BERTによる言い換え（拡張）
        user_text = item["messages"][0]["content"]
        words = user_text.split()
        
        # 指示内容を壊さないよう、ある程度長い文のみ言い換えを実施
        if len(words) > 10:
            for _ in range(AUGMENT_PER_ITEM):
                try:
                    # ランダムな1単語を [MASK] に置換
                    target_idx = random.randint(0, len(words) - 1)
                    masked_words = words.copy()
                    masked_words[target_idx] = "[MASK]"
                    masked_prompt = " ".join(masked_words)
                    
                    # BERT推論 (512トークン制限に注意)
                    preds = fill_mask(masked_prompt[:512])
                    new_word = preds[0]["token_str"]
                    
                    # 新しいメッセージを作成
                    new_item = {
                        "id": f"aug_{item['id']}_{random.getrandbits(16)}",
                        "messages": [
                            {"role": "user", "content": masked_prompt.replace("[MASK]", new_word)},
                            {"role": "assistant", "content": item["messages"][1]["content"]}
                        ]
                    }
                    final_train_data.append(new_item)
                except Exception:
                    continue

    # --- 保存 ---
    train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    valid_path = os.path.join(OUTPUT_DIR, "valid.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for entry in final_train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    with open(valid_path, "w", encoding="utf-8") as f:
        for entry in valid_subset:
            # 検証データもクリーニングだけは実施
            entry["messages"][1]["content"] = clean_content(entry["messages"][1]["content"])
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Success!")
    print(f" - Train: {len(final_train_data)} samples (Original {TRAIN_TOTAL} + Augmented)")
    print(f" - Valid: {len(valid_subset)} samples")
    print(f" - Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()