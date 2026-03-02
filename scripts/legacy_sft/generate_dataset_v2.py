import json
import random
import os

# --- 設定 ---
DATA_DIR = "data/hf_datasets"
TRAIN_FILE = "data/golden_train.jsonl"
VAL_FILE = "data/golden_val.jsonl"
TEST_FILE = "data/golden_test.jsonl"

# 運営承認済みファイルのパス
FILES = {
    "mix_5k": os.path.join(DATA_DIR, "structured-5k-mix-sft.jsonl"),
    "sft_cot": os.path.join(DATA_DIR, "sft_1-1.jsonl"),
    "hard_4k": os.path.join(DATA_DIR, "structured-hard-sft-4k.jsonl")
}

# 取得件数の目安
TARGET_COUNTS = {
    "mix_5k": 4000,
    "sft_cot": 4000,
    "hard_4k": 2000
}

def load_jsonl(path):
    data = []
    if not os.path.exists(path):
        print(f"❌ ファイルが見つかりません: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    print(f"Skipping bad line in {path}: {e}")
    return data

def main():
    all_selected_data = []
    
    # 1. 各ファイルから目的の件数をサンプリング (運営承認データの純粋なミックス)
    for key, path in FILES.items():
        print(f"Loading {path}...")
        data = load_jsonl(path)
        target = TARGET_COUNTS[key]
        
        if len(data) >= target:
            sampled = random.sample(data, target)
        else:
            print(f"Warning: {key} has only {len(data)} items, taking all.")
            sampled = data
            
        all_selected_data.extend(sampled)
        print(f"Added {len(sampled)} items from {key}")

    # 2. シャッフル
    print(f"Total items: {len(all_selected_data)}. Shuffling...")
    random.shuffle(all_selected_data)
    
    # 3. 分割 (合計10,000件の場合)
    # Train: 8,000 (80%), Val: 1,000 (10%), Test: 1,000 (10%)
    total = len(all_selected_data)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    train_data = all_selected_data[:train_end]
    val_data = all_selected_data[train_end:val_end]
    test_data = all_selected_data[val_end:]
    
    # 📝 運営への報告用: 
    # train_augmented_bert.jsonl 等、BERTによる追加拡張はこの中身に対して
    # 運営の許可範囲（エンコーダーBERT使用可）で行うことも可能ですが、
    # まずは純粋なミックスで基礎力を固めます。

    # 4. 保存
    for data, path in [(train_data, TRAIN_FILE), (val_data, VAL_FILE), (test_data, TEST_FILE)]:
        print(f"Saving {len(data)} items to {path}...")
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"✅ データの準備が完了しました！")
    print(f"   - 学習用: {TRAIN_FILE}")
    print(f"   - 検証用: {VAL_FILE}")
    print(f"   - テスト用: {TEST_FILE}")

if __name__ == "__main__":
    # シード値を固定して再現性を確保
    random.seed(42)
    main()
