import json
from pathlib import Path

data_path = Path("/Users/yutako/dev/struct-eval-comp/data/public_150.json").expanduser()

def analyze_samples(file_path, count=10):
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # for i, line in enumerate(f):
        #     if i >= count: break
            # samples.append(json.loads(line))
            samples = json.load(f)
    
    # 1. データの「キー」を網羅
    all_keys = set().union(*(s.keys() for s in samples))
    print(f"🔑 全体で使われているキー: {all_keys}")

    # 2. テキストの長さの統計
    lengths = [len(str(s.get('instruction', ''))) + len(str(s.get('input', ''))) for s in samples]
    print(f"📏 テキスト長: 最短={min(lengths)}, 最長={max(lengths)}, 平均={sum(lengths)/len(lengths):.1f}")

    # 3. 最初の3件だけ詳細表示
    for i in range(min(10, len(samples))):
        print(f"\n--- Sample {i} ---")
        print(json.dumps(samples[i], indent=2, ensure_ascii=False))

# 実行（今夜データを入れたら！）
analyze_samples(data_path)