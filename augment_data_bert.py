import json
import os
import random
import torch
from transformers import pipeline
from tqdm import tqdm

def main():
    # 入出力パス
    input_path = "data/hf_datasets/structured-hard-sft-4k.jsonl"
    output_path = "data/train_augmented_bert.jsonl"
    
    # 準備
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("🚀 BERTパイプラインをロード中... (nlpaugを使わない安定版)")
    # デバイスの設定（M4なら mps, なければ cpu）
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # BERTによる穴埋めモデルを直接ロード
    fill_mask = pipeline(
        "fill-mask", 
        model='bert-base-multilingual-cased',
        device=device
    )

    print(f"📖 データを読み込み中: {input_path}")
    if not os.path.exists(input_path):
        print(f"❌ エラー: {input_path} が見つかりません。")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    augmented_data = []
    
    print(f"🔥 データ拡張を開始します（BERT穴埋め法）...")
    for line in tqdm(lines):
        if not line.strip(): continue
        item = json.loads(line)
        
        # 1. オリジナルを保存
        augmented_data.append(json.dumps(item, ensure_ascii=False))
        
        # userの質問文を特定
        user_msg_idx = -1
        for i, msg in enumerate(item["messages"]):
            if msg["role"] == "user":
                user_msg_idx = i
                break
        
        if user_msg_idx == -1:
            continue

        original_text = item["messages"][user_msg_idx]["content"]
        words = original_text.split()
        
        # 短すぎる文は言い換えを諦める
        if len(words) < 5:
            continue

        # 2パターンの言い換えを生成
        for _ in range(2):
            try:
                # ランダムに1単語選んで [MASK] に置き換える
                # (指示文を壊さないよう、適当な位置の単語を狙う)
                target_idx = random.randint(0, len(words) - 1)
                masked_words = words.copy()
                masked_words[target_idx] = "[MASK]"
                masked_text = " ".join(masked_words)
                
                # BERTに穴埋めさせる
                # 全文が長すぎるとBERTが死ぬので、念のため短く制限
                predictions = fill_mask(masked_text[:512])
                
                # 最も自信のある候補を採用 (オリジナルと同じ単語なら2番目を採用)
                best_subst = predictions[0]["token_str"]
                if best_subst.strip() == words[target_idx] and len(predictions) > 1:
                    best_subst = predictions[1]["token_str"]
                
                # 新しい文章を組み立てる
                new_words = words.copy()
                new_words[target_idx] = best_subst
                new_text = " ".join(new_words)
                
                # データをコピーして追加
                new_item = item.copy()
                new_item["messages"] = [m.copy() for m in item["messages"]]
                new_item["messages"][user_msg_idx]["content"] = new_text
                augmented_data.append(json.dumps(new_item, ensure_ascii=False))
                
            except Exception:
                continue

    print(f"📄 拡張データを保存中: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in augmented_data:
            f.write(line + "\n")

    print(f"✅ 完了！ 元の {len(lines)} 件 -> {len(augmented_data)} 件に増幅されました。")

if __name__ == "__main__":
    main()
