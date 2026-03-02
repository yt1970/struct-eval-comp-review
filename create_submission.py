import json
import os
from mlx_lm import load, generate
from tqdm import tqdm
from datetime import datetime
import pytz

# 設定
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
INPUT_FILE = "data/public_150.json"

def get_timestamp_jst():
    jst = pytz.timezone('Asia/Tokyo')
    return datetime.now(jst).strftime("%Y%m%d_%H%M")

def main():
    # 提出用フォルダ作成
    os.makedirs("submissions", exist_ok=True)
    
    # 手動で指定する場合や、自動で最新を探す場合に備えて
    # 今回は「合体させたモデル（fused_model_...）」を使う想定
    timestamp = get_timestamp_jst()
    # 直前に作ったはずの合体モデルのパス（手動でコマンドを叩く際に揃えてください）
    SAVE_PATH = f"fused_model_{timestamp}" 
    
    # 暫定：もし引数などで指定しなければ、とりあえず直近作ったフォルダ名か
    # 手動で合体させたフォルダを指定するように促す
    print(f"📦 モデルをロード中... (Path: {SAVE_PATH})")
    
    # モデルのロード（合体済みなので adapter_path は不要）
    try:
        model, tokenizer = load(SAVE_PATH)
    except:
        print(f"⚠️ {SAVE_PATH} が見つかりません。最新のフォルダ名を確認してください。")
        # フォルダ一覧を表示して選べるようにするなどの工夫も可能
        return

    print(f"📖 本番データを読み込み中: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    print(f"🔥 推論開始（150問）...")
    for item in tqdm(data):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["query"]}],
            tokenize=False,
            add_generation_prompt=True
        )
        response = generate(model, tokenizer, prompt=prompt, max_tokens=2048)
        results.append({
            "task_id": item["task_id"],
            "answer": response.strip()
        })

    output_file = f"submissions/submission_{timestamp}.json"
    print(f"📄 提出ファイルを保存中: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 完了！ 提出用ファイルが作成されました: {output_file}")

if __name__ == "__main__":
    main()
