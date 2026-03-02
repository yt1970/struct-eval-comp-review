import json
import os
import argparse
from tqdm import tqdm
from mlx_lm import load, generate

# ==========================================
# V10 MLX Evaluation Script
# ==========================================

SNAPSHOT_PATH = "/Users/yutako/dev/struct-eval-comp/models/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"

def main():
    parser = argparse.ArgumentParser(description="Evaluate MLX adapter on 150 tasks")
    parser.add_argument("--adapter", type=str, default=None, help="Path to adapter_path (optional)")
    parser.add_argument("--model", type=str, default=SNAPSHOT_PATH)
    parser.add_argument("--data", type=str, default="data/public_150.json")
    parser.add_argument("--output", type=str, default="experiments/v10_eval_result.json")
    args = parser.parse_args()

    # 1. モデルとアダプタのロード
    print(f"Loading model {args.model}...")
    if args.adapter and args.adapter.strip():
        print(f"Using adapter: {args.adapter}")
        model, tokenizer = load(args.model, adapter_path=args.adapter)
    else:
        print("No adapter specified. Loading base model/merged model.")
        model, tokenizer = load(args.model)

    # 2. 評価データの読み込み
    with open(args.data, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    results = []
    print(f"🔍 Evaluating {len(tasks)} tasks...")

    for item in tqdm(tasks):
        query = item.get("query", "")
        task_id = item.get("id", "")

        # チャットテンプレート適用 (沈黙のために、userメッセージのみ)
        messages = [{"role": "user", "content": query}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 生成 (沈黙を保つため、temperature=0)
        # mlx_lm.generate はデフォルトでいい感じに生成してくれます
        response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=2048)

        print(f"\n--- [Task ID: {task_id}] ---")
        print(response)
        print("-" * 30)

        results.append({
            "id": task_id,
            "query": query,
            "response": response
        })

    # 3. 結果の保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ Evaluation complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()
