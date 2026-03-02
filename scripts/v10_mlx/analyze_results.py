import json
import re

RESULTS_PATH = "/Users/yutako/dev/struct-eval-comp/experiments/exp_20260223_114045/eval_results_merged_150.json"

def analyze():
    print(f"📊 Analyzing V10 results from {RESULTS_PATH}...")
    try:
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return

    total = len(data)
    valid_json = 0
    silent_perfect = 0
    failures = []

    for item in data:
        response = item.get("response", "").strip()
        
        # 1. 沈黙チェック
        if response.startswith("{") and response.endswith("}"):
            silent_perfect += 1
        elif response.startswith("```json") or response.endswith("```"):
            # Markdown囲みが残っている場合は減点
            pass
        
        # 2. 構文チェック
        try:
            json.loads(response)
            valid_json += 1
        except Exception:
            # Markdown除去を試みる
            clean_res = re.sub(r"```json\s*|\s*```", "", response).strip()
            try:
                json.loads(clean_res)
                valid_json += 1
                # 構文は合ってるけどお喋りや囲みがある場合
            except:
                failures.append(item)

    print(f"\n--- Statistics ---")
    print(f"Total Tasks: {total}")
    print(f"Perfectly Silent: {silent_perfect} ({silent_perfect/total*100:.1f}%)")
    print(f"Syntactically Valid: {valid_json} ({valid_json/total*100:.1f}%)")
    print(f"Critical Failures (Broken Format): {total - valid_json}")

    print(f"\n--- Failure Details (First 5) ---")
    for f in failures[:5]:
        print(f"\nID: {f.get('id', 'N/A')}")
        print(f"Response snippet: {f['response'][:300]}")

analyze()
