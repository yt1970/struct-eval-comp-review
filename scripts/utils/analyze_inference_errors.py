import json
import re
import sys

DEFAULT_INFERENCE_FILE = "inference__SFT3_2（0.44976）.json"

def analyze_inference_errors(file_path=None):
    """
    SFT v3の推論結果を分析し、以下のエラーを検出するスクリプトです。
    1. 思考過程（Chain of Thought）の混入 ("Approach: ...")
    2. 無効なJSON構文（Syntax Error）
    3. 重複ループ（同じ内容が繰り返されている）
    4. 不自然なキー重複（"dimensions"などが同じオブジェクト内に複数回出現）
    """
    inference_file = file_path if file_path else DEFAULT_INFERENCE_FILE

    try:
        with open(inference_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"エラー: ファイル {inference_file} が見つかりません。")
        return

    print(f"分析対象: {len(data)} 件のタスク ({inference_file})")
    print("-" * 40)
    
    cot_count = 0        # 思考過程が混入した数
    json_error_count = 0 # JSON構文エラーの数
    repetition_count = 0 # ループの疑いがある数
    duplicate_key_count = 0 # キー重複の数

    for item in data:
        task_id = item.get("task_id", "Unknown")
        generation = item.get("generation", "").strip()
        
        # 1. 思考過程（CoT）の混入チェック
        if "Approach:" in generation or generation.startswith("Here is"):
            cot_count += 1
            continue

        # 2. JSON構文の有効性チェック
        try:
            parsed = json.loads(generation)
            
            # 3. 構造的な異常チェック
            if re.search(r'"dimensions":\s*{.*?},\s*"dimensions":', generation, re.DOTALL):
                 duplicate_key_count += 1

            if re.search(r'"tags":\s*\[.*?\],\s*"tags":', generation, re.DOTALL):
                 duplicate_key_count += 1
            
        except json.JSONDecodeError as e:
            json_error_count += 1
            
            # エラーの詳細分析
            if generation.count("{") != generation.count("}"):
                pass # Unbalanced braces
            elif len(generation) > 1000 and len(set(generation.split())) < 50:
                repetition_count += 1
            else:
                pass # Just invalid syntax

    print(f"【分析結果サマリ】")
    print(f"総タスク数: {len(data)}")
    print(f"1. 思考過程の混入 (CoT Mistake): {cot_count} 件")
    print(f"2. JSON構文エラー (Invalid JSON): {json_error_count} 件")
    print(f"3. 重複ループの疑い (Repetition): {repetition_count} 件")
    print(f"4. キー重複 (Duplicate Keys): {duplicate_key_count} 件")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_inference_errors(sys.argv[1])
    else:
        analyze_inference_errors()
