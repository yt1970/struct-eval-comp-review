import os
import subprocess
import shutil

ADAPTER_DIR = "/Users/yutako/dev/struct-eval-comp/experiments/exp_20260223_114045/adapter"
EVAL_SCRIPT = "scripts/v10_mlx/eval_mlx_v10.py"
BASE_ADAPTER = os.path.join(ADAPTER_DIR, "adapters.safetensors")

def run_mini_eval(iter_num):
    print(f"--- 🧪 Testing Version: Iteration {iter_num} ---")
    
    # 1. アダプタファイルを差し替え
    target_safetensors = os.path.join(ADAPTER_DIR, f"{iter_num:07}_adapters.safetensors")
    if not os.path.exists(target_safetensors):
        print(f"❌ Error: {target_safetensors} not found!")
        return
    
    # 現在の adapters.safetensors をバックアップ（初回のみ）
    if not os.path.exists(BASE_ADAPTER + ".bak"):
        shutil.copy(BASE_ADAPTER, BASE_ADAPTER + ".bak")
        
    shutil.copy(target_safetensors, BASE_ADAPTER)
    
    # 2. 最初の5問だけを評価 (高速化のため)
    output_path = f"experiments/exp_20260223_114045/eval_mini_iter_{iter_num}.json"
    
    # 評価スクリプトを書き換えて「件数制限」をつけるのが面倒なので、
    # 評価スクリプト自体を呼ぶが、分析しやすいように出力だけチェックする。
    # 今回は scripts/v10_mlx/eval_mlx_v10.py が 150問全件やるので、
    # 手動で最初の数件終わったら止めるか、あるいは全件やらせる。
    
    print(f"Running evaluation...")
    cmd = [
        ".venv/bin/python", EVAL_SCRIPT,
        "--adapter", ADAPTER_DIR,
        "--output", output_path
    ]
    # 全件やると時間がかかるので、10件で止めるように eval_mlx_v10.py を改造した暫定版を使う
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    print("Quick Look (First Task Response):")
    try:
        with open(output_path, "r") as f:
            data = json.load(f)
            for d in data[:2]:
                print(f"ID: {d.get('id')} -> {d['response'][:200]}...")
    except:
        print("Could not read output.")

if __name__ == "__main__":
    import json
    # Iter 100 と 500 をテスト
    for i in [100, 500]:
        run_mini_eval(i)
