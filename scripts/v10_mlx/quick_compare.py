import json
import os
import shutil
from mlx_lm import load, generate

SNAPSHOT_PATH = "/Users/yutako/dev/struct-eval-comp/models/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
ADAPTER_ROOT = "/Users/yutako/dev/struct-eval-comp/experiments/exp_20260223_114045/adapter"
DATA_PATH = "data/public_150.json"

def run_test(iter_num):
    print(f"\n🚀 Testing Iteration {iter_num}...")
    
    # アダプタの差し替え
    target_file = f"{ADAPTER_ROOT}/{iter_num:07}_adapters.safetensors"
    shutil.copy(target_file, f"{ADAPTER_ROOT}/adapters.safetensors")
    
    # モデルのロード
    model, tokenizer = load(SNAPSHOT_PATH, adapter_path=ADAPTER_ROOT)
    
    # 最初の3問だけテスト
    with open(DATA_PATH, "r") as f:
        tasks = json.load(f)[:3]
    
    results = []
    for task in tasks:
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{task['instruction']}<|im_end|>\n<|im_start|>assistant\n"
        print(f"Processing Task ID: {task.get('id', 'N/A')}...")
        response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=1000)
        results.append({
            "iter": iter_num,
            "id": task.get("id"),
            "response": response
        })
    
    return results

if __name__ == "__main__":
    summary = []
    # 100 と 500 を順番にテスト
    for val in [100, 500]:
        try:
            res = run_test(val)
            summary.extend(res)
        except Exception as e:
            print(f"Error testing {val}: {e}")
            
    with open("iter_comparison_result.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n✅ Comparison finished! Check iter_comparison_result.json")
