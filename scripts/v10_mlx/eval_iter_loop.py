import json
import os
import shutil
import subprocess

SNAPSHOT_PATH = "/Users/yutako/dev/struct-eval-comp/models/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
ADAPTER_ROOT = "/Users/yutako/dev/struct-eval-comp/experiments/exp_20260223_114045/adapter"
MERGE_PATH = "/Users/yutako/dev/struct-eval-comp/models/v10_iter_eval_merged"

TEST_TASKS = [
    {
        "name": "Ecosystem (Complex)",
        "prompt": "Summarize a fictional underwater ecosystem called 'Azure-Deep' with information about its bioluminescent species, water pressure, and coral health. Output JSON with fields: name, average_depth, species_list[{name, glow_color}]."
    },
    {
        "name": "Ancient Artifact (The Aurora Test)",
        "prompt": "A silver sword named 'Sky-Breaker' found in the ruins of 'Odin-Peak'. Materials: Meteoritic iron and dragon bone. Output TOML format."
    },
    {
        "name": "Long CSV Data (Processing)",
        "prompt": "Convert this CSV to JSON: item,price,stock,category\nSword,150,5,Weapon\nShield,80,12,Armor\nPotion,10,100,Item\nRing,500,1,Accessory"
    }
]

def run_iter(iter_num):
    print(f"\n--- 🧪 Processing Iteration {iter_num} ---")
    
    # マージ
    shutil.copy(f"{ADAPTER_ROOT}/{iter_num:07}_adapters.safetensors", f"{ADAPTER_ROOT}/adapters.safetensors")
    if os.path.exists(MERGE_PATH):
        shutil.rmtree(MERGE_PATH)
    
    print("Merging model...")
    subprocess.run([
        ".venv/bin/python", "-m", "mlx_lm.fuse",
        "--model", SNAPSHOT_PATH,
        "--adapter-path", ADAPTER_ROOT,
        "--save-path", MERGE_PATH,
        "--de-quantize"
    ], check=True)
    
    print("\n--- 📝 Test Results for Iteration " + str(iter_num) + " ---")
    for task in TEST_TASKS:
        print(f"\n[Task: {task['name']}]")
        # 直接 mlx_lm.generate CLI を叩く (Python内load死回避)
        cmd = [
            ".venv/bin/python", "-m", "mlx_lm.generate",
            "--model", MERGE_PATH,
            "--max-tokens", "300",
            "--temp", "0.0",
            "--prompt", f"<|im_start|>user\n{task['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        # 不要な出力を削って回答だけ表示
        output = res.stdout
        if "assistant" in output:
            output = output.split("assistant")[-1]
        print(output.strip())

if __name__ == "__main__":
    import sys
    it = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_iter(it)
