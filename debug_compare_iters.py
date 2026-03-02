import mlx.core as mx
from mlx_lm import load, generate
import json

SNAPSHOT_PATH = "/Users/yutako/dev/struct-eval-comp/models/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
ADAPTER_BASE = "/Users/yutako/dev/struct-eval-comp/experiments/exp_20260223_114045/adapter"

# テスト問題 (Task 1: Ecosystem)
PROMPT = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Summarize a fictional ecosystem with detailed information about its climate, species, and threats.

Feature Requirements:
1. The field name is 'name', which is a string representing the official designation of the ecosystem.
Please output JSON code.<|im_end|>
<|im_start|>assistant
"""

def test_adapter(iter_num):
    print(f"\n--- 🧪 Testing Iteration {iter_num} ---")
    adapter_file = f"{ADAPTER_BASE}/{iter_num:07}_adapters.safetensors"
    
    # MLX-LM の load は adapter_path フォルダ内の adapters.safetensors を読むので
    # 一時的にコピーする
    import shutil
    shutil.copy(adapter_file, f"{ADAPTER_BASE}/adapters.safetensors")
    
    model, tokenizer = load(SNAPSHOT_PATH, adapter_path=ADAPTER_BASE)
    
    response = generate(model, tokenizer, prompt=PROMPT, verbose=False, max_tokens=200)
    print(f"Response:\n{response}")

if __name__ == "__main__":
    for i in [100, 500, 1000]:
        test_adapter(i)
