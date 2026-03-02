import mlx.core as mx
from mlx_lm import load, generate

ADAPTER_PATH = "/Users/yutako/dev/struct-eval-comp/experiments/exp_20260223_114045/adapter"
MODEL_PATH = "/Users/yutako/dev/struct-eval-comp/models/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"

print("Trying load...")
try:
    model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)
    print("Load success!")
    
    prompt = "Hello, tell me a quick fact about Tokyo."
    # apply_chat_template を使わずに素のテキストでテスト
    print("Generating...")
    response = generate(model, tokenizer, prompt=prompt, verbose=True)
    print("\n--- Response ---")
    print(response)
    print("----------------")
except Exception as e:
    print(f"FAILED: {e}")
