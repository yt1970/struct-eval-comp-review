from mlx_lm import load, generate
import json

MODEL_PATH = "/Users/yutako/dev/struct-eval-comp/models/v10_merged_final"
DATA_PATH = "data/public_150.json"

print("🔥 Loading V10 Final Model...")
model, tokenizer = load(MODEL_PATH)

print("🔍 Loading a sample task...")
with open(DATA_PATH, "r") as f:
    data = json.load(f)
    sample = data[0] # 一問目

messages = [{"role": "user", "content": sample["query"]}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(f"\nQUERY:\n{sample['query']}\n")
print("🤖 ASSISTANT (V10):")
print("=" * 30)
# 画面に直接流し込むために verbose=True にします！！
response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=1000)
print("=" * 30)
