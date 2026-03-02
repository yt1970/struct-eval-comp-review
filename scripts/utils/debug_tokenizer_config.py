from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "adapters/adapter_legacy_silence_20260210_Silence_v2"

print(f"Loading tokenizer from {BASE_MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

print("-" * 30)
print(f"BOS Token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
print(f"EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"PAD Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"Special Tokens Map: {tokenizer.special_tokens_map}")
print("-" * 30)

# Check specific IDs (Qwen typical values)
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
print(f"<|im_end|> ID: {im_end_id}")

print("Checking Adapter Config...")
# Load adapter config simply
import json
import os

try:
    with open(os.path.join(ADAPTER_PATH, "adapter_config.json"), "r") as f:
        print(json.load(f))
except FileNotFoundError:
    print("Adapter config not found.")
