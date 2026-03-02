from transformers import AutoTokenizer
import os

model_path = "/Users/yutako/dev/struct-eval-comp/models/v14_dpo_silent_merged"
print(f"Loading tokenizer from {model_path}...")

# Load with trust_remote_code=True if needed for Qwen
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Re-save it. This should generate tokenizer_config.json, special_tokens_map.json, etc. correctly.
print(f"Re-saving tokenizer to {model_path}...")
tokenizer.save_pretrained(model_path)

print("✅ Tokenizer fixed and re-saved.")
