import os
from huggingface_hub import HfApi

# --- Configurations ---
REPO_ID = "satoyutaka/LLM2026_v14_silent_merged"
# Corrected file is already at this local path
LOCAL_FILE = "/Users/yutako/dev/struct-eval-comp/models/v14_dpo_silent_merged/tokenizer_config.json"
REMOTE_PATH = "tokenizer_config.json"
HF_TOKEN = "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"

def main():
    print(f"🚀 Uploading corrected {REMOTE_PATH} to {REPO_ID}...")
    api = HfApi()
    
    try:
        api.upload_file(
            path_or_fileobj=LOCAL_FILE,
            path_in_repo=REMOTE_PATH,
            repo_id=REPO_ID,
            token=HF_TOKEN,
            commit_message="Fix tokenizer_config.json AttributeError: 'list' object has no attribute 'keys'"
        )
        print("✅ Successfully updated tokenizer_config.json on Hugging Face!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    main()
