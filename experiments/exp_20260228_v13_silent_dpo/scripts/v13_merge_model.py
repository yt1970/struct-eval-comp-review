import os
import torch
from unsloth import FastLanguageModel

# ==============================================================================
# V13 Model Merge Script: "The Silent Brain"
# ==============================================================================

# 1. Configuration
V13_DIR = "/Users/yutako/dev/struct-eval-comp/experiments/exp_20260228_v13_silent_dpo"
ADAPTER_DIR = os.path.join(V13_DIR, "v13_silent_dpo_adapter")
SAVE_DIR = "/Users/yutako/dev/struct-eval-comp/models/v13_silent_brain_merged"

# Hub Config
HF_USER = "satoyutaka"
HF_REPO = "llm2025_main_v13_silent_brain"

def main():
    print(f"🚀 Loading V13 Adapter from: {ADAPTER_DIR}...")
    
    # Load model and adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = ADAPTER_DIR,
        max_seq_length = 2048, # Increase for margin
        load_in_4bit = True,
    )

    print("🔗 Merging LoRA weights into base model...")
    # Merge and save to 16bit for quality
    model.save_pretrained_merged(SAVE_DIR, tokenizer, save_method = "merged_16bit")
    
    print(f"✅ Model successfully merged and saved to: {SAVE_DIR}")

    # Optional: Push to Hugging Face
    push_to_hub = input("\nDo you want to push to Hugging Face? (y/n): ")
    if push_to_hub.lower() == 'y':
        print(f"⬆️ Pushing to Hub: {HF_USER}/{HF_REPO}...")
        model.push_to_hub_merged(f"{HF_USER}/{HF_REPO}", tokenizer, save_method = "merged_16bit")
        print("✅ Push complete!")

if __name__ == "__main__":
    main()
