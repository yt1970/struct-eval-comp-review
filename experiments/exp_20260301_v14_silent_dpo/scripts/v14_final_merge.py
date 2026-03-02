"""
V14 Final Merge Script
Merge Base + SFT Adapter + V14 DPO Adapter into a single model.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- Configurations ---
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_ID = "satoyutaka/LLM2026_SFT_0_again"
DPO_ADAPTER_ID = "satoyutaka/LLM2026_v14_silent_dpo_adapter"
HF_UPLOAD_REPO_MERGED = "satoyutaka/LLM2026_v14_silent_merged"
OUTPUT_DIR = "/Users/yutako/dev/struct-eval-comp/models/v14_dpo_silent_merged"

def main():
    print("🧠 Step 1: Loading Original Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # Mac/CPUで走らせる場合、low_cpu_mem_usage=True が必須
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu", # ローカルマージはCPUで十分
        low_cpu_mem_usage=True,
    )

    print(f"🔗 Step 2: Merging SFT Adapter ({SFT_ADAPTER_ID})...")
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_ID)
    model = model.merge_and_unload()

    print(f"🔗 Step 3: Merging V14 DPO Adapter ({DPO_ADAPTER_ID})...")
    model = PeftModel.from_pretrained(model, DPO_ADAPTER_ID)
    model = model.merge_and_unload()

    print(f"💾 Step 4: Saving final V14 merged model to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"🚀 Step 5: Uploading V14 final merged model to HF: {HF_UPLOAD_REPO_MERGED}...")
    try:
        hf_token = os.getenv("HF_TOKEN")
        model.push_to_hub(HF_UPLOAD_REPO_MERGED, token=hf_token)
        tokenizer.push_to_hub(HF_UPLOAD_REPO_MERGED, token=hf_token)
        print(f"✅ Successfully uploaded (Full Model) to {HF_UPLOAD_REPO_MERGED}!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("💡 Hint: Make sure HF_TOKEN environment variable is set or you are logged in via huggingface-cli.")

    print(f"✅ V14 Model is ready at {OUTPUT_DIR}!")

if __name__ == "__main__":
    main()
