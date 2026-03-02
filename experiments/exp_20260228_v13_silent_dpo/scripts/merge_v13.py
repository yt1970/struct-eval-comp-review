import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    print("=" * 60)
    print("🔥 V13 (Silent Brain) Merge Script")
    print("=" * 60)

    # 1. Paths
    BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
    SFT_ADAPTER_PATH = "satoyutaka/LLM2026_SFT_0_again"
    DPO_ADAPTER_PATH = "experiments/exp_20260228_v13_silent_dpo/v13_silent_dpo_adapter"
    FINAL_MERGE_DIR = "models/v13_dpo_silent_merged"

    # Memory Check
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # 2. Load Base Model
    print(f"\n[1/4] Loading Full Precision Base Model: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device,
    )

    # 3. Apply and Merge 0.75 SFT Adapter
    print(f"\n[2/4] Merging SFT Adapter from Hugging Face: {SFT_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()
    print("✅ SFT Adapter merged.")

    # 4. Apply and Merge V13 DPO Adapter (Local)
    print(f"\n[3/4] Merging V13 DPO Adapter from Local: {DPO_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, DPO_ADAPTER_PATH)
    model = model.merge_and_unload()
    print("✅ V13 DPO Adapter merged.")

    # 5. Save Final Model
    print(f"\n[4/4] Saving Final Merged Model to: {FINAL_MERGE_DIR}")
    os.makedirs(FINAL_MERGE_DIR, exist_ok=True)
    model.save_pretrained(FINAL_MERGE_DIR, safe_serialization=True)
    tokenizer.save_pretrained(FINAL_MERGE_DIR)

    print("\n" + "=" * 60)
    print("🎉 V13 Merge Complete!")
    print(f"   Saved to: {FINAL_MERGE_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
