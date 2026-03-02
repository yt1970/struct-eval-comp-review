import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_075_ADAPTER = "adapters/adapter_legacy_sft_20260209_0709"
V12_LORA_DIR = "outputs/v12_dpo_on_075/checkpoint-54"
FINAL_MERGED_DIR = "models/v12_075_dpo_merged"

def manual_merge():
    print("🚀 手動マージを開始します...")
    device = "cpu" # メモリ節約のためCPUで
    
    # 1. ベース復元
    print("[1/3] ベースモデル + 0.75 SFT をロード...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    
    model = PeftModel.from_pretrained(base_model, SFT_075_ADAPTER)
    model = model.merge_and_unload()
    print("  ✅ 0.75 SFTマージ完了")
    
    # 2. V12 DPO LoRA をロード
    print("[2/3] V12 DPO LoRA をロード中...")
    model = PeftModel.from_pretrained(model, V12_LORA_DIR)
    model = model.merge_and_unload()
    print("  ✅ V12 DPOマージ完了")
    
    # 3. 保存
    print("[3/3] 保存中...")
    os.makedirs(FINAL_MERGED_DIR, exist_ok=True)
    model.save_pretrained(FINAL_MERGED_DIR)
    tokenizer.save_pretrained(FINAL_MERGED_DIR)
    
    # tokenizer_config.json 補正
    import shutil, glob
    base_tc = glob.glob("models/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/tokenizer_config.json")
    if base_tc:
        shutil.copy(base_tc[0], os.path.join(FINAL_MERGED_DIR, "tokenizer_config.json"))
        print("  ✅ tokenizer_config.json を補正しました")

    print(f"🎉 マージ完了！保存先: {FINAL_MERGED_DIR}")

if __name__ == "__main__":
    manual_merge()
