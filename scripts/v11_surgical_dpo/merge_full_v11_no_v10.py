import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DPO3_ADAPTER_PATH = "/Users/yutako/Downloads/DPO３"
V11_ADAPTER_PATH = "adapters/adapter_v11_surgical_dpo"
OUTPUT_DIR = "models/v11_merged_no_v10"

def main():
    print("=" * 60)
    print("🔧 V11 完全マージ (V10なし版)")
    print("  Base → +DPO3 → +V11 LoRA → 完全統合")
    print("  ※V10はロゼッタストーン汚染のため除外")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # 1. ベースモデル
    print("\n[1/4] ベースモデルをロード中...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(device)

    # 2. DPO3 (Intelligence) のみマージ
    print(f"[2/4] DPO3 (Intelligence 0.77) をマージ中...")
    model = PeftModel.from_pretrained(model, DPO3_ADAPTER_PATH)
    model = model.merge_and_unload()
    print("  ✅ DPO3 マージ完了")

    # 3. V11 Surgical DPO LoRA をマージ (V10はスキップ！)
    print(f"[3/4] V11 Surgical DPO LoRA をマージ中...")
    model = PeftModel.from_pretrained(model, V11_ADAPTER_PATH)
    model = model.merge_and_unload()
    print("  ✅ V11 マージ完了")

    # 4. 保存
    print(f"\n[4/4] 保存中... → {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ベースモデルのtokenizer_configで上書き (Colab互換性修正)
    import shutil
    import glob
    base_tc = glob.glob("models/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/tokenizer_config.json")
    if base_tc:
        shutil.copy(base_tc[0], os.path.join(OUTPUT_DIR, "tokenizer_config.json"))
        print("  ✅ tokenizer_config.json をベースモデル版で上書き (Colab互換)")

    print("\n" + "=" * 60)
    print("🎉 V11 (V10なし) 完全マージ完了！")
    print("=" * 60)

if __name__ == "__main__":
    main()
