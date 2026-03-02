import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DPO3_ADAPTER_PATH = "/Users/yutako/Downloads/DPO３"
V10_ADAPTER_PATH = "outputs/train_sft_v3_20260211_Silence_v3/checkpoint-100"
V11_ADAPTER_PATH = "adapters/adapter_v11_surgical_dpo"
OUTPUT_DIR = "models/v11_fully_merged"

def main():
    print("=" * 60)
    print("🔧 V11 完全マージモデル生成")
    print("  Base → +DPO3 → +V10 → +V11 LoRA → 完全統合")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. ベースモデルをロード
    print("\n[1/5] ベースモデルをロード中...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,  # Colabと同じfloat16
        trust_remote_code=True
    ).to(device)

    # 2. DPO3 (Intelligence) をマージ
    print(f"[2/5] DPO3 (Intelligence) をマージ中... ({DPO3_ADAPTER_PATH})")
    model = PeftModel.from_pretrained(model, DPO3_ADAPTER_PATH)
    model = model.merge_and_unload()
    print("  ✅ DPO3 マージ完了")

    # 3. V10 Iter 100 (Silence) をマージ
    print(f"[3/5] V10 Iter 100 (Silence) をマージ中... ({V10_ADAPTER_PATH})")
    model = PeftModel.from_pretrained(model, V10_ADAPTER_PATH)
    model = model.merge_and_unload()
    print("  ✅ V10 マージ完了")

    # 4. V11 Surgical DPO LoRA をマージ
    print(f"[4/5] V11 Surgical DPO LoRA をマージ中... ({V11_ADAPTER_PATH})")
    model = PeftModel.from_pretrained(model, V11_ADAPTER_PATH)
    model = model.merge_and_unload()
    print("  ✅ V11 マージ完了")

    # 5. 完全マージモデルを保存
    print(f"\n[5/5] 完全マージモデルを保存中... → {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("🎉 完全マージ完了！")
    print(f"   保存先: {OUTPUT_DIR}")
    print("   次のステップ: HFにアップロードしてColabで推論")
    print("=" * 60)

if __name__ == "__main__":
    main()
