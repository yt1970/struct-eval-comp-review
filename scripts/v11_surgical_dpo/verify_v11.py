import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- 設定 ---
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "adapters/adapter_v11_surgical_dpo"

TEST_CASES = [
    {
        "id": "A_Ecosystem",
        "description": "Complex extraction (Azure-Deep)",
        "prompt": "Summarize a fictional underwater ecosystem called 'Azure-Deep' with information about its bioluminescent species, water pressure, and coral health. Output JSON with fields: name, average_depth_m, species_list[{name, glow_color}]."
    },
    {
        "id": "B_Artifact",
        "description": "Hallucination Check (Sky-Breaker)",
        "prompt": "Extract details about an ancient silver sword named 'Sky-Breaker' found in 'Odin-Peak', made of Dragon bone. Output JSON."
    },
    {
        "id": "C_CSV",
        "description": "Data Conversion (Han Solo)",
        "prompt": "Convert this CSV to JSON: item,owner,ship\nMedal of Bravery,Han Solo,Millennium Falcon"
    }
]

def main():
    print("=" * 60)
    print(f"🚀 V11 Surgical DPO 3問検証 (抜き打ちテスト)")
    print(f"   Model: {BASE_MODEL_ID}")
    print(f"   Adapter: {ADAPTER_PATH}")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f" usando dispositivo: {device}")

    # 1. ロード
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    
    print("\n🔗 アダプターをロード中...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    # 2. テスト実行
    for case in TEST_CASES:
        print(f"\n" + "-" * 40)
        print(f"【Test: {case['id']}】({case['description']})")
        print(f"Prompt: {case['prompt']}")
        print("-" * 40)

        # ChatML テンプレート適用
        messages = [{"role": "user", "content": case["prompt"]}]
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt",
            return_dict=True
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False, # 決定論的
                pad_token_id=tokenizer.eos_token_id
            )

        # 回答部分のみ抽出
        input_length = inputs["input_ids"].shape[-1]
        response = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
        
        print(f"Output:\n{response.strip()}")
        
        # 簡易評価
        is_silent = not response.lstrip().startswith(("Sure", "Certainly", "Here", "Below", "```"))
        print(f"\n[評価] 沈黙チェック: {'OK (一文字目から出力)' if is_silent else 'NG (お喋りあり)'}")
        
    print("\n" + "=" * 60)
    print("✅ 検証完了")

if __name__ == "__main__":
    main()
