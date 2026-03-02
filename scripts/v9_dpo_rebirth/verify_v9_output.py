import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==============================================================================
# V9 (DPO Rebirth) 検証用スクリプト
# ==============================================================================

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "adapters/adapter_dpo_V9_Rebirth"

def test_inference(test_prompt):
    print(f"\n🔄 モデルとトークナイザーをロード中...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # ベースモデル
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # アダプタのロード（V9 DPOアダプタ）
    print(f"📦 アダプタをロード中: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    # 完全に評価モードへ
    model.eval()
    
    # No-Preamble フォーマットでのプロンプト構築
    messages = [{"role": "user", "content": test_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    print("\n🧠 推論開始...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.01,   # 決定論的に
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n" + "="*50)
    print("【ユーザー指示】")
    print(test_prompt[:100] + "...")
    print("\n【モデルの回答 (V9)】")
    print("-" * 50)
    print(response)
    print("-" * 50)
    
    # 検査
    if response.startswith("```") or response.strip().lower().startswith("here") or response.strip().lower().startswith("sure"):
        print("❌ 警告: チャッティな出力（Markdownや挨拶）が含まれています！")
    elif response.startswith("{") or response.startswith("[") or response.startswith("<"):
        print("✅ 成功: 純粋なデータ（No-Preamble）で出力されました！")
    else:
        print("⚠️ 判定不能: 予期せぬ文字列から始まっています。")
    print("="*50)

if __name__ == "__main__":
    # ハードルが高い「CSVからJSONへの変換」テスト（わざと丁寧な回答を誘発しやすいお題）
    test_1 = """Convert this CSV into JSON format.

id,name,role
1,Alice,Admin
2,Bob,User"""
    
    test_inference(test_1)
