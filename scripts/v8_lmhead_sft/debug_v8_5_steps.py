"""
v8.3 でハルシネーションや形式エラーが起きた典型的な問題を使い、
v8.5 の各ステップ (50, 100, Final) を比較検証する。
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTERS = {
    "v8.5_step50": "adapters/adapter_sft_v8_5_anti_hallucination_step50",
    "v8.5_step100": "adapters/adapter_sft_v8_5_anti_hallucination_step100",
    "v8.5_final": "adapters/adapter_sft_v8_5_anti_hallucination_final",
}

# ハルシネーションを誘発しやすいテストケース（v8.3 で失敗したパターンに近いもの）
TEST_CASES = [
    {
        "name": "Hallucination Test (Luminara context)",
        "query": "以下の資料に基づき、地域の名前を抽出してJSONで答えて。資料：『エメラルド・バレーは美しい場所です。』",
        # v8.3 は「ルミナラ・レインフォレスト」と答えてしまった
    },
    {
        "name": "Markdown/Silence Test",
        "query": "以下のCSVをJSONに変換して。説明は不要。CSV:\nid,name\n1,Alice\n2,Bob",
    }
]

def run_test(model, tokenizer, query):
    messages = [{"role": "user", "content": query}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return gen_text.strip()

def main():
    print("🚀 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for name, path in ADAPTERS.items():
        print(f"\n" + "="*50)
        print(f"📦 Testing Adapter: {name}")
        print("="*50)
        
        model = PeftModel.from_pretrained(base_model, path)
        
        for i, test in enumerate(TEST_CASES):
            print(f"\n--- Test {i+1}: {test['name']} ---")
            print(f"Query: {test['query']}")
            result = run_test(model, tokenizer, test['query'])
            print(f"Result:\n{result}")
            
            # Markdownチェック
            if "```" in result:
                print("⚠️  Markdown detected!")
            else:
                print("✅ Pure text/code output.")
                
        # アダプタをアンロードして次へ
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    main()
