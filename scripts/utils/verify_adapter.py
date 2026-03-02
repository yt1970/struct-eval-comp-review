import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# --- 設定 ---
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
# 学習が終わったら、最新のフォルダ名をここにセットしてください
ADAPTER_PATH = "" 
TEST_DATA_PATH = "data/golden_test.jsonl"
OUTPUT_FILE = "test_inference_results.jsonl"

def main():
    if not ADAPTER_PATH:
        print("❌ ADAPTER_PATH をセットしてください。 (例: 'adapters/adapter_20260204_1955' または 'outputs/train_20260204_1955/checkpoint-xxx')")
        return

    print(f"🚀 モデルをロード中: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": "mps"}
    )
    
    print(f"💉 アダプタを適用中: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    print(f"📖 テストデータを読み込み中: {TEST_DATA_PATH}")
    test_data = []
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line))

    # 時間短縮のため、まずは最初の10件を推論
    # 全件やる場合は test_data に戻してください
    target_data = test_data[:10] 

    print(f"🧠 推論開始 ({len(target_data)}件)...")
    results = []
    
    for item in tqdm(target_data):
        messages = item["messages"]
        # アシスタントの回答を空にする
        user_messages = [m for m in messages if m["role"] != "assistant"]
        
        input_ids = tokenizer.apply_chat_template(
            user_messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to("mps")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False, # 再現性のため
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        
        results.append({
            "instruction": user_messages[-1]["content"],
            "expected": [m["content"] for m in messages if m["role"] == "assistant"][0],
            "generated": response
        })

    print(f"💾 結果を保存中: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("✅ 推論完了！結果を test_inference_results.jsonl で確認してください。")

if __name__ == "__main__":
    main()
