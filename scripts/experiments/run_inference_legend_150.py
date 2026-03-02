import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# --- 設定 ---
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "./adapters/adapter_sft_legend_20260207_1929" 
EVAL_DATA_PATH = "./data/public_150.json"
OUTPUT_FILE = "inference_legend_150.json"

def main():
    print(f"📥 評価データ読み込み中: {EVAL_DATA_PATH}")
    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"🔄 ベースモデル読み込み中: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": "mps"}
    )

    print(f"🔗 伝説アダプタ適用中: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    results = []
    print(f"🚀 全 {len(eval_data)} 件の推論を開始します...")

    for item in tqdm(eval_data):
        query = item.get("query", "")
        task_id = item.get("task_id", "")
        
        # 伝説のプロンプト形式を再現
        prompt = f"### 指示\n{query}\n\n### 応答\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("mps")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # 保存形式を合わせる
        results.append({
            "task_id": task_id,
            "generation": response.strip()
        })

    # JSON保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 推論完了！結果を {OUTPUT_FILE} に保存しました。")

if __name__ == "__main__":
    main()
