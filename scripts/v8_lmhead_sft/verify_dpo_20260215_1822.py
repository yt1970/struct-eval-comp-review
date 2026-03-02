"""
DPO Anti-Chatty Verification Script (150 tasks)
- 目的: DPO 矯正済みモデル (20260215_1822) の品質をスキャンする。
- チェック項目: Markdown混入、饒舌さ、EOS停止、ハルシネーション。
- 出力: data/verification_dpo_20260215_1822_results.jsonl
"""
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# === Config ===
TIMESTAMP = "20260215_1822"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = f"adapters/adapter_dpo_anti_chatty_{TIMESTAMP}"
INPUT_PATH = "data/public_150_full_eval_v8_5.json"
OUTPUT_LOG_PATH = f"data/verification_dpo_{TIMESTAMP}_results.jsonl"

# ハルシネーションしやすい旧キーワード
HALLUCINATION_WATCHLIST = ["Luminara", "Verdant", "Aetheria", "Aethelgard", "Zyphora"]

def main():
    print("=" * 60)
    print(f"🧐 Starting Quality Verification for DPO Model: {TIMESTAMP}")
    print(f"📄 Full details (Query, Format, Response) -> {OUTPUT_LOG_PATH}")
    print("=" * 60)
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: {INPUT_PATH} not found.")
        return

    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    # Load Model (Optimized for Mac)
    print(f"📦 Loading model {BASE_MODEL_ID} + adapter {ADAPTER_PATH}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    stats = {
        "total": len(tasks),
        "pure_code": 0,
        "markdown_error": 0,
        "chatty_error": 0,
        "hallucination_suspect": 0,
        "eos_failure": 0
    }

    print("\n🔍 Scanning 150 tasks...")
    failed_ids = []
    with open(OUTPUT_LOG_PATH, "w", encoding="utf-8") as out_f:
        for item in tqdm(tasks):
            task_id = item.get("task_id", "unknown")
            source = item.get("source_text", "")
            query_raw = item.get("query", "")
            output_type = item.get("output_type", "unknown")
            
            query = query_raw.replace("[DATA_BLOCK]", source)
            
            # 標準コードに合わせたプロンプト構築
            prompt_content = f"{source}\n\nTask:\n{query}" if (source and source not in query) else query
            messages = [{"role": "user", "content": prompt_content}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
            
            response = tokenizer.decode(outputs.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            full_tokens = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)

            # Evaluation
            errors = []
            if "```" in response:
                stats["markdown_error"] += 1
                errors.append("markdown_block")
            
            if any(w in response.lower()[:30] for w in ["sure", "ok", "here", "certainly"]):
                stats["chatty_error"] += 1
                errors.append("chatty_intro")
            
            hallu_words = []
            for word in HALLUCINATION_WATCHLIST:
                if word.lower() in response.lower() and word.lower() not in query.lower():
                    hallu_words.append(word)
            
            if hallu_words:
                stats["hallucination_suspect"] += 1
                errors.append(f"hallucination({','.join(hallu_words)})")
            
            if not full_tokens.endswith(tokenizer.eos_token):
                stats["eos_failure"] += 1
                errors.append("eos_missing")
                
            if errors:
                failed_ids.append(task_id)
            else:
                stats["pure_code"] += 1

            # Log to file
            log_entry = {
                "task_id": task_id,
                "task_name": item.get("task_name"),
                "output_type": output_type,
                "errors": errors,
                "response": response,
                "source_text": source,
                "query": query
            }
            out_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            out_f.flush()

    print("\n" + "="*50)
    print(f"📊 DPO Anti-Chatty ({TIMESTAMP}) 150問一括検証レポート")
    print("="*50)
    print(f"総計: {stats['total']} 問")
    print(f"✅ 全クリア (沈黙かつ正確): {stats['pure_code']} / 150")
    print(f"❌ Markdown混入: {stats['markdown_error']}")
    print(f"❌ 饒舌エラー:   {stats['chatty_error']}")
    print(f"⚠️  ハルシネーション疑い: {stats['hallucination_suspect']}")
    print(f"⚠️  EOS未停止: {stats['eos_failure']}")
    print("="*50)
    if failed_ids:
        print(f"❌ 失敗タスクID: {', '.join(failed_ids[:20])}{' ...' if len(failed_ids) > 20 else ''}")
    print(f"📝 Full logs saved to: {OUTPUT_LOG_PATH}")

if __name__ == "__main__":
    main()
