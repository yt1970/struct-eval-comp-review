"""
SFT v8.1 Epoch別 3問デバッグテスト
- v8.1 (lm_head LoRA rank=4) のチェックポイントを Epoch 1/2/3 順にテスト
- Markdown残存、chattyフレーズ、EOS停止を確認
"""
import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "adapters/adapter_legacy_sft_20260209_0709"
ADAPTER_BASE = "adapters/adapter_sft_v8_1_lmhead_lora"
DATA_PATH = "data/public_150.json"
NUM_TASKS = 3

def run_test(epoch):
    print(f"\n{'='*60}")
    print(f"  Epoch {epoch} テスト")
    print(f"{'='*60}")
    
    adapter_path = f"{ADAPTER_BASE}_epoch{epoch}"
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  📦 Loading base model + SFT v2...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()
    
    print(f"  📦 Loading v8.1 epoch{epoch} adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    with open(DATA_PATH, 'r') as f:
        tasks = json.load(f)
    
    results = {"markdown": 0, "chatty": 0, "eos_stop": 0}
    
    for i, task in enumerate(tasks[:NUM_TASKS]):
        task_id = task.get("task_id", f"task_{i}")
        instruction = task.get("query", "")
        
        messages = [{"role": "user", "content": instruction}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=500, do_sample=False, temperature=1.0,
            )
        
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        has_markdown = "```" in generated
        chatty_phrases = ["Sure!", "Here is", "Certainly", "I can help", "Below is", "I hope this helps", "Let me know"]
        has_chatty = any(p.lower() in generated.lower() for p in chatty_phrases)
        
        full_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        has_eos = full_output.strip().endswith(tokenizer.eos_token) or len(outputs[0]) - inputs["input_ids"].shape[1] < 500
        
        if has_markdown: results["markdown"] += 1
        if has_chatty: results["chatty"] += 1
        if has_eos: results["eos_stop"] += 1
        
        print(f"\n  --- Task {i+1} ({task_id}) ---")
        print(f"  Markdown: {'❌ あり' if has_markdown else '✅ なし'}")
        print(f"  Chatty:   {'❌ あり' if has_chatty else '✅ なし'}")
        print(f"  EOS停止:  {'✅' if has_eos else '❌ 到達せず'}")
        print(f"  生成長:   {len(generated)} 文字")
        print(f"  出力先頭300文字:")
        print(f"  {generated[:300]}")

    print(f"\n  📊 Epoch {epoch} 結果サマリー:")
    print(f"     Markdown残存: {results['markdown']}/{NUM_TASKS}")
    print(f"     Chatty残存:   {results['chatty']}/{NUM_TASKS}")
    print(f"     EOS停止:      {results['eos_stop']}/{NUM_TASKS}")
    
    del model, base_model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return results

def main():
    print("🧪 SFT v8.1 Epoch別 3問デバッグテスト")
    
    all_results = {}
    for epoch in [1, 2, 3]:
        all_results[epoch] = run_test(epoch)
    
    print(f"\n{'='*60}")
    print(f"  最終比較")
    print(f"{'='*60}")
    print(f"  {'指標':<15} {'Epoch1':>8} {'Epoch2':>8} {'Epoch3':>8}")
    print(f"  {'-'*45}")
    print(f"  {'Markdown残存':<13} {all_results[1]['markdown']:>6}/3  {all_results[2]['markdown']:>6}/3  {all_results[3]['markdown']:>6}/3")
    print(f"  {'Chatty残存':<14} {all_results[1]['chatty']:>6}/3  {all_results[2]['chatty']:>6}/3  {all_results[3]['chatty']:>6}/3")
    print(f"  {'EOS停止':<15} {all_results[1]['eos_stop']:>6}/3  {all_results[2]['eos_stop']:>6}/3  {all_results[3]['eos_stop']:>6}/3")

if __name__ == "__main__":
    main()
