import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

# --- Settings ---
# DPO v5 Verification Script
# Update adapter path after training completes (check outputs/dpo_legacy_silence_*)
# For now, use placeholder or latest timestamp logic?
# Let's search for the latest adapter in main block.

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
PUBLIC_DATA_PATH = "data/public_150.json"

def get_latest_adapter_path():
    import glob
    import os
    # Find directories starting with 'adapters/adapter_legacy_silence_20260211_'
    candidates = glob.glob("adapters/adapter_legacy_silence_20260211_*")
    if not candidates:
        print("⚠️ No DPO v5 adapter found in adapters/ directory.")
        return None
    # Sort by name (timestamp) descending
    latest = sorted(candidates)[-1]
    return latest

def main():
    adapter_path = get_latest_adapter_path()
    if not adapter_path:
        return

    print(f"🕵️‍♀️ Verifying DPO v5 Adapter: {adapter_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Loading Adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # Load tasks
    with open(PUBLIC_DATA_PATH, "r") as f:
        public_data = json.load(f)
        
    # Select diverse tasks for verification
    # 1. XML task (potential markdown trap)
    # 2. JSON task (potential chatty trap)
    # 3. Simple task
    indices = [0, 10, 20, 30] 
    tasks = [public_data[i] for i in indices if i < len(public_data)]
    
    print("-" * 50)
    print(f"🚀 Running inference on {len(tasks)} tasks (Streaming mode)...")
    print("-" * 50)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    results = []

    for i, task in enumerate(tasks):
        task_id = task.get("task_id", i)
        query = task.get("query", task.get("input", "SAMPLE QUERY MISSING"))

        print(f"\n\n[Task ID: {task_id}] Length: {len(query)}")
        print(f"Query (head): {query[:100]}...")
        
        prompt = f"### 指示\n{query}\n\n### 応答\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print("\n--- Model Output (Start) ---")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512, 
                temperature=0.0, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, 
                streamer=streamer 
            )
            
        decoded_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        results.append({
            "task_id": task_id,
            "output": decoded_output
        })
        print("\n--- Model Output (End) ---")

    # Analysis
    print("\n\n📊 Verification Analysis:")
    for res in results:
        out = res["output"]
        has_markdown = "```" in out
        has_chatty = "Sure" in out or "Here is" in out or "Approach:" in out
        is_clean = not has_markdown and not has_chatty
        status = "✅ CLEAN" if is_clean else "❌ DIRTY"
        print(f"Task {res['task_id']}: {status} (Markdown: {has_markdown}, Chatty: {has_chatty})")
        # print(f"Output snippet: {out[:50]}...")

if __name__ == "__main__":
    main()
