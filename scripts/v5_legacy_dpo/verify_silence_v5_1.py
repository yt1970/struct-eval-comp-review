
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
import glob
import os

# --- Settings ---
# DPO v5.1 Verification Script (Clean Data Version)
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
PUBLIC_DATA_PATH = "data/public_150.json"

# Explicitly target the Clean V5.1 adapter
TARGET_ADAPTER_GLOB = "adapters/adapter_legacy_silence_20260211_Silence_v5_1_Clean*"

def get_adapter_path():
    candidates = glob.glob(TARGET_ADAPTER_GLOB)
    if not candidates:
        print(f"⚠️ No DPO v5.1 adapter found matching: {TARGET_ADAPTER_GLOB}")
        return None
    # If multiple (e.g. checkpoints), take the latest
    latest = sorted(candidates)[-1]
    return latest

def main():
    adapter_path = get_adapter_path()
    if not adapter_path:
        return

    print(f"🕵️‍♀️ Verifying DPO v5.1 Adapter (Clean): {adapter_path}")
    
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
    # 0, 10, 20 are arbitrary indices to inspect different items
    indices = [0, 5, 10, 15, 20] 
    tasks = [public_data[i] for i in indices if i < len(public_data)]
    
    print("-" * 50)
    print(f"🚀 Running inference on {len(tasks)} tasks...")
    print("-" * 50)
    
    # Use TextStreamer for real-time visibility (though script runs in bg usually)
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    results = []

    for i, task in enumerate(tasks):
        task_id = task.get("task_id", i)
        query = task.get("query", task.get("input", "SAMPLE QUERY MISSING"))

        print(f"\n\n[Task ID: {task_id}] Input Preview: {query[:50]}...")
        
        # Consistent prompt format
        prompt = f"### 指示\n{query}\n\n### 応答\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print("\n--- Model Output (Start) ---")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512, 
                temperature=0.0, # Greedy for reproducibility and silence check
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
    print("\n\n📊 Verification Analysis (v5.1 Clean):")
    clean_count = 0
    for res in results:
        out = res["output"]
        has_markdown = "```" in out
        has_chatty = "Sure" in out or "Here is" in out or "Approach:" in out or "Certainly" in out
        
        is_clean = not has_markdown and not has_chatty
        status = "✅ CLEAN" if is_clean else "❌ DIRTY"
        if is_clean:
            clean_count += 1
            
        print(f"Task {res['task_id']}: {status}")
        if has_markdown:
            print(f"  - Markdown detected: True")
        if has_chatty:
            print(f"  - Chatty detected: True")

    print("-" * 30)
    print(f"Clean Rate: {clean_count}/{len(tasks)} ({clean_count/len(tasks)*100:.1f}%)")
    
if __name__ == "__main__":
    main()
