
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# --- Configuration ---
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
CHECKPOINT_PATH = "outputs/train_sft_v3_20260211_Silence_v3/checkpoint-100"
DATA_PATH = "data/public_150.json"
OUTPUT_FILE = "inference_sft_v3_early_100.json"

def main():
    print(f"🏃 SFT v3 Early Checkpoint (Step 100) Verification (LIMITED SAMPLE)")
    print(f"   Model: {CHECKPOINT_PATH}")
    print(f"   Data: {DATA_PATH}")

    with open(DATA_PATH, 'r') as f:
        tasks = json.load(f)
    
    # LIMIT to 5
    tasks = tasks[:5]
    print(f"   Processing ONLY {len(tasks)} tasks.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("   Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"   Loading Adapter from {CHECKPOINT_PATH}...")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()

    results = []
    print("   Starting Inference...")
    
    for i, task in enumerate(tqdm(tasks)):
        task_id = task.get("task_id")
        query = task.get("query")
        
        prompt_text = f"### 指示\n{query}\n\n### 応答\n"
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### 応答" in generated_text:
            response = generated_text.split("### 応答")[-1].strip()
        else:
            response = generated_text.strip()
            
        results.append({
            "task_id": task_id,
            "generation": response
        })
        
        # DEBUG PRINT
        print(f"\n--- Task {i+1} Output (ID: {task_id}) ---")
        print(response[:500] + "..." if len(response) > 500 else response)
        print("-" * 30)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
