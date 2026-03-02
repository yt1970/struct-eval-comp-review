
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# --- Configuration ---
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "adapters/adapter_dpo_minimal_v6"
DATA_PATH = "data/public_150.json"
OUTPUT_FILE = "inference_DPO_v6_minimal.json"

def main():
    print(f"🕵️ DPO v6 Minimal (Silence) Verification")
    print(f"   Adapter: {ADAPTER_PATH}")
    print(f"   Data: {DATA_PATH}")

    # 1. Load Data
    with open(DATA_PATH, 'r') as f:
        tasks = json.load(f)
    print(f"   Loaded {len(tasks)} tasks.")

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Base Model
    print("   Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 4. Load Adapter
    print(f"   Loading Adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # 5. Inference Loop
    results = []
    print("   Starting Inference...")
    
    for task in tqdm(tasks):
        task_id = task.get("task_id")
        query = task.get("query")
        
        # SFT/DPO format
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
        
        # Extract response part
        if "### 応答" in generated_text:
            response = generated_text.split("### 応答")[-1].strip()
        else:
            response = generated_text.strip()
            
        results.append({
            "task_id": task_id,
            "generation": response
        })

    # 6. Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
