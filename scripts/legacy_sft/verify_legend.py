import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

# --- Settings ---
# Legend Adapter Path
ADAPTER_PATH = "adapters/adapter_sft_legend_20260207_1929"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
PUBLIC_DATA_PATH = "data/public_150.json"

def main():
    print(f"🗡️ loading LEGEND SFT Adapter: {ADAPTER_PATH} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading Legend adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    # Load tasks
    with open(PUBLIC_DATA_PATH, "r") as f:
        public_data = json.load(f)
        
    tasks = public_data[:3] # First 3 tasks
    
    print("-" * 50)
    print(f"🚀 Running inference on {len(tasks)} tasks (Streaming mode)...")
    print("-" * 50)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)

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
                max_new_tokens=512, # Short generation for quick check
                temperature=0.0, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, 
                streamer=streamer 
            )
            
        print("\n--- Model Output (End) ---")

if __name__ == "__main__":
    main()
