
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "adapters/adapter_dpo_minimal_v6"
DATA_PATH = "data/public_150.json"

def main():
    print(f"🕵️ Debugging DPO v6 Output")
    with open(DATA_PATH, 'r') as f:
        tasks = json.load(f)
    
    tasks = tasks[:3] # Just 3 tasks
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    for i, task in enumerate(tasks):
        query = task.get("query")
        prompt_text = f"### 指示\n{query}\n\n### 応答\n"
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        print(f"\n--- Task {i+1} ---")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200, # Limit to see if it starts looping
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"Full Text (with tokens):\n{response}")
        print("-" * 30)

if __name__ == "__main__":
    main()
