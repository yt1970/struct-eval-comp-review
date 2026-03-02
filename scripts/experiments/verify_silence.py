import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

# --- Settings ---
TIMESTAMP = "20260209_0709" # Base SFT timestamp
TIMESTAMP_DPO = "20260209_Silence" # DPO adapter timestamp

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = f"adapters/adapter_legacy_silence_{TIMESTAMP_DPO}"
PUBLIC_DATA_PATH = "data/public_150.json"

TARGET_TASK_IDS = [
    "p_bb594bd2d86606dbd1d1823d", # Text to JSON (Previously missing keys)
    "p_bb8bcd0930ae9f9e4f0692ae", # CSV to JSON
    "p_b3fcb16b0778d50065908799", # CSV to JSON (Complex)
    "p_0cc0d1708fce08c15c260262", # CSV to JSON (Complex 2)
]

def main():
    print(f"🤫 loading Silence Adapter: {ADAPTER_PATH} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    
    # Load tasks
    with open(PUBLIC_DATA_PATH, "r") as f:
        public_data = json.load(f)
        
    tasks = [t for t in public_data if t["task_id"] in TARGET_TASK_IDS]
    
    print("-" * 50)
    print(f"🚀 Running inference on {len(tasks)} selected tasks (with Streaming)...")
    print("-" * 50)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    for task in tasks:
        task_id = task["task_id"]
        query = task["query"]
        task_name = task.get("task_name", "Unknown")
        
        print(f"\n[Task: {task_name}] (ID: {task_id})")
        print(f"Query (first 100 chars): {query[:100]}...")
        
        prompt = f"### 指示\n{query}\n\n### 応答\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print("\n--- Model Output (Start) ---")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.0, # Deterministic
                do_sample=False,
                start_token_id=tokenizer.bos_token_id, 
                pad_token_id=tokenizer.pad_token_id,
                streamer=streamer # Streaming output
            )
            
        print("\n--- Model Output (End) ---")
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### 応答\n" in generated_text:
            response = generated_text.split("### 応答\n")[-1].strip()
        else:
            response = generated_text.strip()
            
        # Check for Silence
        conversational_markers = ["Sure!", "Here is", "Certainly", "I have converted", "Below is"]
        markdown_markers = ["```json", "```"]
        
        has_chat = any(marker in response for marker in conversational_markers)
        has_md = any(marker in response for marker in markdown_markers)
        
        if not has_chat and not has_md and (response.startswith("{") or response.startswith("[")):
             print("✅ PERFECT FORMAT: Pure JSON detected!")
        else:
             print("⚠️ FORMAT WARNING: ")
             if has_chat: print("  - Conversational filler detected")
             if has_md: print("  - Markdown usage detected")
             if not (response.startswith("{") or response.startswith("[")): print("  - Doesn't start with JSON brace")

if __name__ == "__main__":
    main()
