import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

# --- Settings ---
# TIMESTAMP = "20260209_0709" # Base SFT timestamp (This is implicit in the adapter path structure usually, but here we just need the adapter path)
# But wait, adapter path is constructed below.
# OLD: ADAPTER_PATH = f"adapters/adapter_legacy_silence_{TIMESTAMP_DPO}"

# NEW: Point directly to the v3 folder
ADAPTER_PATH = "adapters/adapter_legacy_silence_20260210_Silence_v3_Aggressive"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
PUBLIC_DATA_PATH = "data/public_150.json"

# Tasks known to be problematic (asking for CSV->JSON conversion etc)
TARGET_TASK_IDS = [
    142, # "Extract entities ... output JSON" (Often chats)
    145, # "Convert the following CSV ..." (Classis chatter)
    148, # "Please reformat ..."
]
# Note: public_150 might use integer IDs or string IDs. Let's check logic below. 
# The previous code used "p_..." string IDs. 
# Let's inspect inference_SFT_0.json to see real IDs.
# Actually, let's just pick the first few tasks from public_150 to test.
# Or use the same IDs as v2 script if they are valid.
# public_150.json usually has "task_id": 1, 2, ...
# The v2 script had "p_bb59..." which looks like the competition IDs, not public_150 IDs?
# Ah, verify_silence_v2.py lines 14-19 had specific IDs.
# Let's use a simpler approach: Run on FIRST 5 tasks of public_150 to see general behavior.

def main():
    print(f"🤫 loading Silence Adapter v3 (Aggressive): {ADAPTER_PATH} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading adapter from {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    # Load tasks
    with open(PUBLIC_DATA_PATH, "r") as f:
        public_data = json.load(f)
        
    # Just take first 3 tasks for quick check
    tasks = public_data[:3]
    
    print("-" * 50)
    print(f"🚀 Running inference on {len(tasks)} tasks (Streaming mode)...")
    print("-" * 50)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    for i, task in enumerate(tasks):
        task_id = task.get("task_id", i)
        # Use simple key lookup based on previous success
        query = task.get("query", task.get("input", "SAMPLE QUERY MISSING"))

        print(f"\n\n[Task ID: {task_id}] Length: {len(query)}")
        print(f"Query (head): {query[:100]}...")
        
        # Prompt format must match training (SFT/DPO)
        # SFT used: "### 指示\n{input}\n\n### 応答\n"
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
        
        # Simple analysis
        full_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_out.split("### 応答\n")[-1].strip()
        
        # Check for chatty words
        chat_markers = ["Sure", "Here is", "Certainly", "I can", "Below"]
        if any(m in response for m in chat_markers):
            print("❌ Still Chatty! (Found marker)")
        elif response.startswith("```json"):
             print("⚠️ Markdown detected (Still outputting ```json?)")
        elif response.startswith("{") or response.startswith("["):
             print("✅ Looks like pure JSON/Structure!")
        else:
             print("❓ Unknown format start")

if __name__ == "__main__":
    main()
