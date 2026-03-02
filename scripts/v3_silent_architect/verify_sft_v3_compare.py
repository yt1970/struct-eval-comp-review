
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Constants
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_DIR = "outputs/train_sft_v3_20260211_Silence_v3"
# Checkpoints to test: Final (1126), and Mid-point (600, closest to 500 saved step)
CHECKPOINTS = ["checkpoint-1126", "checkpoint-600"] 

def main():
    print(f"🔇 Verifying SFT v3 Models (Final vs Mid-point)...")

    # 1. Load Tokenizer & Base Model (Shared)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    prompts = [
        # 1. Extraction (JSON)
        "Extract attributes: name, age. Text: User name is John Doe, 35 years old.",
        # 2. Silence Check (Chatty Input)
        "Hello! Please extract data. Text: Name: Sarah, Age: 28.",
        # 3. Catastrophic Forgetting (General Knowledge)
        "What is the capital of France?",
        # 4. Japanese Simple Question
        "日本の首都は？"
    ]

    for ckpt in CHECKPOINTS:
        adapter_path = os.path.join(OUTPUT_DIR, ckpt)
        print(f"\n\n{'='*20} Testing Adapter: {ckpt} {'='*20}")
        
        try:
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model.eval()
        except Exception as e:
            print(f"Skipping {ckpt}: {e}")
            continue

        print("\n--- Generation Tests ---")
        for i, prompt in enumerate(prompts):
            print(f"\n[Test {i+1}] Input: {prompt}")
            
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1, 
                    do_sample=True 
                )
                
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"Response:\n{response}")
            # Clean up adapter to load next one
            # model.unload() # PeftModel doesn't always unload cleanly on some versions, simpler to just re-wrap base_model or let it be replaced. 
            # Ideally we unmerge, but here we just load a new PeftModel on top of the ORIGINAL base_model.
            # However, PeftModel modifies base_model in place sometimes. 
            # Safer to reload base model? Or use `set_adapter` if we loaded all?
            # For simplicity in this script, we'll just reload the adapter on the base model object.
            
        # Cleanup for next loop (simple approach: unwrap)
        base_model = model.unload() 

if __name__ == "__main__":
    main()
