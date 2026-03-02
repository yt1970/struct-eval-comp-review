import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "adapters/adapter_legacy_sft_20260209_0709"
DPO_ADAPTER_PATH = "adapters/adapter_dpo_v7_1_surgical"
DATA_PATH = "data/public_150.json"

def main():
    print("DPO v7.1 Debug Test")
    
    with open(DATA_PATH, 'r') as f:
        tasks = json.load(f)
    tasks = tasks[:3]
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    print("   Loading base model + SFT adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()
    
    print("   Applying DPO v7.1 adapter...")
    model = PeftModel.from_pretrained(model, DPO_ADAPTER_PATH)
    model.eval()
    print("   Model ready")
    
    for i, task in enumerate(tasks):
        query = task.get("query")
        prompt_text = "### 指示\n" + query + "\n\n### 応答\n"
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        print("\n" + "=" * 60)
        print("--- Task " + str(i+1) + " ---")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "### 応答\n" in response:
            resp_part = response.split("### 応答\n")[-1]
        else:
            resp_part = response
        
        print("Response:")
        print(resp_part[:800])
        
        has_md = "```" in resp_part
        has_chat = any(x in resp_part.lower() for x in ["sure!", "here is", "certainly", "i hope", "let me know"])
        eos_token = "<" + "|im_end|" + ">"
        has_eos = eos_token in resp_part
        
        print("\n--- Checks ---")
        print("  Markdown: " + ("FOUND (bad)" if has_md else "CLEAN (good)"))
        print("  Chatty:   " + ("FOUND (bad)" if has_chat else "CLEAN (good)"))
        print("  EOS stop: " + ("YES (good)" if has_eos else "NO (bad)"))
        print("-" * 30)

if __name__ == "__main__":
    main()
