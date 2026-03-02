import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# V3.1 (Deep Silence) の全チェックポイントを順に確認
CHECKPOINTS = [
    "outputs/dpo_anti_chatty_V3_1_Deep_Silence/checkpoint-20",
    "outputs/dpo_anti_chatty_V3_1_Deep_Silence/checkpoint-40",
    "outputs/dpo_anti_chatty_V3_1_Deep_Silence/checkpoint-60",
    "outputs/dpo_anti_chatty_V3_1_Deep_Silence/checkpoint-80",
    "outputs/dpo_anti_chatty_V3_1_Deep_Silence/checkpoint-100"
]
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

def debug_inference():
    for cp_path in CHECKPOINTS:
        print("\n" + "="*60)
        print(f"📦 Testing: {cp_path.split('/')[-1]}")
        print("="*60)
        
        # Load Model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.float32, device_map="mps"
        )
        model = PeftModel.from_pretrained(base_model, cp_path)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        
        # あえて「口止めなし」の標準プロンプトで試す
        query = "Please convert the following CSV code to JSON code.\n\nname,model,length_m\nStar Voyager,XJ-9,120"

        messages = [{"role": "user", "content": query}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        print(f"🔥 Generating...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"\n[RESPONSE]\n{response}")
        
        # Cleanup
        del model
        del base_model
        torch.mps.empty_cache()

if __name__ == "__main__":
    debug_inference()
