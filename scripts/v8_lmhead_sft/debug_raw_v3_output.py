import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Step 20 と 30 を連投して確認
CHECKPOINTS = [
    "outputs/dpo_anti_chatty_V3_Surgical_Strike/checkpoint-20",
    "outputs/dpo_anti_chatty_V3_Surgical_Strike/checkpoint-30"
]
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

def debug_inference():
    for cp_path in CHECKPOINTS:
        print("\n" + "="*50)
        print(f"📦 Loading model from {cp_path}...")
        print("="*50)
        
        # Load Model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.float32, device_map="mps"
        )
        model = PeftModel.from_pretrained(base_model, cp_path)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        
        # テスト指示 (V2で失敗した CSV 変換)
        query = """Please convert the following CSV code to JSON code.\n\nname,model,captain,pilot,engineer,length_m,mass_kg,propulsion_type,propulsion_thrust_kN,mission_1_name,mission_1_destination,mission_1_year,mission_1_outcome,mission_2_name,mission_2_destination,mission_2_year,mission_2_outcome\nStar Voyager,XJ-9,Amelia Hawk,Leo Tran,Samira Voss,120,45000,Antimatter,1500,Alpha Pioneer,Proxima Centauri,2087,Success,Stellar Rescue,Barnard's Star,2091,Partial Success"""

        messages = [{"role": "user", "content": query}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        print(f"\n🔥 Generating with {cp_path.split('/')[-1]}...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        raw_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        
        print(f"\n--- {cp_path.split('/')[-1]} RAW RESPONSE START ---")
        print(raw_response)
        print(f"--- {cp_path.split('/')[-1]} RAW RESPONSE END ---\n")

        # Cleanup for next checkpoint
        del model
        del base_model
        torch.mps.empty_cache()

if __name__ == "__main__":
    debug_inference()
