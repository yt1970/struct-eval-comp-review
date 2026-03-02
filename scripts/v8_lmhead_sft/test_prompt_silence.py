import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 最強の DPO がかかっているはずの Step 30 でテスト
CHECKPOINT_PATH = "outputs/dpo_anti_chatty_V3_Surgical_Strike/checkpoint-30"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

def test_negative_prompt():
    print(f"📦 Loading model from {CHECKPOINT_PATH}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float32, device_map="mps"
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    csv_data = "name,model,length_m\nStar Voyager,XJ-9,120"
    
    # テストパターン
    prompts = [
        # A: 通常（今までと同じ）
        f"Please convert the following CSV code to JSON code.\n\n{csv_data}",
        # B: 強力な口止め
        f"Please convert the following CSV code to JSON code.\n\n{csv_data}\n\nIMPORTANT: Provide ONLY the raw JSON. DO NOT include any introductory text, concluding notes, or Markdown formatting (no ```json blocks). Start your response directly with '{{'."
    ]

    for i, p in enumerate(prompts):
        label = "STANDARD" if i == 0 else "STRICT (Silent)"
        print(f"\n--- TESTING PROMPT: {label} ---")
        
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"RESPONSE:\n{response}")
        print("-" * 30)

if __name__ == "__main__":
    test_negative_prompt()
