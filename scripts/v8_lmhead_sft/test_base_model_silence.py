import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# SFTもDPOもしていない「素」のモデル
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

def test_base_model():
    print(f"📦 Loading RAW BASE model: {BASE_MODEL_ID}...")
    
    # 素のモデルをロード (アダプタなし)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float32, device_map="mps"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    csv_data = "name,model,length_m\nStar Voyager,XJ-9,120"
    
    # 究極の口止めプロンプト (DPOモデルが黙ったのと同じもの)
    prompt = f"Please convert the following CSV code to JSON code.\n\n{csv_data}\n\nIMPORTANT: Provide ONLY the raw JSON. DO NOT include any introductory text, concluding notes, or Markdown formatting (no ```json blocks). Start your response directly with '{{'."

    print(f"\n--- TESTING RAW BASE MODEL WITH STRICT PROMPT ---")
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"RESPONSE:\n{response}")
    print("-" * 30)

if __name__ == "__main__":
    test_base_model()
