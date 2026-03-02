import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "adapters/adapter_v11_surgical_dpo"

QUERY = """Convert the following information into a single row of CSV with headers.

INFORMATION:
The archaeology team discovered a gilded bronze incense burner (ArtifactID: A-987) at the site of the ancient city of Zorvex. Dating back to the 4th-century BCE, the artifact weighs 1,250 grams and was uncovered on 2023-01-20.

CSV HEADERS:
ArtifactID,DiscoverySite,Era,Material,WeightGrams,DiscoveryDate"""

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    # 1. ChatML
    print("\n--- Testing with ChatML ---")
    messages = [{"role": "user", "content": QUERY}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip())

    # 2. Legendary
    print("\n--- Testing with Legendary Format ---")
    prompt = f"### 指示\n{QUERY}\n\n### 応答\n"
    inputs = tokenizer(prompt, return_tensors="pt", return_dict=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip())

if __name__ == "__main__":
    main()
