"""
SFT v8 出力詳細確認 - Epoch 2 で1問だけフル出力を確認
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "adapters/adapter_legacy_sft_20260209_0709"
ADAPTER_PATH = "adapters/adapter_sft_v8_lmhead_epoch2"
DATA_PATH = "data/public_150.json"

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    with open(DATA_PATH, 'r') as f:
        tasks = json.load(f)

    task = tasks[0]
    instruction = task.get("instruction", "")
    
    print("=== INSTRUCTION (先頭300文字) ===")
    print(instruction[:300])
    print(f"\n=== INSTRUCTION 長さ: {len(instruction)} 文字 ===\n")

    messages = [{"role": "user", "content": instruction}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("=== CHAT TEMPLATE (先頭500文字) ===")
    print(text[:500])
    print(f"\n=== TEMPLATE 長さ: {len(text)} 文字 ===\n")
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(f"=== INPUT TOKEN 数: {inputs['input_ids'].shape[1]} ===\n")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            temperature=1.0,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    print(f"=== 生成 TOKEN 数: {len(generated_ids)} ===")
    print(f"=== 生成テキスト (skip_special=False) ===")
    print(generated_text[:1000])
    print(f"\n...")
    print(generated_text[-200:])
    
    # 先頭10トークンを個別に確認
    print(f"\n=== 先頭20トークン ===")
    for i, tid in enumerate(generated_ids[:20]):
        token_str = tokenizer.decode([tid])
        print(f"  [{i:3d}] token_id={tid.item():6d} -> '{token_str}'")

if __name__ == "__main__":
    main()
