import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 最も学習が進んだ checkpoint-100 を使用
CHECKPOINT_PATH = "outputs/dpo_anti_chatty_V3_1_Deep_Silence/checkpoint-100"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

def test_lora_scaling():
    print(f"📦 Loading model and adapter: {CHECKPOINT_PATH}")
    
    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float32, device_map="mps"
    )
    # Load LoRA Adapter
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH, adapter_name="silence_adapter")
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # 標準プロンプト
    query = "Please convert the following CSV code to JSON code.\n\nname,model,length_m\nStar Voyager,XJ-9,120"
    messages = [{"role": "user", "content": query}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    scales = [1.0, 2.0, 4.0, 8.0] # より極端な値までテスト
    
    # 元の scaling 値を保存しておく（一回だけ実行）
    base_scalings = {}
    for name, module in model.named_modules():
        if hasattr(module, "scaling"):
            base_scalings[name] = module.scaling

    for scale in scales:
        print("\n" + "="*60)
        print(f"🚀 Testing with LoRA Scale: {scale}")
        print("="*60)
        
        with torch.no_grad():
            # scaling を書き換え
            for name, module in model.named_modules():
                if name in base_scalings:
                    orig = base_scalings[name]
                    if isinstance(orig, dict):
                        # 辞書形式の場合
                        new_scaling = {k: v * scale for k, v in orig.items()}
                    else:
                        # 単一の float の場合
                        new_scaling = orig * scale
                    module.scaling = new_scaling
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"[RESPONSE (Scale {scale})]\n{response}")

if __name__ == "__main__":
    test_lora_scaling()
