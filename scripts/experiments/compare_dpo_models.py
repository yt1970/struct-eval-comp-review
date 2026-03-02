import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# 比較対象のアダプタ
ADAPTER_MAP = {
    "SFTなしDPO直行便 (今回の実験)": "adapters/adapter_dpo_direct_base",
    "SFTありDPO初期版 (Legend 150 + DPO)": "adapters/adapter_dpo_fix_10674"
}

test_data = [
    # TOML Loop Hell Case
    """[systems]
type = "cluster"
[[systems.nodes]]
name = "node-01"
role = "master"
[[systems.nodes.interfaces]]
eth0 = "192.168.1.10"
[[systems.nodes]]
name = "node-02"
role = "worker"
[[systems.nodes.interfaces]]
eth0 = "192.168.1.11"
eth1 = "10.0.0.2"
""",
    # Velox Hallucination Trigger Case
    """id, type, status
101, server, active
102, router, maintenance
"""
]

def main():
    print(f"🚀 モデル比較テスト開始！ ベース: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": "mps"}
    )
    
    for case_idx, data in enumerate(test_data):
        print(f"\n{'='*20} Test Case {case_idx+1} {'='*20}")
        print("【入力データ】")
        print(data.strip())
        print("-" * 50)
        
        prompt = f"### 指示\n{data}\n\n### 応答\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

        for name, adapter_path in ADAPTER_MAP.items():
            print(f"\n🔹 {name} の推論中...")
            try:
                # アダプタをロードして推論
                model = PeftModel.from_pretrained(base_model, adapter_path)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
                    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    print(response.strip())
            except Exception as e:
                print(f"Error: {e}")
            finally:
                # メモリ節約のため、アダプタを無効化（アンロード的な挙動を期待）
                 base_model.disable_adapters()
                 base_model.unmerge_adapter() # 念のため
    
    print("\n✅ 比較完了！")

if __name__ == "__main__":
    main()
