"""
v8.3 Legacy Prompt Test
- v8.3 (rank=64) epoch 3 をレガシープロンプト形式でテスト
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "adapters/adapter_legacy_sft_20260209_0709"
V8_3_ADAPTER = "adapters/adapter_sft_v8_3_lmhead_lora_epoch3"
DATA_PATH = "data/public_150.json"

SOURCE_DATA_MAP = {
    "p_7b3394e21698627665533715": "Source Text:\nThe 'Aetheria' ecosystem (Latitude 45.0, Longitude -120.5) has a Tropical climate with an average temperature of 28.5 Celsius. The dominant species is the 'Lumina Fern'. Plant species include 'Flora Magica' (Common name: Glow Leaf) with a population of 1500. Animal species include 'Panthera Caelum' (Endangered, 15 years, Carnivore). Environmental threats include 'Habitat Loss' with High severity.",
    "p_bb594bd2d86606dbd1d1823d": "Source Text:\nThe 'Aethelgard Shield' was created by the Valorian civilization during the Silver Era. It is made of Mithril (provenance: Northern Peaks) and Gold (provenance: Sun Mines). Dimensions: Height 120cm, Width 80cm, Depth 5cm. Two inscriptions: 'Ancient High Valorian' translated as 'Protector of Kings' (outer rim); 'Runes' translated as 'Never Shall I Break' (center). Location: Global Archeological Museum, Display Case 42. Discovered: 2024-05-12 by Dr. Jones Expedition Team.",
    "p_bb8bcd0930ae9f9e4f0692ae": "Source Text (CSV):\nname,model,captain,pilot,engineer,length_m,mass_kg,propulsion_type,propulsion_thrust_kN,mission_1_name,mission_1_destination,mission_1_year,mission_1_outcome,mission_2_name,mission_2_destination,mission_2_year,mission_2_outcome\nStar Voyager,XJ-9,Amelia Hawk,Leo Tran,Samira Voss,120,45000,Antimatter,1500,Alpha Pioneer,Proxima Centauri,2087,Success,Stellar Rescue,Barnard's Star,2091,Partial Success"
}

def main():
    print("🔥 v8.3 Legacy Prompt Test (### 指示 形式)")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load Model (安定したロード手順)
    print("  📦 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    print("  📦 Applying SFT v2 adapter...")
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()
    
    print(f"  📦 Applying v8.3 epoch3 adapter: {V8_3_ADAPTER}")
    model = PeftModel.from_pretrained(model, V8_3_ADAPTER)
    model.eval()

    with open(DATA_PATH, 'r') as f:
        tasks = json.load(f)
    
    target_ids = ["p_7b3394e21698627665533715", "p_bb594bd2d86606dbd1d1823d", "p_bb8bcd0930ae9f9e4f0692ae"]
    test_tasks = [t for t in tasks if (t.get('task_id') or t.get('id')) in target_ids]

    for i, task in enumerate(test_tasks):
        task_id = task.get('task_id') or task.get('id')
        context = SOURCE_DATA_MAP.get(task_id, "")
        query = task.get('query') or task.get('instruction', "")
        
        # 指示文のクリーンアップ: 重複する "Task:" や不要な導入文を削除
        if "Task:\n" in query:
            query = query.split("Task:\n")[-1].strip()
        if "Please output JSON code:" in query:
            query = query.replace("Please output JSON code:", "").strip()
        if "<code>" in query:
            query = query.split("<code>")[0].strip()

        # Legacy Prompt Format (SFT v2 の純正規格)
        # 資料を「### 指示」の直後に置き、その後に具体的な「抽出ルール」を続ける
        prompt = f"### 指示\n{context}\n\nTask:\n{query}\n\n### 応答\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1024, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        print(f"\n--- Task {i+1} ({task_id}) ---")
        print(f"Markdown: {'❌ あり' if '```' in response else '✅ なし'}")
        print(f"EOS停止:  {'✅' if full_response.endswith(tokenizer.eos_token) else '❌'}")
        print(f"Output:\n{response[:500]}...") # 長いので先頭のみ表示

if __name__ == "__main__":
    main()
