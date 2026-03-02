"""
SFT v8.3 Epoch別デバッグテスト
- Epoch 1, 2, 3 のアダプタをロードして、Markdown除去・EOS停止・品質を確認
"""
import os
import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# === Config ===
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "adapters/adapter_legacy_sft_20260209_0709"
V8_ADAPTER_BASE = "adapters/adapter_sft_v8_3_lmhead_lora"
TEST_DATA_PATH = "data/public_150.json"

# テストするタスクID (Markdownが出やすかったものを抜粋)
TARGET_TASK_IDS = [
    "p_7b3394e21698627665533715",  # Text to JSON (Sample 0)
    "p_bb594bd2d86606dbd1d1823d",  # Text to JSON (Sample 1)
    "p_bb8bcd0930ae9f9e4f0692ae"   # CSV to JSON (Sample 2)
]

# 外部ジェミサンの指摘を反映：資料(Context)を先に、指示(Query)を後にする構成
# CSVタスクの本文も切り出して完全な抽出タスクとして定義
SOURCE_DATA_MAP = {
    "p_7b3394e21698627665533715": "Source Text:\nThe 'Aetheria' ecosystem (Latitude 45.0, Longitude -120.5) has a Tropical climate with an average temperature of 28.5 Celsius. The dominant species is the 'Lumina Fern'. Plant species include 'Flora Magica' (Common name: Glow Leaf) with a population of 1500. Animal species include 'Panthera Caelum' (Endangered, 15 years, Carnivore). Environmental threats include 'Habitat Loss' with High severity.",
    "p_bb594bd2d86606dbd1d1823d": "Source Text:\nThe 'Aethelgard Shield' was created by the Valorian civilization during the Silver Era. It is made of Mithril (provenance: Northern Peaks) and Gold (provenance: Sun Mines). Dimensions: Height 120cm, Width 80cm, Depth 5cm. Two inscriptions: 'Ancient High Valorian' translated as 'Protector of Kings' (outer rim); 'Runes' translated as 'Never Shall I Break' (center). Location: Global Archeological Museum, Display Case 42. Discovered: 2024-05-12 by Dr. Jones Expedition Team.",
    "p_bb8bcd0930ae9f9e4f0692ae": "Source Text (CSV):\nname,model,captain,pilot,engineer,length_m,mass_kg,propulsion_type,propulsion_thrust_kN,mission_1_name,mission_1_destination,mission_1_year,mission_1_outcome,mission_2_name,mission_2_destination,mission_2_year,mission_2_outcome\nStar Voyager,XJ-9,Amelia Hawk,Leo Tran,Samira Voss,120,45000,Antimatter,1500,Alpha Pioneer,Proxima Centauri,2087,Success,Stellar Rescue,Barnard's Star,2091,Partial Success"
}

def load_test_tasks():
    with open(TEST_DATA_PATH, 'r') as f:
        data = json.load(f)
    return [t for t in data if (t.get('task_id') or t.get('id')) in TARGET_TASK_IDS]

def test_checkpoint(epoch_num):
    adapter_path = f"{V8_ADAPTER_BASE}_epoch{epoch_num}"
    print(f"\n============================================================")
    print(f"  Epoch {epoch_num} テスト")
    print(f"============================================================")
    
    if not os.path.exists(adapter_path):
        print(f"⚠️  Adapter not found: {adapter_path}")
        return

    print("  📦 Loading base model + SFT v2...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    # まず SFT v2 を適用してマージ
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()
    
    print(f"  📦 Loading v8.3 epoch{epoch_num} adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tasks = load_test_tasks()

    metrics = {"markdown": 0, "chatty": 0, "eos": 0}

    for i, task in enumerate(tasks):
        task_id = task.get('task_id') or task.get('id')
        print(f"\n  --- Task {i+1} ({task_id}) ---")
        # 構成の最適化：資料(Context)を先に、指示(Query)を後に配置
        context = SOURCE_DATA_MAP.get(task_id, "")
        query = task.get('query') or task.get('instruction', "")
        
        # クエリ内の余計なCSVチップを整理（Source Textとして分離したため）
        if "<code>" in query:
            query = query.split("<code>")[0].strip()
            
        full_content = f"{context}\n\nTask:\n{query}"
        
        messages = [{"role": "user", "content": full_content}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # 応答部分のみ抽出
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        has_markdown = "```" in response
        has_chatty = any(phrase in response.lower()[:50] for phrase in ["sure", "here", "certainly", "ok"])
        ends_with_eos = full_output.endswith(tokenizer.eos_token)

        if has_markdown: metrics["markdown"] += 1
        if has_chatty: metrics["chatty"] += 1
        if ends_with_eos: metrics["eos"] += 1

        print(f"  Markdown: {'❌ あり' if has_markdown else '✅ なし'}")
        print(f"  Chatty:   {'❌ あり' if has_chatty else '✅ なし'}")
        print(f"  EOS停止:  {'✅' if ends_with_eos else '❌'}")
        print(f"  生成長:   {len(response)} 文字")
        print(f"  出力先頭300文字:\n  {response[:300].replace(chr(10), '  ' + chr(10))}")

    print(f"\n  📊 Epoch {epoch_num} 結果サマリー:")
    print(f"     Markdown残存: {metrics['markdown']}/3")
    print(f"     Chatty残存:   {metrics['chatty']}/3")
    print(f"     EOS停止:      {metrics['eos']}/3")

    # メモリ解放
    del model
    del base_model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if hasattr(torch, 'mps'): torch.mps.empty_cache()

def main():
    print(f"🧪 SFT v8.3 Epoch別 3問デバッグテスト (rank=64)")
    for epoch in [1, 2, 3]:
        test_checkpoint(epoch)

if __name__ == "__main__":
    main()
