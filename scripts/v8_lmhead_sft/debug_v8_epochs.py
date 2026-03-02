"""
SFT v8 Epoch別 3問デバッグテスト
- Epoch 1, 2, 3 のチェックポイントを順番にロードして3問テスト
- Markdown残存、chattyフレーズ、EOS停止を確認
- メモリ節約のため、各Epochごとにモデルを解放
"""
import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH = "adapters/adapter_legacy_sft_20260209_0709"
ADAPTER_BASE = "adapters/adapter_sft_v8_lmhead"
DATA_PATH = "data/public_150.json"
NUM_TASKS = 3

# 外部ジェミサンの指摘を反映：資料(Context)を先に、指示(Query)を後にする構成
# CSVタスクの本文も切り出して完全な抽出タスクとして定義
SOURCE_DATA_MAP = {
    "p_7b3394e21698627665533715": "Source Text:\nThe 'Aetheria' ecosystem (Latitude 45.0, Longitude -120.5) has a Tropical climate with an average temperature of 28.5 Celsius. The dominant species is the 'Lumina Fern'. Plant species include 'Flora Magica' (Common name: Glow Leaf) with a population of 1500. Animal species include 'Panthera Caelum' (Endangered, 15 years, Carnivore). Environmental threats include 'Habitat Loss' with High severity.",
    "p_bb594bd2d86606dbd1d1823d": "Source Text:\nThe 'Aethelgard Shield' was created by the Valorian civilization during the Silver Era. It is made of Mithril (provenance: Northern Peaks) and Gold (provenance: Sun Mines). Dimensions: Height 120cm, Width 80cm, Depth 5cm. Two inscriptions: 'Ancient High Valorian' translated as 'Protector of Kings' (outer rim); 'Runes' translated as 'Never Shall I Break' (center). Location: Global Archeological Museum, Display Case 42. Discovered: 2024-05-12 by Dr. Jones Expedition Team.",
    "p_bb8bcd0930ae9f9e4f0692ae": "Source Text (CSV):\nname,model,captain,pilot,engineer,length_m,mass_kg,propulsion_type,propulsion_thrust_kN,mission_1_name,mission_1_destination,mission_1_year,mission_1_outcome,mission_2_name,mission_2_destination,mission_2_year,mission_2_outcome\nStar Voyager,XJ-9,Amelia Hawk,Leo Tran,Samira Voss,120,45000,Antimatter,1500,Alpha Pioneer,Proxima Centauri,2087,Success,Stellar Rescue,Barnard's Star,2091,Partial Success"
}

def run_test(epoch):
    print(f"\n{'='*60}")
    print(f"  Epoch {epoch} テスト")
    print(f"{'='*60}")
    
    adapter_path = f"{ADAPTER_BASE}_epoch{epoch}"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model + SFT v2 merge
    print(f"  📦 Loading base model + SFT v2...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()
    
    # Apply v8 adapter
    print(f"  📦 Loading v8 epoch{epoch} adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    # Load tasks
    with open(DATA_PATH, 'r') as f:
        tasks = json.load(f)
    
    results = {"markdown": 0, "chatty": 0, "eos_stop": 0}
    
    for i, task in enumerate(tasks[:NUM_TASKS]):
        task_id = task.get("task_id") or task.get("id") or f"task_{i}"
        instruction = task.get("query") or task.get("instruction", "")
        
        # 構成の最適化：資料(Context)を先に、指示(Query)を後に配置
        context = SOURCE_DATA_MAP.get(task_id, "")
        
        # クエリ内の余計なCSVチップを整理（Source Textとして分離したため）
        if "<code>" in instruction:
            instruction = instruction.split("<code>")[0].strip()
            
        full_content = f"{context}\n\nTask:\n{instruction}"
        
        messages = [{"role": "user", "content": full_content}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
                temperature=1.0,
            )
        
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Check markers
        has_markdown = "```" in generated
        chatty_phrases = ["Sure!", "Here is", "Certainly", "I can help", "Below is", "I hope this helps", "Let me know"]
        has_chatty = any(p.lower() in generated.lower() for p in chatty_phrases)
        
        full_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        has_eos = full_output.strip().endswith(tokenizer.eos_token) or len(outputs[0]) - inputs["input_ids"].shape[1] < 500
        
        if has_markdown:
            results["markdown"] += 1
        if has_chatty:
            results["chatty"] += 1
        if has_eos:
            results["eos_stop"] += 1
        
        print(f"\n  --- Task {i+1} ({task_id}) ---")
        print(f"  Markdown: {'❌ あり' if has_markdown else '✅ なし'}")
        print(f"  Chatty:   {'❌ あり' if has_chatty else '✅ なし'}")
        print(f"  EOS停止:  {'✅' if has_eos else '❌ 到達せず'}")
        print(f"  生成長:   {len(generated)} 文字")
        print(f"  出力先頭200文字:")
        print(f"  {generated[:200]}")
        print(f"  ---")

    print(f"\n  📊 Epoch {epoch} 結果サマリー:")
    print(f"     Markdown残存: {results['markdown']}/{NUM_TASKS}")
    print(f"     Chatty残存:   {results['chatty']}/{NUM_TASKS}")
    print(f"     EOS停止:      {results['eos_stop']}/{NUM_TASKS}")
    
    # Cleanup
    del model, base_model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return results

def main():
    print("🧪 SFT v8 Epoch別 3問デバッグテスト開始!")
    print("   各Epochのチェックポイントで3問テストして比較します")
    
    all_results = {}
    for epoch in [1, 2, 3]:
        all_results[epoch] = run_test(epoch)
    
    # Final comparison
    print(f"\n{'='*60}")
    print(f"  最終比較")
    print(f"{'='*60}")
    print(f"  {'指標':<15} {'Epoch1':>8} {'Epoch2':>8} {'Epoch3':>8}")
    print(f"  {'-'*45}")
    print(f"  {'Markdown残存':<13} {all_results[1]['markdown']:>6}/3  {all_results[2]['markdown']:>6}/3  {all_results[3]['markdown']:>6}/3")
    print(f"  {'Chatty残存':<14} {all_results[1]['chatty']:>6}/3  {all_results[2]['chatty']:>6}/3  {all_results[3]['chatty']:>6}/3")
    print(f"  {'EOS停止':<15} {all_results[1]['eos_stop']:>6}/3  {all_results[2]['eos_stop']:>6}/3  {all_results[3]['eos_stop']:>6}/3")
    
    # Recommend best epoch
    scores = {}
    for epoch in [1, 2, 3]:
        r = all_results[epoch]
        # Lower markdown/chatty is better, higher eos_stop is better
        scores[epoch] = (NUM_TASKS - r['markdown']) + (NUM_TASKS - r['chatty']) + r['eos_stop']
    
    best = max(scores, key=scores.get)
    print(f"\n  🏆 推奨 Epoch: {best} (スコア: {scores[best]})")

if __name__ == "__main__":
    main()
