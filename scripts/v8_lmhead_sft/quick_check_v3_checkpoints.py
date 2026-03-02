import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# --- CONFIG ---
VERSION = "V3_Surgical_Strike"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_BASE_DIR = f"outputs/dpo_anti_chatty_{VERSION}"
CHECKPOINTS = ["checkpoint-10", "checkpoint-20", "checkpoint-30"]
TEST_DATA_PATH = "data/public_150_full_eval_v8_5.json"

# テストに使う「お喋り誘発」問題ID (V2検証ログで失敗していたもの)
TARGET_TASK_IDS = [
    "p_bb8bcd0930ae9f9e4f0692ae", # Star Voyager (CSV)
    "p_b3fcb16b0778d50065908799", # Aurelia (CSV)
    "p_067c0c165398faf5e1414ee8", # Kepler (CSV)
]

def load_data():
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [t for t in data if t["task_id"] in TARGET_TASK_IDS]

def check_model(checkpoint_name, tasks):
    print(f"\n🔍 Checking {checkpoint_name}...")
    adapter_path = f"{OUTPUT_BASE_DIR}/{checkpoint_name}"
    
    # Load Model
    # MPS + Memory usage improvement: load in float16
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="mps"
    )
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        print(f"❌ Failed to load adapter {checkpoint_name}: {e}")
        return []

    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    results = []
    
    for task in tasks:
        messages = [{"role": "user", "content": task["query"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1024, do_sample=False, temperature=0.0
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 簡易評価
        has_markdown = "```" in response
        is_chatty = len(response.split("\n")[0]) > 20 and not response.strip().startswith("{")
        
        # 合格条件: Markdownなし AND 挨拶なし
        is_safe = not has_markdown and not is_chatty
        status = "✅" if is_safe else "❌"
        
        print(f"  Task {task['task_id'][:8]}: {status} (MD: {has_markdown}, Chatty: {is_chatty})")
        # デバッグ用に冒頭を表示
        print(f"    Head: {response[:60].replace(chr(10), ' ')}...") 
        
        results.append({"checkpoint": checkpoint_name, "task": task["task_id"], "status": status, "response": response})
        
    del model
    del base_model
    torch.cuda.empty_cache() # MPSでも気休めに
    return results

def main():
    tasks = load_data()
    print(f"Loaded {len(tasks)} target tasks for quick check.")
    
    all_results = {}
    for cp in CHECKPOINTS:
        res = check_model(cp, tasks)
        all_results[cp] = res

    print("\n" + "="*50)
    print("🏆 FINAL VERDICT")
    print("="*50)
    
    best_cp = None
    
    for cp in CHECKPOINTS:
        if not all_results[cp]:
            print(f"{cp}: Failed to run")
            continue
            
        score = sum(1 for r in all_results[cp] if r["status"] == "✅")
        print(f"{cp}: {score}/{len(tasks)} passed")
        
        if score == len(tasks) and best_cp is None:
            best_cp = cp # 最初に全問正解したものを採用候補とする
            
    if best_cp:
        print(f"\n✨ Recommended: {best_cp}")
    else:
        print("\n⚠️ No perfect candidate found among checkpoints.")

if __name__ == "__main__":
    main()
