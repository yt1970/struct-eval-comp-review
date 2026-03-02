import json
import random
import os
import re
from transformers import AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SOURCE_DATA = "/Users/yutako/dev/struct-eval-comp/data/hf_datasets/dpo-dataset-qwen-cot.jsonl"
OUTPUT_DIR = "/Users/yutako/dev/struct-eval-comp/experiments/exp_20260301_v14_silent_dpo/data"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "v14_silent_dpo_train.jsonl")

# Target DPO dataset size (150 train, no eval)
TRAIN_SIZE = 150

def clean_structured_data(text):
    """
    Remove any Markdown blocks, preambles, and postambles to leave only pure structural output.
    """
    # Remove 'Approach: ... Output:\n' artifacts
    pattern = r"Approach:.*?Output:\s*\n?"
    if re.search(pattern, text, re.DOTALL):
        text = re.sub(pattern, "", text, flags=re.DOTALL)

    # Remove basic preambles
    text = re.sub(r'^(Sure!|Certainly|Here is|Below is|I have|As requested)[^\n]*\n+', '', text, flags=re.IGNORECASE)
    
    # Remove markdown codeblocks (```json, ```csv, etc) and their closing ```
    text = re.sub(r'```[a-zA-Z]*\n', '', text)
    text = re.sub(r'\n```$', '', text)
    text = re.sub(r'^```\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    
    # Remove Note from bottom
    if "\nNote:" in text:
        text = text.split("\nNote:")[0]

    return text.strip()

def create_rejected_pattern(structural_text, original_rejected):
    """
    Synthetically introduce chattiness that we want the model to UNLEARN.
    We can occasionally use the original rejected which has CoT, or wrap it in Markdown and chat.
    """
    patterns = [
        f"Sure! Here is the data in the requested format:\n\n```json\n{structural_text}\n```",
        f"```json\n{structural_text}\n```", # Pure markdown
        original_rejected.strip() # The original rejected text which often contains formatting errors or excessive chat
    ]
    return random.choice(patterns)

def get_user_prompt(raw_prompt):
    """
    Extract the bare user instruction from the raw prompt in the dataset.
    The dataset prompt looks like:
    <|im_start|>system\nYou are a helpful...<|im_end|>\n<|im_start|>user\nProduce a TOML...<|im_end|>\n<|im_start|>assistant\n
    """
    match = re.search(r'<\|im_start\|>user\n(.*?)\n?<\|im_end\|>', raw_prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_prompt.strip()

def main():
    print(f"🚀 V14 Structural Silence DPO Data Generator")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    if not os.path.exists(SOURCE_DATA):
        print(f"❌ Source data not found: {SOURCE_DATA}")
        return

    with open(SOURCE_DATA, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    
    # Filter for structural data requests (JSON, CSV, XML, TOML, YAML)
    struct_keywords = ["json", "csv", "xml", "toml", "yaml"]
    filtered_lines = [line for line in lines if any(kw in line.get("prompt", "").lower() for kw in struct_keywords)]
    
    print(f"Loaded {len(lines)} items, filtered down to {len(filtered_lines)} structural requests.")

    # Shuffle to get variety of tasks
    random.seed(3407)
    random.shuffle(filtered_lines)
    
    dpo_items = []
    
    for item in filtered_lines:
        if len(dpo_items) >= TRAIN_SIZE:
            break
            
        raw_prompt = item.get("prompt", "")
        asst_msg_chosen = item.get("chosen", "")
        asst_msg_rejected = item.get("rejected", "")
        
        if not raw_prompt or not asst_msg_chosen or not asst_msg_rejected:
            continue
            
        user_msg = get_user_prompt(raw_prompt)
        
        # 1. Generate the HARDCODED prompt using Qwen's exact template
        prompt_chat = [{"role": "user", "content": user_msg}]
        prompt_str = tokenizer.apply_chat_template(prompt_chat, tokenize=False, add_generation_prompt=True)
        
        # 2. Chosen: Pure structural silence
        chosen = clean_structured_data(asst_msg_chosen)
        if len(chosen) < 10:
            continue
            
        # 3. Rejected: Chatty output
        rejected = create_rejected_pattern(chosen, asst_msg_rejected)
        
        dpo_items.append({
            "prompt": prompt_str,
            "chosen": chosen + tokenizer.eos_token, # EOS ensures model stops after pure data
            "rejected": rejected
        })
        
    print(f"Generated {len(dpo_items)} DPO pairs.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for entry in dpo_items:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"✅ Saved train: {len(dpo_items)} items -> {TRAIN_FILE}")

    # Preview
    if dpo_items:
        print("\n--- SAMPLE PREVIEW ---")
        print("[PROMPT]")
        print(dpo_items[0]["prompt"])
        print("[CHOSEN]")
        print(dpo_items[0]["chosen"])
        print("[REJECTED]")
        print(dpo_items[0]["rejected"])
        print("----------------------")

if __name__ == "__main__":
    main()
