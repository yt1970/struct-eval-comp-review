import json
import random
import os
import datetime
import re
from datasets import load_dataset

# --- Configuration ---
# 1. Primary Source (Hard SFT) - 2-3
PRIMARY_SOURCE_PATH = "data/hf_datasets/structured-hard-sft-4k.jsonl"

# 2. Secondary Source (SFT 1-1) - 1-1
SECONDARY_SOURCE_PATH = "data/hf_datasets/sft_1-1.jsonl"

# 3. CoT Source (for DPO Hybrid)
COT_SOURCE_PATH = "data/hf_datasets/dpo-dataset-qwen-cot.jsonl"

# Timestamp for output filename
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_DATA_PATH = f"data/dpo_synthetic_dataset_v5_{TIMESTAMP}.jsonl"

# Noise Templates
MARKDOWN_TEMPLATES = [
    "```json\n{output}\n```",
    "```xml\n{output}\n```", 
    "```\n{output}\n```",
]

CHATTY_PREFIXES = [
    "Sure! Here is the JSON output:\n",
    "Here is the JSON code you requested:\n",
    "Certainly! Below is the structured data:\n",
    "I have converted the data:\n",
]

SUFFIX_NOISES = [
    "\n\n### Important\nPlease ensure the format is valid.",
    "\n\n### Explanation\nThe output contains the requested fields.",
    "\n\nWarning: Do not include markdown formatting.",
]

def strip_markdown(text):
    """
    Universally strips markdown code blocks (```json, ```xml, ```, etc) using regex.
    """
    # Remove ``` followed by optional language identifier (or nothing)
    # This covers ```json, ```xml, ```toml, ```, etc.
    text = re.sub(r'```\w*', '', text)
    return text.strip()

def clean_cot_output(text):
    """
    Removes 'Approach: ... Output:' preamble.
    Markdown stripping is handled separately.
    """
    if "Output:" in text:
        parts = text.split("Output:")
        clean_content = parts[-1].strip()
        return clean_content
    return text.strip()

def clean_sft_content(text):
    """
    Clean intro text from SFT sources if any.
    """
    text = text.strip()
    xml_start = text.find("<root>")
    json_start = text.find("{")
    
    candidates = []
    if xml_start != -1: candidates.append(xml_start)
    if json_start != -1: candidates.append(json_start)
    
    if candidates:
        start_idx = min(candidates)
        return text[start_idx:].strip()
        
    return text

def process_item(item, source_name):
    """
    Process raw item from SFT/CoT sources.
    """
    prompt = ""
    original_output = ""
    rejected_original = "" # For CoT which has it
    
    # Handle different formats
    if "messages" in item:
        # SFT format: messages list
        for msg in item['messages']:
            if msg['role'] == 'user':
                prompt = msg['content']
            elif msg['role'] == 'assistant':
                original_output = msg['content']
    elif "prompt" in item and "chosen" in item:
        # CoT format
        prompt = item['prompt']
        original_output = item['chosen']
        rejected_original = item.get('rejected', "")
    else:
        return None

    if not prompt or not original_output:
        return None

    # --- Create CHOSEN (Silent) ---
    if source_name == "cot":
        chosen = clean_cot_output(original_output)
        
        # Clean prompt for CoT (remove special tokens if present)
        # We want raw text so the trainer can apply chat template correctly.
        if "<|im_start|>user" in prompt:
             try:
                parts = prompt.split("<|im_start|>user")
                # Handle case where multiple turns exist, take the last one or merge?
                # Usually CoT dataset is single turn.
                user_content = parts[-1].split("<|im_end|>")[0].strip()
                prompt = user_content
             except: pass
             
    else:
        # SFT sources
        chosen = clean_sft_content(original_output)

    # --- UNIVERSAL CLEANING (Markdown Stripping) ---
    # Ensure NO markdown remains in chosen, regardless of source.
    chosen = strip_markdown(chosen)

    # --- Create REJECTED (Noisy) ---
    is_xml = "<root>" in chosen or "<?xml" in chosen
    
    if source_name == "cot":
        # Use existing rejected if available (Thinking/Markdown)
        rejected = rejected_original if rejected_original else original_output
    else:
        # SFT sources: Inject noise
        noise_type = random.choice(["markdown", "chatty", "suffix", "all"])
        rejected = chosen 
        
        if noise_type in ["markdown", "all"]:
            template = "```xml\n{output}\n```" if is_xml else "```json\n{output}\n```"
            rejected = template.format(output=rejected)
            
        if noise_type in ["chatty", "all"]:
            rejected = random.choice(CHATTY_PREFIXES) + rejected
            
        if noise_type in ["suffix", "all"]:
            rejected = rejected + random.choice(SUFFIX_NOISES)

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": source_name
    }

def main():
    print(f"👻 Generating DPO v5 Dataset (Hybrid) - Script: {os.path.basename(__file__)}")
    print(f"   Output Target: {OUTPUT_DATA_PATH}")
    
    formatted_data = []

    # 1. Load Primary SFT (Hard SFT)
    print(f"   Loading {PRIMARY_SOURCE_PATH}...")
    try:
        with open(PRIMARY_SOURCE_PATH, 'r') as f:
            for line in f:
                formatted_data.append(process_item(json.loads(line), "sft_hard"))
    except FileNotFoundError:
        print("   ❌ Primary file not found!")

    # 2. Load Secondary SFT (SFT 1-1)
    print(f"   Loading {SECONDARY_SOURCE_PATH}...")
    try:
        sft_1_data = []
        with open(SECONDARY_SOURCE_PATH, 'r') as f:
            for line in f:
                sft_1_data.append(json.loads(line))
        random.shuffle(sft_1_data)
        sft_1_data = sft_1_data[:500] 
        
        for item in sft_1_data:
            formatted_data.append(process_item(item, "sft_1-1"))
            
    except FileNotFoundError:
         print("   ❌ Secondary file not found!")

    # 3. Load CoT
    print(f"   Loading {COT_SOURCE_PATH}...")
    try:
        with open(COT_SOURCE_PATH, 'r') as f:
            for line in f:
                formatted_data.append(process_item(json.loads(line), "cot"))
    except FileNotFoundError:
         print("   ❌ CoT file not found!")

    # Final Clean
    formatted_data = [d for d in formatted_data if d and d['chosen'].strip()]
    random.shuffle(formatted_data)

    print(f"💾 Saving {len(formatted_data)} pairs to {OUTPUT_DATA_PATH}...")
    with open(OUTPUT_DATA_PATH, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print("🎉 Done!")

if __name__ == "__main__":
    main()
