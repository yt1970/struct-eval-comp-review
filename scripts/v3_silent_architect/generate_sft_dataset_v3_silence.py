
import json
import os
import re
import random
import datetime

# --- Configuration ---
SOURCE_FILE_HARD = "data/hf_datasets/structured-hard-sft-4k.jsonl"
SOURCE_FILE_1_1 = "data/hf_datasets/sft_1-1.jsonl"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_FILE = f"data/sft_train_data_v3_silence_{TIMESTAMP}.jsonl"

def strip_markdown(text):
    """
    Universally strips markdown code blocks (```json, ```xml, ```, etc) using regex.
    Also strips common chatty prefixes/suffixes if present (though SFT data is usually cleaner).
    """
    # 1. Remove Markdown code blocks
    text = re.sub(r'```\w*', '', text) # Remove opening ```json, ```xml etc.
    text = text.replace("```", "")      # Remove closing ```

    # 2. Remove common chatty prefixes (just in case)
    chatty_patterns = [
        r"^Sure!.*?:",
        r"^Here is the JSON.*?:",
        r"^Certainly!.*?:",
        r"^I can help.*?:",
    ]
    for pattern in chatty_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # 3. Remove "Note:" sections at the end
    if "Note:" in text:
        text = text.split("Note:")[0]

    return text.strip()

def clean_sft_content(text):
    """
    Clean intro text from SFT sources if any (reused logic + regex).
    """
    text = strip_markdown(text)
    
    # Heuristic: Find first { or <
    xml_start = text.find("<root>")
    json_start = text.find("{")
    
    candidates = []
    if xml_start != -1: candidates.append(xml_start)
    if json_start != -1: candidates.append(json_start)
    
    if candidates:
        start_idx = min(candidates)
        text = text[start_idx:]
        
    return text.strip()

def main():
    print(f"🤐 Generating SFT v3 (Silence) Dataset - {TIMESTAMP}")
    data = []

    # 1. Load Hard SFT
    print(f"Loading {SOURCE_FILE_HARD}...")
    try:
        with open(SOURCE_FILE_HARD, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Format: usually {'messages': [...]}
                if 'messages' in item:
                    # Extract User/Assistant
                    user_msg = next((m['content'] for m in item['messages'] if m['role'] == 'user'), None)
                    asst_msg = next((m['content'] for m in item['messages'] if m['role'] == 'assistant'), None)
                    
                    if user_msg and asst_msg:
                        clean_asst = clean_sft_content(asst_msg)
                        if clean_asst:
                            data.append({
                                "text": f"### 指示\n{user_msg}\n\n### 応答\n{clean_asst}" 
                                # Note: We use "text" field for SFTTrainer (packing=False usually needs checking)
                                # Actually, standard SFTTrainer expects 'text' or 'messages'. 
                                # Let's stick to the format used in v1/v2 scripts or use 'messages' format if supported.
                                # Checking v1 script... it used "text" field with manual formatting.
                                # Let's use manual formatting for safety.
                            })
    except FileNotFoundError:
        print("❌ Source file not found!")

    # 2. Load SFT 1-1 (Optional, mix a bit for diversity?)
    # Let's add 500 samples like v2
    print(f"Loading {SOURCE_FILE_1_1}...")
    try:
        sft1_data = []
        with open(SOURCE_FILE_1_1, 'r') as f:
             for line in f:
                item = json.loads(line)
                if 'messages' in item:
                    user_msg = next((m['content'] for m in item['messages'] if m['role'] == 'user'), None)
                    asst_msg = next((m['content'] for m in item['messages'] if m['role'] == 'assistant'), None)
                    if user_msg and asst_msg:
                        clean_asst = clean_sft_content(asst_msg)
                        if clean_asst:
                             sft1_data.append({
                                "text": f"### 指示\n{user_msg}\n\n### 応答\n{clean_asst}"
                            })
        random.shuffle(sft1_data)
        data.extend(sft1_data[:500])
        
    except FileNotFoundError:
         print("❌ Source file 1-1 not found!")

    print(f"Total samples: {len(data)}")
    
    # Final check for markdown residue (Just logging)
    dirty_count = 0
    for d in data:
        if "```" in d['text']:
            dirty_count += 1
    
    print(f"Dirty samples detected (post-clean): {dirty_count}")
    if dirty_count > 0:
        print("⚠️ Warning: Some markdown might still be present!")

    with open(OUTPUT_FILE, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"💾 Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
