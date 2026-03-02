import json
import os
import random
from tqdm import tqdm

# --- Configuration ---
# Use absolute paths
BASE_DIR = "/Users/yutako/dev/struct-eval-comp"
DATA_DIR = os.path.join(BASE_DIR, "data/hf_datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/train_data")

# Source files
PRIMARY_SOURCE = os.path.join(DATA_DIR, "structured-hard-sft-4k.jsonl")
SECONDARY_SOURCE = os.path.join(DATA_DIR, "sft_1-1.jsonl")

# Dataset mixing parameters
PRIMARY_COUNT = 4001 # Take all
SECONDARY_COUNT = 500 # Take a subset for variety/regularization

TRAIN_RATIO = 0.9

def clean_content(text):
    """
    Removes introductory text like 'Here is the XML:' or 'Sure, here creates...'
    Ensures content starts with <root> (for XML) or { (for JSON) or [ (for JSON arrays).
    """
    text = text.strip()
    
    # Check for XML start
    xml_start = text.find("<root>")
    if xml_start != -1:
        return text[xml_start:].strip()
    
    # Check for JSON start
    json_start = text.find("{")
    json_array_start = text.find("[")
    
    start_idx = -1
    
    # Find the earliest valid start character
    candidates = []
    if xml_start != -1: candidates.append(xml_start)
    if json_start != -1: candidates.append(json_start)
    if json_array_start != -1: candidates.append(json_array_start)
    
    if candidates:
        start_idx = min(candidates)
        return text[start_idx:].strip()
        
    return text  # Return original if no structure found (unlikely for this dataset)


def load_jsonl(filepath, limit=None):
    data = []
    print(f"Loading {filepath}...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line in {filepath}")
        
        if limit and len(data) > limit:
            random.shuffle(data)
            data = data[:limit]
        
        print(f"Loaded {len(data)} items from {os.path.basename(filepath)}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return []

def main():
    random.seed(42) # Ensure reproducibility
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    primary_data = load_jsonl(PRIMARY_SOURCE, limit=PRIMARY_COUNT)
    secondary_data = load_jsonl(SECONDARY_SOURCE, limit=SECONDARY_COUNT)
    
    if not primary_data and not secondary_data:
        print("Error: No data loaded. Exiting.")
        return

    # 2. Combine and Shuffle
    all_data = primary_data + secondary_data
    random.shuffle(all_data)
    
    print(f"Total Combined Data: {len(all_data)} items")
    
    cleaned_count = 0 
    
    # 3. Clean Assistant Responses
    cleaned_final_data = []
    for item in tqdm(all_data, desc="Cleaning Data"):
        # Assuming struct: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
        if "messages" in item and len(item["messages"]) >= 2:
            # Check if assistant is the last message
            last_msg = item["messages"][-1]
            if last_msg["role"] == "assistant":
                cleaned_content = clean_content(last_msg["content"])
                last_msg["content"] = cleaned_content
                cleaned_final_data.append(item)
                cleaned_count += 1
            else:
                 # Check if assistant is not last (unlikely but possible)
                 found_assistant = False
                 for msg in item["messages"]:
                     if msg["role"] == "assistant":
                         msg["content"] = clean_content(msg["content"])
                         found_assistant = True
                 if found_assistant:
                     cleaned_final_data.append(item)
                     cleaned_count += 1
        else:
            # Fallback for different formats if any (unlikely for these datasets)
            pass

    print(f"Total Cleaned Data: {len(cleaned_final_data)} items")

    # 4. Split Train/Valid
    split_idx = int(len(cleaned_final_data) * TRAIN_RATIO)
    train_data = cleaned_final_data[:split_idx]
    valid_data = cleaned_final_data[split_idx:]
    
    # 5. Save
    train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    valid_path = os.path.join(OUTPUT_DIR, "valid.jsonl")

    print(f"Saving to {train_path}...")
    with open(train_path, "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Saving to {valid_path}...")
    with open(valid_path, "w", encoding="utf-8") as f:
        for entry in valid_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("Done!")
    print(f"Train: {len(train_data)}")
    print(f"Valid: {len(valid_data)}")

if __name__ == "__main__":
    main()
