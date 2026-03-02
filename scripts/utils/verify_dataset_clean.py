
import json
import sys

DATA_PATH = "data/dpo_synthetic_dataset_v5_20260211_1100.jsonl"

def verify_clean():
    print(f"🕵️‍♀️ Verifying {DATA_PATH}...")
    dirty_count = 0
    total_count = 0
    
    with open(DATA_PATH, 'r') as f:
        for i, line in enumerate(f):
            total_count += 1
            try:
                item = json.loads(line)
                chosen = item.get('chosen', '')
                
                if "```" in chosen:
                    dirty_count += 1
                    if dirty_count <= 5:
                        print(f"❌ Dirty Item #{i+1}:")
                        print(f"   Chosen start: {chosen[:50]}...")
                        print(f"   Markdown present: '```'")
            except json.JSONDecodeError:
                print(f"⚠️ JSON Error at line {i+1}")

    print("-" * 30)
    print(f"Total evaluated: {total_count}")
    print(f"Dirty samples found: {dirty_count}")
    
    if dirty_count == 0:
        print("✨ CLEAN! The dataset is ready for training.")
        sys.exit(0)
    else:
        print("💀 DIRTY! Something is still wrong.")
        sys.exit(1)

if __name__ == "__main__":
    verify_clean()
