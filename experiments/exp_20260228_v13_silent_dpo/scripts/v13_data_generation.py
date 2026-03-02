import json
import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizer, BertModel

# ==============================================================================
# V13 Data Generation Script: "The Silent Brain"
# Based on: /Users/yutako/dev/struct-eval-comp/LLM2025_main_DPOデータ作成-2.ipynb
# Strategy: Hidden CoT DPO (Contrast silent output with chatty thought processes)
# ==============================================================================

# Path configuration
BASE_DIR = "/Users/yutako/dev/struct-eval-comp"
V13_DIR = os.path.join(BASE_DIR, "experiments/exp_20260228_v13_silent_dpo")
DATA_DIR = os.path.join(V13_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Dataset source (Local path provided by user)
LOCAL_DATASET_PATH = os.path.join(BASE_DIR, "data/hf_datasets/dpo-dataset-qwen-cot.jsonl")

# Device setup for logic scoring
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

print(f"🧠 Using device: {device} for logic scoring (BERT)")

# Initialize BERT for filtering logic density (as per DPO2 notebook logic)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)

def get_logic_score(text):
    """Calculates logic density based on the norm of BERT embeddings."""
    if not text:
        return 0
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).norm().item()

def format_v13_silent(example):
    """
    Transforms DPO2 data into V13 'Hidden CoT' format.
    - Prompt: Consistent with DPO2 legend notebook.
    - Chosen: Extraction of structured output ONLY (Silence).
    - Rejected: Original thought process + output (Penalize chattiness).
    """
    original_chosen = example["chosen"]
    
    # 1. Extraction: Extract part after "Output:" for Chosen
    if "Output:" in original_chosen:
        chosen_text = original_chosen.split("Output:")[-1].strip()
    else:
        chosen_text = original_chosen.strip()
    
    # 2. Markdown Slaughter: Remove backticks to ensure pure structured data
    bad_markings = ["```json", "```csv", "```yaml", "```python", "```", "'''"]
    for mark in bad_markings:
        chosen_text = chosen_text.replace(mark, "")
    chosen_text = chosen_text.strip()

    # 3. Contrast: Use the original chatty 'chosen' (including Approach) as Rejected for V13.
    # This teaches the model that "Outputting Thoughts" is a negative trait.
    rejected_text = original_chosen.strip()

    return {
        "prompt": f"### Instruction:\n{example['prompt']}\n\n### Response:\n",
        "chosen": chosen_text,    # SILENT BRAIN
        "rejected": rejected_text  # CHATTY BRAIN
    }

def main():
    if not os.path.exists(LOCAL_DATASET_PATH):
        print(f"❌ Error: Local dataset not found at {LOCAL_DATASET_PATH}")
        return

    print(f"🚀 Loading DPO2 source dataset from local: {LOCAL_DATASET_PATH}...")
    dataset = load_dataset("json", data_files=LOCAL_DATASET_PATH, split="train")

    print(f"✨ Generating V13 'Silent Brain' format for {len(dataset)} items...")
    formatted = dataset.map(format_v13_silent)

    print("🔍 Analyzing logic density with BERT (this might take a moment)...")
    def apply_score(batch):
        # We score the 'rejected' field because it contains the CoT part (Approach)
        batch["logic_score"] = [get_logic_score(t) for t in batch["rejected"]]
        return batch

    scored_dataset = formatted.map(apply_score, batched=True, batch_size=32)

    # 3. Logic Filtering & Sampling
    print("📈 Selecting top-tier logical samples (150 structured + 100 general)...")
    struct_keywords = ["JSON", "CSV", "YAML", "TOML"]
    
    struct_data = scored_dataset.filter(
        lambda x: any(kw in x["prompt"].upper() for kw in struct_keywords)
    ).sort("logic_score", reverse=True)
    
    other_data = scored_dataset.filter(
        lambda x: not any(kw in x["prompt"].upper() for kw in struct_keywords)
    ).sort("logic_score", reverse=True)

    # Sampling per strategy
    final_struct = struct_data.select(range(min(150, len(struct_data))))
    final_other = other_data.select(range(min(100, len(other_data))))

    v13_dataset = concatenate_datasets([final_struct, final_other]).shuffle(seed=3407)

    # Train/Eval Split (90/10)
    ds_split = v13_dataset.train_test_split(test_size=0.1, seed=3407)

    # Save to JSONL
    def save_jsonl(ds, filename):
        path = os.path.join(DATA_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            for item in ds:
                # Keep only DPO required columns for the final file
                output_item = {
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"]
                }
                f.write(json.dumps(output_item, ensure_ascii=False) + "\n")
        return path

    train_path = save_jsonl(ds_split["train"], "v13_silent_dpo_train.jsonl")
    eval_path = save_jsonl(ds_split["test"], "v13_silent_dpo_eval.jsonl")

    print("\n" + "="*50)
    print(f"✅ V13 DATA GENERATION COMPLETE!")
    print(f"📂 Train: {train_path} ({len(ds_split['train'])} items)")
    print(f"📂 Eval:  {eval_path} ({len(ds_split['test'])} items)")
    print("="*50)

    # Verify first item
    sample = ds_split["train"][0]
    print("\n👀 SAMPLE PREVIEW:")
    print(f"[PROMPT]\n{sample['prompt'][:150]}...")
    print(f"\n[CHOSEN - MUST BE SILENT]\n{sample['chosen'][:150]}")
    print(f"\n[REJECTED - MUST BE CHATTY]\n{sample['rejected'][:150]}...")

if __name__ == "__main__":
    main()
