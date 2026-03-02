import json
import re
import random
from tqdm import tqdm

SOURCE_FILE = "data/hf_datasets/dpo-dataset-qwen-cot.jsonl"
OUTPUT_FILE = "data/train_data/dpo_flat_structure_v2.jsonl"

def flatten_dict(d, parent_key='', sep='_'):
    """
    辞書を再帰的にフラット化する
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def clean_output(text):
    if "Output:" in text:
        text = text.split("Output:")[1]
    text = re.sub(r"```[a-z]*\n?", "", text)
    text = text.replace("```", "")
    return text.strip()

def process():
    with open(SOURCE_FILE, "r") as f:
        lines = [json.loads(line) for line in f]

    dpo_data = []
    for item in tqdm(lines[:1500]):
        # 指示文
        prompt_raw = item["prompt"]
        user_match = re.search(r"<\|im_start\|>user\n(.*?)\n<\|im_end\|>", prompt_raw, re.DOTALL)
        if not user_match: continue
        instruction = user_match.group(1).strip()

        # 1. Chosen (フラット化+クリーン)
        # 提供データの chosen が JSON の場合のみフラット化を試みる
        original_chosen = clean_output(item["chosen"])
        try:
            if original_chosen.startswith("{"):
                data = json.loads(original_chosen)
                flat_data = flatten_dict(data)
                chosen = json.dumps(flat_data, ensure_ascii=False)
            else:
                chosen = original_chosen
        except:
            chosen = original_chosen

        # 2. Rejected (ネストしたまま+装飾)
        # 敢えて解説文やマークダウンを残し、かつ複雑なネストを維持
        rejected = item["rejected"]

        dpo_data.append({
            "prompt": f"### 指示\n{instruction}\n\n### 応答\n",
            "chosen": chosen,
            "rejected": rejected
        })

    # DPO3(0.77)版の「成功パターン」をシミュレートしたデータも追加
    # フラットなキーこそが正解であるというペア
    with open(OUTPUT_FILE, "w") as f:
        for entry in dpo_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    process()
    print(f"Saved: {OUTPUT_FILE}")
