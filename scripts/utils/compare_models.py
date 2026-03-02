import json
import re
import os

file_v14 = "/Users/yutako/dev/struct-eval-comp/inference_main_v14.json"
file_dpo3 = "/Users/yutako/dev/struct-eval-comp/inference_0_DPO3(0.77064).json"
file_dpo6 = "/Users/yutako/dev/struct-eval-comp/inference_0_DPO6（0.77136）.json"

def analyze_file(path):
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total_len = 0
    markdown_count = 0
    repetitive_tasks = []
    
    for item in data:
        gen = item.get("generation", item.get("answer", ""))
        total_len += len(gen)
        if "```" in gen:
            markdown_count += 1
            
        # Check for repetition (e.g., "H. H. H.")
        if re.search(r"(\s.\.){5,}", gen):
            repetitive_tasks.append(item["task_id"])
            
    return {
        "count": len(data),
        "total_len": total_len,
        "avg_len": total_len / len(data) if data else 0,
        "markdown_pct": (markdown_count / len(data)) * 100 if data else 0,
        "repetitive_count": len(repetitive_tasks),
        "repetitive_ids": repetitive_tasks[:3]
    }

print("Comparing V14 vs DPO3 vs DPO6...")
res_v14 = analyze_file(file_v14)
res_dpo3 = analyze_file(file_dpo3)
res_dpo6 = analyze_file(file_dpo6)

if res_v14:
    print(f"\n[V14 Analysis (Current)]")
    print(f"Avg Length: {res_v14['avg_len']:.2f}")
    print(f"Markdown %: {res_v14['markdown_pct']:.2f}%")
    print(f"Repetitive Loops Found: {res_v14['repetitive_count']}")

if res_dpo3:
    print(f"\n[DPO3 Analysis (0.77064)]")
    print(f"Avg Length: {res_dpo3['avg_len']:.2f}")
    print(f"Markdown %: {res_dpo3['markdown_pct']:.2f}%")
    print(f"Repetitive Loops Found: {res_dpo3['repetitive_count']}")

if res_dpo6:
    print(f"\n[DPO6 Analysis (0.77136)]")
    print(f"Avg Length: {res_dpo6['avg_len']:.2f}")
    print(f"Markdown %: {res_dpo6['markdown_pct']:.2f}%")
    print(f"Repetitive Loops Found: {res_dpo6['repetitive_count']}")
