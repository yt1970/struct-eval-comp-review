import json
import re

def analyze_chatty(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    chatty_count = 0
    total = len(data)
    
    # Preambles to look for
    preambles = ["Sure", "Here is", "Below is", "Here's", "Based on", "As requested"]
    
    chatty_ids = []
    
    for item in data:
        gen = item.get("generation", "").strip()
        
        # Simple heuristic: If it doesn't start with a structural marker, it's chatty
        # Structural markers: { (JSON), [ (JSON array), < (XML), --- (YAML), ``` (Markdown code block)
        if not (gen.startswith("{") or gen.startswith("[") or gen.startswith("<") or gen.startswith("---") or gen.startswith("```")):
            chatty_count += 1
            chatty_ids.append(item.get("task_id"))
            continue
            
        # Even if it starts with structural markers, it might have extra text after it.
        # But usually in these competitions, the preamble is the biggest issue.
        # Some models put the code block first then add "Hope this helps!".
        
        # Checking for text after code block
        if gen.startswith("```"):
            # If it has more than one code block or text outside the code block
            # This is harder to catch with simple startswith, but let's check for the closing ```
            if not gen.endswith("```"):
                chatty_count += 1
                chatty_ids.append(item.get("task_id"))

    return chatty_count, total, chatty_ids[:5] # Show first 5 chatty IDs

v12_path = "/Users/yutako/dev/struct-eval-comp/inference_v12_075_dpo_merged.json"
dpo3_path = "/Users/yutako/dev/struct-eval-comp/inference_0_DPO3(0.77064).json"

print("📊 Analyzing V12 VS DPO3 Silence Strategy...")
v12_chatty, v12_total, v12_samples = analyze_chatty(v12_path)
dpo3_chatty, dpo3_total, dpo3_samples = analyze_chatty(dpo3_path)

print(f"\nV12 (0.75 base + DPO3 recipe):")
print(f"  Chatty Count: {v12_chatty} / {v12_total} ({v12_chatty/v12_total:.1%})")
print(f"  Samples: {v12_samples}")

print(f"\nDPO3 (0.73 base + DPO3 recipe):")
print(f"  Chatty Count: {dpo3_chatty} / {dpo3_total} ({dpo3_chatty/dpo3_total:.1%})")
print(f"  Samples: {dpo3_samples}")

if v12_chatty < dpo3_chatty:
    print("\n✅ V12 is more silent than DPO3!")
elif v12_chatty > dpo3_chatty:
    print("\n⚠️ V12 is MORE chatty than DPO3!")
else:
    print("\n🤝 Same chatty level.")
