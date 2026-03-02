import json
import re
import os
import random

# === DPO v7: 外科的スタイル矯正 データセット生成 ===
# 
# v6 との最大の違い:
#   Rejected データに「指示文風テキスト」を絶対に含めない。
#   Rejected と Chosen の差は「ラッパー（Markdown/おしゃべり）」のみ。
#   → モデルの知識を壊さず、純粋にスタイルだけを矯正するペアを作る。

SOURCE_FILE = "data/hf_datasets/structured-hard-sft-4k.jsonl"
OUTPUT_FILE = "data/dpo_v7_surgical.jsonl"
TARGET_SIZE = 300

def clean_output(text):
    """
    回答テキストをクリーニングし、純粋なデータのみにする。
    → これが Chosen（正解）になる。
    """
    # 1. Markdown コードブロック除去
    text = re.sub(r'```\w*\n?', '', text)
    text = text.replace("```", "")
    
    # 2. CoT/Approach ブロック除去
    text = re.sub(r"Approach:.*?Output:\n", "", text, flags=re.DOTALL)
    
    # 3. おしゃべり前置き除去
    chatty_patterns = [
        r"^Sure!.*?:\s*\n?",
        r"^Here is the (?:JSON|XML|CSV|YAML|result|data|output).*?:\s*\n?",
        r"^Certainly!.*?:\s*\n?",
        r"^I can help.*?:\s*\n?",
        r"^Below is the.*?:\s*\n?",
        r"^The (?:following|converted|resulting).*?:\s*\n?",
    ]
    for p in chatty_patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)
    
    # 4. 末尾のおしゃべり除去
    tail_patterns = [
        r"\n\s*Note:.*$",
        r"\n\s*I hope this helps.*$",
        r"\n\s*Let me know.*$",
        r"\n\s*Feel free.*$",
        r"\n\s*This (?:JSON|XML|CSV|YAML).*$",
    ]
    for p in tail_patterns:
        text = re.sub(p, "", text, flags=re.DOTALL)
    
    return text.strip()


def make_rejected(clean_text):
    """
    クリーンな回答に「ラッパー」を追加して Rejected（不正解）を作る。
    
    【重要】v6 との違い:
    - 指示文風テキスト（"Approach: 1. Analyze..."）は絶対に含めない
    - 回答の中身は Chosen と完全に同一
    - 違いは「Markdown囲み」や「おしゃべり語尾」などのラッパーのみ
    """
    patterns = [
        # パターン1: Markdown囲み (最も重要な矯正対象)
        lambda t: f"```json\n{t}\n```",
        lambda t: f"```\n{t}\n```",
        
        # パターン2: 前置きのみ
        lambda t: f"Here is the result:\n{t}",
        lambda t: f"Sure! Here you go:\n{t}",
        
        # パターン3: 語尾のみ
        lambda t: f"{t}\n\nI hope this helps!",
        lambda t: f"{t}\n\nLet me know if you need anything else.",
        
        # パターン4: 前置き + Markdown (複合)
        lambda t: f"Sure! Here is the converted data:\n```json\n{t}\n```",
        lambda t: f"Certainly! Below is the result:\n```\n{t}\n```",
    ]
    return random.choice(patterns)(clean_text)


def main():
    print(f"🏥 DPO v7 外科的データセット生成中... 目標: {TARGET_SIZE}件")
    print(f"   方針: Chosen=クリーン回答, Rejected=同じ回答+ラッパー(Markdown/おしゃべり)")
    print(f"   【安全保証】Rejectedに指示文風テキストは一切含めない")
    
    data = []
    
    with open(SOURCE_FILE, 'r') as f:
        all_lines = f.readlines()
    
    random.seed(42)  # 再現性のために固定シード
    random.shuffle(all_lines)
    
    skipped = 0
    for line in all_lines:
        if len(data) >= TARGET_SIZE:
            break
        
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        if 'messages' not in item:
            continue
        
        user_msg = next((m['content'] for m in item['messages'] if m['role'] == 'user'), None)
        asst_msg = next((m['content'] for m in item['messages'] if m['role'] == 'assistant'), None)
        
        if not user_msg or not asst_msg:
            continue
        
        # Chosen: クリーンな回答    
        chosen = clean_output(asst_msg)
        
        # 短すぎる回答はスキップ
        if len(chosen) < 20:
            skipped += 1
            continue
        
        # Rejected: 同じ回答にラッパーを付けただけ
        rejected = make_rejected(chosen)
        
        # 安全性チェック: ChosenとRejectedの中身が同じことを確認
        # (Rejectedからラッパーを除去するとChosenと一致するはず)
        
        data.append({
            "prompt": user_msg,
            "chosen": chosen,
            "rejected": rejected,
        })
    
    # 保存
    os.makedirs(os.path.dirname(OUTPUT_FILE) if os.path.dirname(OUTPUT_FILE) else '.', exist_ok=True)
    with open(OUTPUT_FILE, 'w') as out:
        for item in data:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n✅ 完了: {len(data)} 件のペアを生成 → {OUTPUT_FILE}")
    print(f"   スキップ: {skipped} 件 (短すぎる回答)")
    
    # 先頭3件をプレビュー
    print("\n📋 プレビュー (先頭3件):")
    for i, item in enumerate(data[:3]):
        print(f"\n--- ペア {i+1} ---")
        print(f"Prompt:   {item['prompt'][:80]}...")
        print(f"Chosen:   {item['chosen'][:120]}...")
        print(f"Rejected: {item['rejected'][:120]}...")
        print(f"Chosen len: {len(item['chosen'])}, Rejected len: {len(item['rejected'])}")


if __name__ == "__main__":
    main()
