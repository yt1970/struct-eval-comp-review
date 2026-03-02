import os
import argparse
import subprocess
import json
import shutil
from huggingface_hub import HfApi

# ==========================================
# V10 Pro Finalizer: Fuse, Fix, and Upload
# ==========================================

def fix_tokenizer_config(model_path):
    """ゾンビバグ (extra_special_tokens がリスト形式) を根絶します"""
    config_path = os.path.join(model_path, "tokenizer_config.json")
    if not os.path.exists(config_path):
        print(f"⚠️ {config_path} not found.")
        return
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    if "extra_special_tokens" in config and isinstance(config["extra_special_tokens"], list):
        print(f"🧹 Fixing incompatible extra_special_tokens in {config_path}...")
        del config["extra_special_tokens"]
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print("✅ Config cleaned!")

def main():
    parser = argparse.ArgumentParser(description="Fuse adapter and upload model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to trained adapter")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--repo", type=str, default="satoyutaka/Qwen2.5-4B-AgentBench-llm2025_v10_silent")
    parser.add_argument("--output", type=str, default="models/v10_merged_final")
    args = parser.parse_args()

    # 1. Fuse (マージ)
    print(f"🚀 Fusing adapter {args.adapter} into base model...")
    # mlx_lm.fuse を使ってマージを実行
    cmd = [
        ".venv/bin/python", "-m", "mlx_lm.fuse",
        "--model", args.model,
        "--adapter-path", args.adapter,
        "--save-path", args.output
    ]
    subprocess.run(cmd, check=True)

    # 2. 自動クレンジング (ゾンビ修正)
    fix_tokenizer_config(args.output)

    # 3. Hugging Face へアップロード
    print(f"⬆️ Uploading to HF repo: {args.repo}...")
    api = HfApi()
    
    # 既存リポジトリがなければ作成 (任意)
    try:
        api.create_repo(repo_id=args.repo, exist_ok=True)
    except:
        pass

    api.upload_folder(
        folder_path=args.output,
        repo_id=args.repo,
        repo_type="model"
    )

    print(f"🎉 SUCCESS!! Model is live at https://huggingface.co/{args.repo}")

if __name__ == "__main__":
    main()
