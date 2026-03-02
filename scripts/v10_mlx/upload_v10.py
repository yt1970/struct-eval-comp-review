import os
import shutil
import json
from huggingface_hub import HfApi

# ==========================================
# V10 Final Uploader & Cleanup
# ==========================================

MODEL_DIR = "/Users/yutako/dev/struct-eval-comp/models/v10_merged_final"
EXP_DIR = "/Users/yutako/dev/struct-eval-comp/experiments/exp_20260223_114045"
README_TEMPLATE = "/Users/yutako/dev/struct-eval-comp/ADAPTER_README.md"
REPO_ID = "satoyutaka/LLM2025_SFT10"

def main():
    print(f"🚀 Preparing final upload for {REPO_ID}...")

    # 1. README.md の準備
    if os.path.exists(README_TEMPLATE):
        print(f"📄 Creating README.md from template...")
        shutil.copy(README_TEMPLATE, os.path.join(MODEL_DIR, "README.md"))
        shutil.copy(README_TEMPLATE, os.path.join(EXP_DIR, "README.md"))
        print(f"✅ README placed in {MODEL_DIR} and {EXP_DIR}")
    else:
        print(f"⚠️ README template not found at {README_TEMPLATE}!!")

    # 2. Hugging Face へアップロード
    print(f"⬆️ Uploading folder {MODEL_DIR} to {REPO_ID}...")
    
    # 環境変数 HF_TOKEN があればそれを使うし、なければエラーを吐く
    hf_token = os.getenv("HF_TOKEN")
    api = HfApi(token=hf_token)
    
    try:
        api.create_repo(repo_id=REPO_ID, exist_ok=True)
        api.upload_folder(
            folder_path=MODEL_DIR,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="V10 Final: Silent Agent (Merged MLX model)"
        )
        print(f"🎉 SUCCESS!! Model is live at https://huggingface.co/{REPO_ID}")
        
        # 3. 成功後のみ後片付け (削除)
        print(f"🧹 Cleaning up local merged model directory...")
        shutil.rmtree(MODEL_DIR)
        print(f"✅ {MODEL_DIR} has been deleted to save space.")
        
    except Exception as e:
        print(f"❌ ERROR during upload or cleanup: {e}")

if __name__ == "__main__":
    main()
