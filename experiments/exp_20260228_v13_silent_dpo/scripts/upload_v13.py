from huggingface_hub import HfApi

def upload_model():
    repo_id = "satoyutaka/LLM2026_main_v13_dpo"
    folder_path = "models/v13_dpo_silent_merged"
    
    api = HfApi()
    
    # Create the repo if it doesn't exist
    print(f"🛠️ Ensuring repository {repo_id} exists...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    print(f"🚀 Uploading {folder_path} to {repo_id}...")
    
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
    )
    
    print(f"✅ Upload successful! View it at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    upload_model()
