import os
import subprocess
import datetime
import yaml
import shutil
import json

# ==========================================
# V10 Pro Experiment Runner (Legendary Final Version)
# ==========================================

def fix_tokenizer_config(model_path):
    """vLLM等の環境でエラーが出る原因、不適切な extra_special_tokens を除去します。"""
    config_path = os.path.join(model_path, "tokenizer_config.json")
    if not os.path.exists(config_path):
        print(f"⚠️ {config_path} not found. Skipping cleanup.")
        return
    
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = json.load(f)
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return
    
    # リスト形式の extra_special_tokens は互換性を壊すため、辞書形式に直すか削除
    if "extra_special_tokens" in config and isinstance(config["extra_special_tokens"], list):
        print(f"🧹 Cleaning up incompatible extra_special_tokens (list format) in {config_path}...")
        # 完全に削除するか、中身があれば適切に再編（通常は削除でOK）
        del config["extra_special_tokens"]
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print("✅ Tokenizer config fixed and compatible with vLLM!")

def run_experiment():
    # 1. 実験IDの生成 (タイムスタンプ)
    exp_id = datetime.datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    exp_root = os.path.abspath(f"experiments/{exp_id}")
    
    # ディレクトリ作成
    dirs = {
        "data": f"{exp_root}/data",
        "logs": f"{exp_root}/logs",
        "adapter": f"{exp_root}/adapter",
        "scripts": f"{exp_root}/scripts",
        "merged": f"{exp_root}/merged_model"
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"🚀 Starting Experiment: {exp_id}")
    print(f"📂 Root: {exp_root}")

    # 2. 学習データの準備 (コピー)
    # v10用に準備したデータを実験フォルダに固定保存（再現性のため）
    shutil.copy("data/mlx_v10/train.jsonl", f"{dirs['data']}/train.jsonl")
    shutil.copy("data/mlx_v10/valid.jsonl", f"{dirs['data']}/valid.jsonl")

    # 3. 実行スクリプトのバックアップ
    # このスクリプト自身や学習定義をコピー（後で振り返れるように）
    shutil.copy("scripts/v10_mlx/prepare_data.py", f"{dirs['scripts']}/prepare_data.py")

    # 3. ハイパーパラメータの保存 (質重視: Context 4096, Batch 1)
    params = {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "iters": 1000,
        "batch_size": 1, 
        "learning_rate": 1e-5,
        "max_seq_length": 4096,  # 完全に記憶を救済！
        "grad_checkpoint": True,
        "num_layers": 16,
        "save_every": 100,
        "steps_per_eval": 100
    }
    with open(f"{exp_root}/experiment_info.yaml", "w") as f:
        yaml.dump(params, f)

    # 4. MLX学習の実行
    train_log_path = f"{dirs['logs']}/train.log"
    print(f"🔥 Training... Max Length {params['max_seq_length']} with Batch {params['batch_size']}")
    
    cmd = [
        ".venv/bin/python", "-m", "mlx_lm.lora",
        "--model", params["model"],
        "--train",
        "--data", dirs["data"],
        "--batch-size", str(params["batch_size"]),
        "--iters", str(params["iters"]),
        "--save-every", str(params["save_every"]),
        "--steps-per-report", "10",
        "--steps-per-eval", str(params["steps_per_eval"]),
        "--learning-rate", str(params["learning_rate"]),
        "--adapter-path", dirs["adapter"],
        "--max-seq-length", str(params["max_seq_length"]),
        "--grad-checkpoint",
        "--num-layers", str(params["num_layers"])
    ]

    with open(train_log_path, "w") as log_file:
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        print(f"⚙️ Training PID: {process.pid}. Keep an eye on it!")
        # ここではバックグラウンド実行を想定し、PIDを返す
        return process.pid, exp_id

if __name__ == "__main__":
    pid, exp_id = run_experiment()
    print(f"✅ Experiment {exp_id} is cruising in background.")
