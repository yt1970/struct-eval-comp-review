import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import DPOConfig, DPOTrainer
import datetime

# ==========================================
# V11 Surgical DPO Trainer
# ==========================================
# 開発目標: 0.77の知能を維持しつつ、お喋り癖を完全に除去
# 戦略: 
#   1. Legend SFT (100 steps) をマージ
#   2. DPO3 (0.77 model) をマージ
#   3. 極低学習率 (2e-6) で 50ステップだけ DPO を実施
# ==========================================

# --- パス設定 ---
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SILENCE_V10_PATH = "outputs/train_sft_v3_20260211_Silence_v3/checkpoint-100" # V10の中で最もマシな沈黙モデル
DPO3_ADAPTER_PATH = "/Users/yutako/Downloads/DPO３" # 知能の頂点 0.77
DPO_DATA_PATH = "data/dpo_v11_surgical.jsonl"

# 出力先
OUTPUT_DIR = f"outputs/v11_surgical_dpo_{TIMESTAMP}"
ADAPTER_OUT_DIR = f"adapters/adapter_v11_surgical_dpo"

# --- ハイパーパラメータ (V9成功パターン踏襲) ---
MAX_STEPS = 50          # 外科手術なので短時間
LEARNING_RATE = 2e-6    # 2e-6 が沈黙化に有効だった実績あり
BATCH_SIZE = 1
GRAD_ACCUM = 8
BETA = 0.1

def main():
    print("=" * 60)
    print(f"🚀 V11 Surgical DPO 学習開始: {TIMESTAMP}")
    print(f"   Base: {BASE_MODEL_ID}")
    print(f"   Surgical Stage 1 (Merge Intelligence - DPO3): {DPO3_ADAPTER_PATH}")
    print(f"   Surgical Stage 2 (Merge Silence - V10 Iter 100): {SILENCE_V10_PATH}")
    print("=" * 60)

    # 1. トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. ベースモデルのロードとマージ
    print("\n📦 モデルのロードと外科的マージを開始...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f" usando dispositivo: {device}")

    # メモリ節約のため bfloat16 (Mac M4)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    
    # DPO3 (0.77モデル) を合体 (知能)
    print(f"🔗 DPO3 (0.77 - Intelligence) をマージ中... ({DPO3_ADAPTER_PATH})")
    model = PeftModel.from_pretrained(model, DPO3_ADAPTER_PATH)
    model = model.merge_and_unload()
    
    # V10 Iter 100 を合体 (沈黙)
    print(f"🔗 V10 Iter 100 (Silence) をマージ中... ({SILENCE_V10_PATH})")
    model = PeftModel.from_pretrained(model, SILENCE_V10_PATH)
    model = model.merge_and_unload()
    
    print("✅ マージ完了。0.77レベルの知能を持つベースモデルが完成しました。")

    # 3. データセットの準備 (ChatML統一)
    dataset = load_dataset("json", data_files=DPO_DATA_PATH, split="train")

    def format_dpo(sample):
        # 推論時と全く同じ `apply_chat_template` を適用
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["prompt"]}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        return {
            "prompt": prompt_text,
            "chosen": sample["chosen"] + tokenizer.eos_token,
            "rejected": sample["rejected"] + tokenizer.eos_token,
        }

    dataset = dataset.map(format_dpo)
    
    # 4. 新しいLoRAアダプタ (V11 DPO用)
    # 知能を壊さないよう、ターゲットは最小限の q, v に絞る
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], 
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
    )

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        save_strategy="no",
        remove_unused_columns=False,
        bf16=True, # Mac M4 なので bfloat16
        report_to="none",
        beta=BETA,
        logging_steps=1
    )

    # 5. DPO トレーナー設定
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # ref_model=None とすることで `model` のコピーが自動的に使われる
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\n🚀 Surgical DPO スタート！ (目標: 沈黙の知能)")
    trainer.train()

    # 6. 保存
    print(f"\n💾 V11アダプタを保存中... -> {ADAPTER_OUT_DIR}")
    trainer.save_model(ADAPTER_OUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUT_DIR)
    
    print("\n" + "=" * 60)
    print("🎉 V11 Surgical DPO 完了！")
    print("   知能 (0.77) と 沈黙 (V9パターン) の融合に成功しました。")
    print("=" * 60)

if __name__ == "__main__":
    main()
