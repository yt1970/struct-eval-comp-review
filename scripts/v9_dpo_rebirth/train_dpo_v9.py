import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import DPOConfig, DPOTrainer

# ==============================================================================
# V9 (DPO Rebirth): 0.75モデルの限界突破作戦
# 過去の「3つのバグ（参照モデル、学習率、プロンプト）」を修正した完全版DPO。
# ==============================================================================

# --- 設定 ---
TIMESTAMP = "V9_Rebirth"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# 【超重要】0.75191を叩き出した天才アダプタ (q_proj, v_proj 100件学習のみ)
SFT_ADAPTER_PATH = "adapters/adapter_legacy_sft_20260209_0709" 

# 最高品質のクリーンデータ (Markdown・挨拶完全除去済み)
DATA_PATH = "data/dpo_anti_chatty_V3_1_Deep_Silence.jsonl" 

# 出力先
OUTPUT_DIR = f"outputs/dpo_{TIMESTAMP}"
ADAPTER_OUT_DIR = f"adapters/adapter_dpo_{TIMESTAMP}"

# --- 学習ハイパーパラメータ ---
MAX_STEPS = 150         # 十分に学習させるため少し長めに設定
LEARNING_RATE = 2e-6    # 【修正2】過去の5e-7は低すぎたため、2e-6に引き上げ
BATCH_SIZE = 1
GRAD_ACCUM = 8
BETA = 0.1              # ペナルティ係数 (通常は0.1)

def main():
    print("=" * 60)
    print(f"🚀 V9 (DPO Rebirth) 開始: {TIMESTAMP}")
    print(f"   Base: {BASE_MODEL_ID} + {SFT_ADAPTER_PATH}")
    print(f"   Data: {DATA_PATH} (Deep Silence)")
    print(f"   LR: {LEARNING_RATE}, Steps: {MAX_STEPS}")
    print("=" * 60)

    # 1. トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 【修正1】参照モデルバグの解消
    # SFT済みのアダプタをベースモデルに「マージ（合体）」して、一個のまっさらなモデルにする。
    # こうすることで、DPOTrainer がこれを「参照モデル（SFTの賢い状態）」として正しく認識し、
    # その上に「新しいLoRAアダプタ（DPO用）」をくっつけて学習することができます。
    print("\n📦 1/4: ベースモデルのロード")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"📦 2/4: SFTアダプタ ({SFT_ADAPTER_PATH}) をロードしてマージ")
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)
    model = model.merge_and_unload()
    print("✅ マージ完了！これがDPOの「正しい参照モデル兼ベースモデル」になります。")

    # 3. データセットの準備と【修正3】フォーマットの修正
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    def format_dpo(sample):
        # 推論時と全く同じ `apply_chat_template` を使ってプロンプトを構築する
        # これにより、「学習」と「本番の推論」の形式のズレがなくなり、DPOが確実に効くようになる
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
    
    # 4. 新しいLoRAアダプタ (DPO用) の設定
    # 0.75モデル（SFT）は q, v だけで賢く育ったので、
    # 形式を矯正するDPOもターゲットを最小限に抑え、元の知能を破壊しないようにします。
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
        save_strategy="steps",
        save_steps=50,
        remove_unused_columns=False,
        bf16=False,
        report_to="none",
        beta=BETA
    )

    # DPOTrainerに peft_config を渡すことで、
    # 『入力した model (マージ済み)』をコピーして『参照モデル (ref_model)』とし、
    # 学習用の model には上記の新しいLoRAアダプタを取り付けて最適化してくれます。
    trainer = DPOTrainer(
        model=model,
        ref_model=None, 
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config, 
    )

    print("\n🚀 DPOトレーニングスタート！")
    trainer.train()

    print(f"\n💾 V9アダプタを保存中... -> {ADAPTER_OUT_DIR}")
    trainer.save_model(ADAPTER_OUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUT_DIR)
    print("🎉 V9 (DPO Rebirth) 完了！ 0.8超える準備が整いました。")

if __name__ == "__main__":
    main()
