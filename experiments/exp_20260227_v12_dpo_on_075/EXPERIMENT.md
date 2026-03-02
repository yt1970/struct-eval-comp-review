# V12 Experiment: DPO3 Recipe on 0.75 SFT Base

## 1. Goal

Break the 0.80 barrier by applying DPO3's proven recipe to the stronger 0.75 SFT base.

### Hypothesis

- 0.73 (SFT 100件) → DPO3 → **0.77** (+0.04)
- 0.75 (SFT 4000件) → 同じDPO → **0.79+** (期待)

0.75 の方が SFT の土台が厚い（40倍のデータ）ため、DPO の効果がさらに高くなる可能性がある。

## 2. Model Lineage (モデル系譜)

```
Qwen/Qwen3-4B-Instruct-2507 (Base)
│
├── SFT (100件, q_proj/v_proj) → 0.73 [satoyutaka/llm2025_main_0]
│   └── DPO1 → DPO2 → DPO3 → 0.77 [satoyutaka/LLM2025_main_0_DPO3]
│
├── SFT (4000件, q_proj/v_proj) → 0.75 [adapter_legacy_sft_20260209_0709]
│   ├── DPO (失敗: 別データ + 低学習率) → 0.75 (変化なし)
│   └── ★ V12 (今回): DPO3と同一条件 → ??? (目標 0.79+)
```

## 3. Why Previous 0.75 DPO Failed (0.75のDPO失敗原因)

| 項目 | 失敗した0.75 DPO | V12（今回） |
|------|-----------------|-----------|
| **データ** | `dpo_synthetic_dataset.jsonl` | `dpo_train.jsonl` (DPO3と同じ) |
| **学習率** | 5e-7 (低すぎた) | **5e-6** (DPO3と同じ) |
| **LoRA対象** | q_proj, v_proj のみ | **全7モジュール** (DPO3と同じ) |
| **学習量** | max_steps=100 | **3 epoch** (≈54 steps, DPO3と同じ) |
| **Optimizer** | adamw_torch | **adamw_torch** |

## 4. Training Configuration (学習設定)

### 4.1 Base Model

- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **SFT Adapter**: `adapters/adapter_legacy_sft_20260209_0709`
  - 学習データ: `structured-hard-sft-4k.jsonl` (4000件)
  - LoRA target: q_proj, v_proj
  - LoRA r=16, alpha=32
  - Score: 0.75191

### 4.2 DPO Training (DPO3と完全同一)

- **Data**: `dpo_train.jsonl` (160件, DPO3成功時と同じ)
  - 元々は Google Drive `/LLM2026/main_competition/DPO_data/dpo_train.jsonl`
  - prompt: ChatMLフォーマット（system + user + assistant）
  - chosen: 正確な構造化データ出力
  - rejected: お喋り・冗長な出力
- **Split**: Train 144件 / Eval 16件 (seed=3407)
- **LoRA**: r=16, alpha=16, target=全7モジュール (q, k, v, o, gate, up, down)
- **Learning Rate**: 5e-6
- **Epochs**: 3
- **Batch**: 1 × grad_accum 8 = effective batch 8
- **Warmup**: 10%
- **Beta**: 0.1
- **Precision**: bfloat16 (MPS)
- **Optimizer**: adamw_torch
- **ref_model**: None (implicit copy)

### 4.3 Post-Training

- DPO LoRA を 0.75 SFT マージ済みモデルに再マージ
- 完全統合モデル (`merged model`) として HF にアップロード
- Colab 標準コード2 で `MODEL_SOURCE = "merged"` として推論

## 5. Data Origin & Compliance (データの由来と規約遵守)

- **SFT データ**: 運営提供の `structured-hard-sft-4k.jsonl` のみ使用
- **DPO データ**: 運営提供のデータセットをベースに作成された `dpo_train.jsonl`
  - Chosen: 正確な構造化出力
  - Rejected: 冗長・不正確な出力
- **蒸留なし**: GPT-4等の商用LLMからの蒸留は一切なし
- **評価データ保護**: `public_150.json` は学習に一切使用していない

## 6. Files in This Experiment

```
experiments/exp_20260227_v12_dpo_on_075/
├── EXPERIMENT.md           ← このファイル
├── scripts/
│   └── train_v12.py        ← 学習スクリプト
├── data/
│   └── dpo_train.jsonl     ← DPO学習データ (160件)
└── logs/
    └── v12_training.log    ← 学習ログ
```

## 7. Expected Output

- `models/v12_075_dpo_merged/` → 完全マージ済みモデル
- HF: `satoyutaka/llm2025_main_v12_075_dpo_merged`
