---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
tags:
- alignment
- dpo
- structural-data
- completion
---

# Qwen3-4B-AgentBench-V11-Surgical-DPO

## 1. Model Summary
V11 Surgical DPO is a LoRA adapter specialized for high-precision structural data tasks, achieving both high logical reasoning and "Perfect Silence" (No-Preamble output). 

This model was constructed by technically fusing two peak performance adapters into the base model:
- **Intelligence Peak**: [satoyutaka/LLM2025_main_0_DPO3](https://huggingface.co/satoyutaka/LLM2025_main_0_DPO3) (0.77064 Score)
- **Silence Prototype**: [satoyutaka/LLM2025_SFT10-iter100](https://huggingface.co/satoyutaka/LLM2025_SFT10-iter100) (Pure Output Specialist)

The fusion was achieved through **Sequential Delta-Weight Integration**. Using the `merge_and_unload()` process, the specific learned weight updates ($\Delta W$) of both DPO3 and V10 were physically incorporated into the base model parameters ($W_{merged} = W_{base} + \Delta W_{DPO3} + \Delta W_{V10}$). This allows the model to inherit the complex reasoning circuits of DPO3 and the radical silence behaviors of V10 at a fundamental weight level, providing an optimized initialization for the final DPO alignment.

## 2. Methodology: Surgical DPO
This model follows a two-stage surgical alignment process:

### 2.1 Technical Weight Merging
The base model parameters were enhanced with existing intelligence and silence adapters as described in the summary, creating a "Hybrid Foundation."

### 2.2 Precision Alignment (DPO)
A final DPO (Direct Preference Optimization) stage was applied to fine-tune the merged weights:
- **Alignment Selection**: Prioritized "Silent/Pure JSON" outputs as the Chosen preferred behavior.
- **Parameters**: Learning Rate 2e-6, 50 steps, Beta 0.1 (Targeting `q_proj` and `v_proj`).
- **Data**: See "Data Origin" section below.

## 3. Data Origin & Compliance (IMPORTANT)
This model strictly adheres to competition rules regarding data ethics.

- **Source Datasets**: Only datasets officially permitted by the organizers (e.g., `structured-hard-sft-4k.jsonl`) were used.
- **Data Generation Script**: Training pairs (Chosen/Rejected) were generated via [**`generate_dpo_v11.py`**](https://huggingface.co/satoyutaka/llm2025_main_v11_surgical_dpo/blob/main/generate_dpo_v11.py). 
- **No LLM Distillation**: This model does NOT use distillation from other LLMs (Gemini, GPT-4, etc.). All augmentation and correction were performed using rule-based mechanical cleansing and BERT-based processing.
- **Dataset License**: Follows the original dataset terms.

## 4. How to Use
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "Qwen/Qwen3-4B-Instruct-2507"
adapter = "satoyutaka/llm2025_main_v11_surgical_dpo"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto")
model = PeftModel.from_pretrained(model, adapter)
```

## 5. Metadata
- **Experiment ID**: `exp_20260225_1303_v11_surgical_dpo`
- **Objective**: Break the 0.80 Score Barrier
- **Precision**: bfloat16

## 6. License
- **License**: Apache-2.0. Follows the original dataset terms.

---

## (日本語訳)

## 1. モデル概要
V11 Surgical DPOは、構造化データ出力タスクにおいて、高い論理推論能力と「完璧な沈黙（前置きなしの出力）」を両立させるために開発されたLoRAアダプターです。

本モデルは、以下の2つの最高性能アダプターをベースモデルに技術的に融合することで構築されました。
- **知能の頂点**: [satoyutaka/LLM2025_main_0_DPO3](https://huggingface.co/satoyutaka/LLM2025_main_0_DPO3) (0.77064 スコア)
- **沈黙のプロトタイプ**: [satoyutaka/LLM2025_SFT10-iter100](https://huggingface.co/satoyutaka/LLM2025_SFT10-iter100) (出力形式特化)

融合には **Sequential Delta-Weight Integration（逐次差分重み統合）** 手法を採用しています。具体的には、PEFTの `merge_and_unload()` プロセスを用い、DPO3とV10のそれぞれが学習した特定の重み更新（$\Delta W$）をベースモデルのパラメータに順次物理的に統合しました（$W_{merged} = W_{base} + \Delta W_{DPO3} + \Delta W_{V10}$）。これにより、DPO3の高度な推論回路とV10の徹底した沈黙性を重みレベルで継承させ、最終的なDPOアライメントに向けた最適な初期化を実現しています。

## 2. 手法：外科的DPO (Surgical DPO)
本モデルは以下の2段階のアライメントプロセスを経て構築されています。

### 2.1 技術的な重みマージ
概要で述べた通り、既存の「知能」と「沈黙」のアダプターをベースモデルに統合し、「ハイブリッドな土台」を構築しました。

### 2.2 精密アライメント (DPO)
マージ後の重みを微調整するため、最終段階のDPOを実施しました。
- **アライメント**: 「沈黙した純粋なJSON」を好ましい振る舞い（Chosen）として優先。
- **パラメータ**: 学習率 2e-6, 50ステップ, Beta 0.1（`q_proj`, `v_proj`をターゲット）。
- **データ**: 下記の「データの由来」セクションを参照。

## 3. データの由来と規約遵守 (重要)
本モデルは、データの倫理性に関するコンペティションの規約を厳格に遵守しています。

- **使用データセット**: 運営から提供された許可済みデータ（`structured-hard-sft-4k.jsonl`等）のみを使用。
- **データ生成スクリプト**: 学習ペア（Chosen/Rejected）は [`generate_dpo_v11.py`](https://huggingface.co/satoyutaka/llm2025_main_v11_surgical_dpo/blob/main/generate_dpo_v11.py) を用いて生成されました。
- **非蒸留の証明**: 他のLLM（Gemini, GPT-4等）からの「蒸留」は一切行っていません。すべてのデータ拡張および修正は、ルールベースの機械的なクレンジングと、BERTベースの処理によって完結しています。
- **データセットライセンス**: 元データのライセンスに準拠します。

## 4. 使い方
上記の「How to Use」を参照してください。

## 5. メタデータ
上記の「Metadata」を参照してください。

## 6. ライセンス
- **ライセンス**: Apache-2.0。元データのライセンスに準拠します。

