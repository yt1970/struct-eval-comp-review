# Qwen-Legend-DPO-Final (Structural Data Specialist)

This repository provides a LoRA adapter fine-tuned through a multi-stage process (SFT -> DPO-Initial -> DPO-Structural-Correction) from **Qwen/Qwen3-4B-Instruct-2507**.
**IMPORTANT**: This repository contains LoRA adapter weights only. The base model must be loaded separately.

## Training Objective
This adapter is specifically trained to eliminate "helpful" but incorrect nesting (hierarchical structures) that large models often produce. It enforces a "Flat Structure" that strictly follows the key paths provided in the instructions, ensuring 100% alignment with evaluation schemas.

### Training Process (Multi-Stage DPO)
1. **SFT (Legend 150)**: Initial fine-tuning to adopt the "Legendary Prompt Format" and eliminate conversational fillers.
2. **DPO Phase 1**: Initial alignment for general instruction following.
3. **DPO Phase 2 (Structural Correction)**: Final stage focused on "Flat Key Advocacy." This phase uses DPO to penalize elegant nesting and reward raw, flat JSON/XML/TOML outputs. During this phase, Accuracy reached 1.0 (100%) and Reward Margin exceeded 20.0.

## Training Configuration
- **Base model**: Qwen/Qwen3-4B-Instruct-2507
- **Method**: LoRA (Fine-tuned on top of previous DPO/SFT weights)
- **Max sequence length**: 1024
- **Max Steps**: 40 (Final structural correction phase)
- **Learning rate**: 2e-6
- **Beta**: 0.1
- **LoRA**: r=16, alpha=32

## Usage
Optimal performance is achieved using the "Legendary Prompt Format":
```text
### 指示
{your_query}

### 応答
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "Qwen/Qwen3-4B-Instruct-2507"
adapter = "satoyutaka/LLM_main_LGSFT_150_DPO2"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter)
```

## Sources & License (IMPORTANT)
- **Training Data**: Based on official competition DPO datasets and rule-based structural flattening.
- **Dataset License**: Follows the original dataset terms.
- **Compliance**: This model does NOT use distillation from other LLMs (Gemini, GPT-4, etc.). All augmentation and correction were performed using rule-based algorithms and BERT-based masking.

---

＜日本語訳＞
# Qwen-Legend-DPO-Final (構造化データ選好・最終版)

本リポジトリは、**Qwen/Qwen3-4B-Instruct-2507** をベースモデルとし、多段階の学習（SFT -> 1次DPO -> 最終構造矯正DPO）を経て開発された LoRA アダプターを提供します。
**【重要】** 本リポジトリには LoRA アダプターの重みのみが含まれています。ベースモデルは別途ロードする必要があります。

## 学習の目的
このアダプターは、高性能モデルが陥りやすい「気を利かせた階層化（ネスト）」を徹底的に排除することを目的としています。指示文で指定されたキーパスを1文字も変えずに「フラット」に出力するように矯正されており、評価スキーマへの100%の適合性を追求しています。

### 学習プロセス
1. **SFT (伝説150)**: 基本的なお喋り禁止と「伝説フォーマット」の習得。
2. **1次DPO**: 回答の質とフォーマット遵守の基礎固め。
3. **最終構造矯正DPO**: 「フラット構造死守」に特化した強化学習。美しいネスト構造をRejected、生のフラット構造をChosenとして、Accuracy 1.0、Margin 20超えの極めて強大な選好を形成しました。

## 学習設定
- **ベースモデル**: Qwen/Qwen3-4B-Instruct-2507
- **手法**: LoRA
- **最大シーケンス長**: 1024
- **最大ステップ数**: 40 (最終構造矯正フェーズ)
- **学習率**: 2e-6
- **Beta**: 0.1
- **LoRA パラメータ**: r=16, α=32

## 使い方
性能を最大限に引き出すためには、以下の「伝説形式」による入力が必須です。
```text
### 指示
{クエリ}

### 応答
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "Qwen/Qwen3-4B-Instruct-2507"
adapter = "satoyutaka/LLM_main_LGSFT_150_DPO2"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter)
```

## ソースおよびライセンス（重要）
- **学習データ**: 運営提供のDPOデータセットおよび、それらをルールベースでフラット化したものを使用。
- **データセットライセンス**: 元データのライセンスに準拠します。
- **遵守事項**: 他のLLM（Gemini, GPT-4等）の出力を学習に使う「蒸留」は一切行わず、ルールベースの加工とBERTによる合法的拡張のみで完結しています。
