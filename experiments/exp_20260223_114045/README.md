# qwen3-4b-struct-evaluation-lora-v1

This repository provides a LoRA adapter fine-tuned from [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) using the PEFT library on Apple Silicon (M4).

This repository contains LoRA adapter weights only. The base model must be loaded separately.

## Training Objective
This adapter is specifically trained to improve structured output accuracy (JSON / XML) for high-difficulty tasks.
A key feature of this model is the **"No-Preamble"** training: the model is trained to output JSON/XML starting directly with `{` or `<` characters, minimizing irrelevant introductory text ("Here is your JSON...") to ensure compatibility with strict parsers.

## Training Configuration (v1-alpha / Proof-of-Concept)
- **Base model:** Qwen/Qwen3-4B-Instruct-2507
- **Method:** LoRA (Standard PEFT)
- **Precision:** bfloat16 (trained on M4 MPS)
- **Training Samples:** 101 samples (extracted from full dataset)
- **Validation Samples:** 10 samples
- **Learning rate:** 5e-5 (Cosine scheduler)
- **LoRA Parameters:** r=16, alpha=32
- **Target Modules:** All major projections (`q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Data Augmentation:** LLM-free BERT-based substitution (3x dataset expansion)

## Data Preparation Process
To ensure high-quality structured outputs, the training data underwent a specialized preparation pipeline:
- **No-Preamble Processing:** Automatically stripped conversational filler (e.g., "Certainly! Here is the XML...") using a custom regex/index-based cleaner to ensure the model learns to output code starting directly with `<root>` or `{`.
- **BERT-Based Augmentation:** Expanded the prompt variety by using `bert-base-multilingual-cased`. Randomly masked non-critical words in instructions and predicted alternatives to improve the model's robustness against different phrasing of tasks.
- **Source Integrity:** Derived from [daichira/structured-hard-sft-4k](https://huggingface.co/datasets/daichira/structured-hard-sft-4k), ensuring a focus on complex transformation logic.

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "Qwen/Qwen3-4B-Instruct-2507"
adapter = "satoyutaka/qwen3-4b-struct-evaluation-lora-v1" # Replace with your actual repo ID

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter)
```

## Sources & License (IMPORTANT)
- **Training Data:** [daichira/structured-hard-sft-4k](https://huggingface.co/datasets/daichira/structured-hard-sft-4k) (v1-alpha uses a 111-sample subset)
- **Dataset License:** Creative Commons Attribution (CC-BY-4.0). This dataset is used and redistributed under the terms of the CC-BY-4.0 license.
- **Compliance:** Users must comply with both the dataset's attribution requirements and the base model's original terms of use.

---

# ＜日本語訳＞
# qwen3-4b-struct-evaluation-lora-v1

このリポジトリは、[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) をベースモデルとし、Apple Silicon (M4) 環境の PEFT ライブラリを用いてファインチューニングされた LoRA アダプターを提供します。

【重要】本リポジトリには LoRA アダプターの重みのみが含まれています。ベースモデルは別途ロードする必要があります。

## 学習の目的
このアダプターは、高難度なタスクにおける構造化出力（JSON / XML）の精度向上を目的としてトレーニングされています。
本モデルの大きな特徴は、**「前置きの排除（No-Preamble）」**学習です。AIが回答の冒頭に不要な文章（例：「はい、承知しました。以下がJSONです：」）を書かず、直接 `{` や `<` から書き始めるように訓練されており、厳格なパースを必要とするシステムとの互換性を最大化しています。

## 学習設定 (v1-alpha / 第0版)
- **ベースモデル:** Qwen/Qwen3-4B-Instruct-2507
- **手法:** LoRA (Standard PEFT)
- **精度:** bfloat16 (M4 MPS デバイスにて学習)
- **学習サンプル数:** 101件 (元データから抽出)
- **検証サンプル数:** 10件
- **学習率:** 5e-5 (Cosine スケジューラ)
- **LoRA パラメータ:** r=16, α=32
- **ターゲットモジュール:** 主要な全プロジェクション層 (`q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **データ拡張:** LLMを使用しないBERTベースの単語置換（データセットを3倍に増幅）

## データ作成プロセス
高品質な構造化出力を実現するため、学習データに対して以下の特別な前処理を行っています。

- **前置きの自動除去 (No-Preamble Processing):** 「承知しました。以下がXMLです」といった会話的な応答を独自のスクリプトで自動的に削除し、モデルが直接 `<root>` や `{` から書き始めるように調整しています。
- **BERTベースのデータ拡張:** `bert-base-multilingual-cased` を使用し、指示文の単語をランダムに置換することで指示のバリエーションを増やしました。これにより、多少異なる命令のされ方に対しても、モデルが安定して同じ形式を出力できるよう「頑健性」を高めています。
- **元データの選定:** 運営提供の [daichira/structured-hard-sft-4k](https://huggingface.co/datasets/daichira/structured-hard-sft-4k) データセットから高難度なタスクを厳選して使用しています。

## 使い方
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "Qwen/Qwen3-4B-Instruct-2507"
adapter = "satoyutaka/qwen3-4b-struct-evaluation-lora-v1" # 自身のリポジトリIDに書き換えてください

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter)
```

## ソースおよびライセンス（重要）
- **学習データ:** [daichira/structured-hard-sft-4k](https://huggingface.co/datasets/daichira/structured-hard-sft-4k) (第0版では111件のサブセットを使用)
- **データセットライセンス:** Creative Commons Attribution (CC-BY-4.0)
  本データセットは、CC-BY-4.0 ライセンスの条項に基づき、使用および再配布が可能です。
- **遵守事項:** 利用者は、データセットの帰属表記（クレジット）に関する要件、およびベースモデルの元の利用規約の両方を遵守する必要があります。
