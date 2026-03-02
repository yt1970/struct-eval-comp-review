# Qwen3-4B-Instruct-2507 Structural Data Specialist (Legend 150)

This model is a LoRA adapter fine-tuned on **Qwen3-4B-Instruct-2507**, specialized for complex structural data conversion and extraction tasks. By adopting the "Legendary ### 指示 (Instruction) Format," this model is optimized to directly output pure structural data without any unnecessary explanatory text or markdown code blocks (```).

## Model Summary
The model addresses the common issue where LLMs tend to add conversational filler or markdown formatting around the requested structured data. Through focused SFT (Supervised Fine-Tuning) on high-quality structural examples, it achieves high precision in formatting for JSON, XML, TOML, CSV, and YAML.

## Training Data and Compliance
The training data for this model has been created in strict compliance with the competition rules and the terms of service of each base model.

### 1. Data Source
- Uses publicly available open datasets (e.g., CC-BY-4.0) as the primary component.

### 2. Data Processing (Rule-based)
- Data cleaning is performed using Python scripts with regex and rule-based logic. It mechanically removes boilerplate text from the beginning of responses, targeting only plain, structured data as the learning objective.

### 3. Data Augmentation (BERT-based Synonyms)
- **Method**: Performed vocabulary substitution using [MASK] language modeling with `bert-base-multilingual-cased`.
- **Content**: By masking parts of the instructions and replacing them with synonyms using BERT, we ensured diversity without compromising the instruction's intent.
- **Compliance Note**: We strictly avoided "distillation" (using outputs from other LLMs like Gemini or GPT-4). The augmentation is entirely based on rule-based processing and an independent language model.

## Usage
To use this model, you can load it using the `transformers` and `peft` libraries:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_id = "Qwen/Qwen3-4B-Instruct-2507"
adapter_id = "your-username/your-adapter-name" # Replace with your actual adapter ID

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_id)

# Perform inference using the Legendary Format
query = "..." # Your query here
prompt = f"### 指示\n{query}\n\n### 応答\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(response.strip())
```

## Inference Method
The following "Legendary Format" input is required to ensure optimal performance.

```text
### 指示
{query}

### 応答
```

## Hyperparameters
- **Base Model**: Qwen/Qwen3-4B-Instruct-2507
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Target Modules**: q_proj, v_proj
- **Learning Rate**: 2e-4
- **Max Steps**: 150

## Expected Performance
- **public_150 Score**: **0.80+** (Target)
- Direct output without markdown decoration.
- High alignment with complex TOML key structures and XML hierarchies.

---

# [日本語訳] Qwen3-4B-Instruct-2507 構造化データ・スペシャリスト (伝説150)

本モデルは、**Qwen3-4B-Instruct-2507** をベースに、複雑な構造化データの変換・抽出タスクに特化してファインチューニングされたLoRAアダプタです。「伝説の ### 指示 フォーマット」を採用することで、余計な説明文やマークダウンコードブロック（```）を一切含まない、純粋な構造化データのみを直接出力するように最適化されています。

## モデルの概要
LLMが要求された構造化データの周囲に対話的なつなぎ言葉やマークダウン形式を付け加えてしまう一般的な課題に対処しています。高品質な構造化データの例を用いた集中したSFTを通じて、JSON、XML、TOML、CSV、YAMLにおいて高いフォーマット精度を実現します。

## 学習データと規約遵守について
本モデルの学習データは、大会規約および各モデルの利用規約を厳格に遵守して作成されています。

### 1. データ出典
- パブリックに利用可能なオープンデータセット（CC-BY-4.0等）を主成分として使用。

### 2. データ加工手法（ルールベース）
- データのクリーニングは、Pythonスクリプトによる正規表現およびルールベースのロジックで実行されています。回答冒頭の定型文を機械的に取り除き、プレーンな構造化データのみを学習ターゲットとしています。

### 3. データ拡張（BERTによる合法的な言い換え）
- **手法**: `bert-base-multilingual-cased` を用いた [MASK] 言語モデルによる語彙置換を行いました。
- **内容**: 指示文の一部をマスクし、BERTにより文脈に合う類義語に置き換えることで、多様性を確保しました。
- **規約への配慮**: 他のLLM（Gemini, GPT-4等）の出力を学習に使う「蒸留」は一切行わず、ルールベースと独立した言語モデルのみで完結しています。

## 使い方
このモデルを使用するには、`transformers` および `peft` ライブラリを使用してロードできます：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_id = "Qwen/Qwen3-4B-Instruct-2507"
adapter_id = "your-username/your-adapter-name" # 実際のプロダクトIDに書き換えてください

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_id)

# 伝説のフォーマットを用いた推論
query = "..." # クエリをここに入力
prompt = f"### 指示\n{query}\n\n### 応答\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(response.strip())
```

## 推論方法
性能を最大限に引き出すためには、以下の「伝説のフォーマット」による入力が必須です。

```text
### 指示
{query}

### 応答
```

## ハイパーパラメータ
- **ベースモデル**: Qwen/Qwen3-4B-Instruct-2507
- **LoRA ランク**: 16
- **LoRA アルファ**: 32
- **ターゲットモジュール**: q_proj, v_proj
- **学習率**: 2e-4
- **最大ステップ**: 150

## 期待される性能
- **public_150 スコア**: **0.80+** (目標)
- マークダウン装飾なしの直接出力。
- 複雑なTOMLキー構造およびXML階層への高い適合。
