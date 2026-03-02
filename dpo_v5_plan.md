# DPO v5 (Legacy Silence Ultimate) 実施計画

## 1. 概要 (Overview)
**目的**: SFTモデルが抱える「おしゃべり癖」「Markdown囲み」「指示の復唱（末尾ノイズ）」を、DPO学習によって**完全に除去**し、推論コード側での後処理を一切不要にする「完全なJSON出力モデル」を作成する。

## 2. データセット構成と作成方法 (Dataset)
SFTで使用した高品質データ `structured-hard-sft-4k.jsonl` をベースに、人工的に「良い例（Chosen）」と「悪い例（Rejected）」を生成する。

- **ソースデータ**: `structured-hard-sft-4k.jsonl` (約4,000件)
- **生成スクリプト**: `scripts/generate_dpo_dataset_synthetic_v5.py` (新規作成)

### データ構成
各サンプルについて、以下のペアを作成する。

| 項目 | 内容 | 特徴 |
| :--- | :--- | :--- |
| **Prompt** | 元の指示文 | 変更なし |
| **Chosen** | **裸のJSON** + EOS | `{"key": "value"}<|im_end|>` <br> ※余計な文字は一切なし |
| **Rejected** | ノイズ付き出力の詰め合わせ | 以下のパターンをランダムに適用して生成：<br> 1. **Chatty**: `Sure, here is...` + JSON <br> 2. **Markdown**: ` ```json` + JSON + ` ``` ` <br> 3. **Suffix Noise**: JSON + `\n\n### Important...` <br> 4. **Combined**: 上記の複合パターン |

※ Rejectedに「Markdown」や「指示復唱」を明示的に含めることで、モデルは「これらはやってはいけないこと」と学習する。

## 3. 学習戦略とハイパーパラメータ (Strategy)
長時間学習によるフリーズや過学習（崩壊）を防ぎつつ、強力に矯正を行う「短期集中・高火力」設定を採用する。

- **ベースモデル**: `adapter_legacy_sft_20260209_0709` (SFT v2)
    - ※あえてv4（Markdown付きで黙るモデル）ではなく、元のSFTに戻って学習し直すことで、Markdown癖とおしゃべり癖を同時に取り除く。
- **学習スクリプト**: `scripts/train_dpo_legacy_silence_v5.py`
- **主要パラメータ**:
    - `max_steps`: **60** (v4で成功した実績値。約2時間で完了)
    - `learning_rate`: **2.0e-6** (通常の4倍。「高火力」設定)
    - `batch_size`: 1 (gradient_accumulation=8 で実質バッチサイズ8)
    - `save_strategy`: steps (20ステップごとの安全保存)

## 4. 期待効果 (Expected Outcome)
1.  **Markdown除去**: ` ```json ` の囲いがなくなる。
2.  **おしゃべり除去**: `Sure!` などのフィラーがなくなる。
3.  **ノイズ除去**: 生成終了後の `### 重要` などの復唱がなくなる。
4.  **EOS停止**: JSONを書き終えた瞬間に `<|im_end|>` で停止する。

-> **結果**: 公式推論コード（後処理なし）でもそのまま通る、完全なJSONが得られる。
