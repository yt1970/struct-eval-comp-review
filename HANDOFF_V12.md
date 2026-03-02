# 🔥 V12 引き継ぎ書 — 次のエージェントへ

## 現状サマリー (2026-02-27 08:38)

コンペの目標は 0.80+ のスコア。現在の最高スコアは **DPO3 = 0.77064**。
**残り提出回数: 3回**（うち1回は安全策として DPO3 = 0.77 を確保すべき）。

## やるべきこと（順番通りに）

### Step 1: V12 学習を実行

```bash
cd /Users/yutako/dev/struct-eval-comp
./.venv/bin/python3 scripts/v12_dpo_on_075/train_v12.py
```

- **所要時間**: 約2.5時間 (Mac M4, MPS)
- **10ステップごとにチェックポイントが保存される**ので、途中で止めても再開可能
- 再開時は同じコマンドを再実行するだけでOK（自動でチェックポイントから再開）
- 出力先: `outputs/v12_dpo_on_075/`
- 最終マージモデル: `models/v12_075_dpo_merged/`

### Step 2: HFにアップロード

```bash
./.venv/bin/hf repo create llm2025_main_v12_075_dpo_merged --repo-type model
./.venv/bin/hf upload satoyutaka/llm2025_main_v12_075_dpo_merged models/v12_075_dpo_merged/ . --repo-type model
```

### Step 3: モデルカードを作成

- `experiments/exp_20260227_v12_dpo_on_075/EXPERIMENT.md` に全情報がある
- README.mdはV11のもの (`experiments/exp_20260225_1303_v11_surgical_dpo/README.md`) を参考に、V12用に書き換える
- 英語→日本語の順序、Sources & License (apache-2.0) を含めること

### Step 4: Colabで推論

標準コード2 (`2025最終課題メインコンペ_標準コード2（提出JSON生成）.ipynb`) で:

```python
MODEL_SOURCE = "merged"
MERGED_MODEL_ID_OR_PATH = "satoyutaka/llm2025_main_v12_075_dpo_merged"
```

### Step 5: 推論結果を分析してから提出判断

```python
# DPO3 (0.77) との比較コマンド
./.venv/bin/python3 -c "
import json
with open('inference_0_DPO3(0.77064).json') as f: dpo3 = json.load(f)
with open('<新しい推論結果>.json') as f: v12 = json.load(f)
# chatty count, avg length, first char distribution を比較
"
```

- お喋り数が DPO3 の 119 件より減っていれば、提出の価値あり
- 「Here」開始が39件以上あったら提出しない（V11 no V10 の二の舞）

### Step 6: 提出計画

| 回 | 内容 | 条件 |
|----|------|------|
| 1 | V12 | 分析結果が良好なら |
| 2 | 予備 | V12不調なら別のアプローチ |
| 3 (最終) | DPO3 (0.77保証) | 安全策 |

## 重要ファイル一覧

| ファイル | 内容 |
|----------|------|
| `scripts/v12_dpo_on_075/train_v12.py` | **V12学習スクリプト（チェックポイント対応済み）** |
| `experiments/exp_20260227_v12_dpo_on_075/EXPERIMENT.md` | **実験の全詳細** |
| `experiments/exp_20260227_v12_dpo_on_075/data/dpo_train.jsonl` | DPO学習データ (160件) |
| `adapters/adapter_legacy_sft_20260209_0709/` | 0.75 SFTアダプタ |
| `/Users/yutako/Downloads/DPO３/` | DPO3アダプタ |
| `inference_0_DPO3(0.77064).json` | DPO3の推論結果（比較用） |
| `2025最終課題メインコンペ_標準コード2（提出JSON生成）.ipynb` | Colab推論コード |
| `LLM2025_main_DPO学習-2.ipynb` | DPO3のオリジナルColab学習コード |

## V12の核心（なぜ上手くいくはず？）

- 0.73 → DPO3 → **0.77 (+0.04)** の実績を、より強い 0.75 ベースで再現
- 前回の 0.75 DPO 失敗は「**違うデータ + 10倍低い学習率**」が原因だった
- 今回は DPO3 と **完全同一の条件** (データ、学習率、LoRA構成) で実行

## やってはいけないこと

- ❌ V10 (ロゼッタストーン暗記モデル) を絶対にマージに含めないこと → 0.15 の惨事
- ❌ 後処理（コンペ規約違反）
- ❌ public_150.json を学習データに使うこと（規約違反）
- ❌ `tokenizer.save_pretrained()` で保存された `tokenizer_config.json` をそのまま使うこと
  → Colab で `AttributeError: 'list' object has no attribute 'keys'` が出る
  → `models/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/tokenizer_config.json` で上書きすること
