# Scripts フォルダ構成メモ

整理後のフォルダ構成とそれぞれの役割です。

## 📁 サブフォルダ
- **`v6_minimal_dpo/`**: 現在進行中の DPO v6 (Minimal Silence) 関連。
    - データ生成 (`generate_dpo_minimal_v6.py`)
    - 学習実行 (`train_dpo_minimal_v6.py`)
- **`v3_silent_architect/`**: SFT v3 (Silent Architect) プロジェクト関連。
    - データ生成、学習、検証用スクリプトが含まれます。
- **`v5_legacy_dpo/`**: 過去の DPO v5 / v5.1 関連。
- **`legacy_sft/`**: 過去の SFT v1 / v2 および、スコア0.75を記録したベースモデル関連。
- **`utils/`**: 分析ツール、デバッグ用、共通処理などのユーティリティ。
    - `analyze_inference_errors.py`: 推論結果の不備（CoT混入や構文エラー）を分析。
- **`experiments/`**: 分類に当てはまらない、単発の実験用スクリプト。

## 💡 注意事項
学習スクリプトを実行する際は、仮想環境（`.venv`）を有効にし、ルートディレクトリから相対パスで指定して実行してください。
例: `.venv/bin/python3 scripts/v6_minimal_dpo/train_dpo_minimal_v6.py`
