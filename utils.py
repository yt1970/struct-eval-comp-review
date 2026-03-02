from datetime import datetime
import pytz

def get_timestamp_jst():
    # 日本時間のタイムスタンプを生成
    jst = pytz.timezone('Asia/Tokyo')
    now = datetime.now(jst)
    return now.strftime("%Y%m%d_%H%M")

# 今後のディレクトリ命名例
timestamp = get_timestamp_jst()
SAVE_PATH = f"models/fused_model_{timestamp}"
ADAPTER_PATH = f"adapters/adapter_{timestamp}"

print(f"📁 フォルダ命名テンプレート: {SAVE_PATH}")
