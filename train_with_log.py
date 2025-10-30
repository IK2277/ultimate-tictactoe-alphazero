"""
学習をログ付きで実行するラッパースクリプト
バッファリングなしでログを書き込む
"""
import subprocess
import sys
import os

LOG_FILE = 'training.log'

print("=" * 60)
print("学習をログ付きで開始します")
print("=" * 60)
print(f"ログファイル: {LOG_FILE}")
print("Ctrl+C で中断できます")
print()

# ログファイルを開く
with open(LOG_FILE, 'w', encoding='utf-8', buffering=1) as log_file:
    try:
        # train_cycle.pyを実行
        process = subprocess.Popen(
            [sys.executable, 'train_cycle.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # 行バッファリング
            universal_newlines=True
        )
        
        # 出力を画面とファイルの両方に表示
        for line in process.stdout:
            print(line, end='', flush=True)  # 画面に表示
            log_file.write(line)  # ファイルに書き込み
            log_file.flush()  # 即座にフラッシュ
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\n中断しました")
        process.terminate()
        process.wait()
        
print(f"\nログは {LOG_FILE} に保存されました")
