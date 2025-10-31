# ====================
# 学習サイクルの実行
# ====================

# パッケージのインポート
from dual_network import dual_network
import json
from pathlib import Path
import sys

# ログファイルへの出力設定
LOG_FILE = 'training.log'
log_file_handle = None

def log_print(message):
    """画面とログファイルの両方に出力"""
    print(message, flush=True)
    if log_file_handle:
        log_file_handle.write(message + '\n')
        log_file_handle.flush()

# ========================================
# 並列化設定
# ========================================
USE_PARALLEL = True  # 高速化のため並列版を使用

# ワーカー数の設定
# - "auto": 自動設定（保守的: 4-6ワーカー）
# - "aggressive": 積極的な設定（8-10ワーカー）
# - 数値: 手動で指定（例: 4, 6, 8, 10）
WORKER_MODE = "aggressive"  # "auto", "aggressive", または数値

if USE_PARALLEL:
    try:
        import uttt_cpp
        from self_play_parallel import self_play_parallel as self_play
        print(">> Using parallel C++ backend for maximum speed!", flush=True)
    except ImportError:
        from self_play_parallel import self_play_parallel as self_play
        print(">> Using parallel Python backend (slower)", flush=True)
else:
    # シリアル版（従来版）
    try:
        import uttt_cpp
        from self_play_cpp import self_play
        print(">> Using C++ backend (serial)", flush=True)
    except ImportError:
        from self_play_hybrid import self_play_hybrid as self_play
        print(">> Using hybrid backend (C++ game logic + Python MCTS)", flush=True)

from train_network import train_network, get_dynamic_learning_rate
from evaluate_network import evaluate_network
from evaluate_best_player import evaluate_best_player
from pv_mcts import get_dynamic_pv_count
from self_play_cpp import get_dynamic_game_count

if __name__ == '__main__':
    # デュアルネットワークの作成
    dual_network()

    # チェックポイントファイル
    checkpoint_file = Path('training_checkpoint.json')
    
    # 前回のサイクル番号と試行回数を読み込み
    start_cycle = 0
    start_attempt = 0
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_cycle = checkpoint.get('cycle', 0)
                start_attempt = checkpoint.get('attempt', 0)
            print(f'>> Resuming from cycle {start_cycle} (attempt {start_attempt})', flush=True)
        except:
            print('>> Starting from cycle 0', flush=True)
    else:
        print('>> Starting from cycle 0', flush=True)
    
    # 無限ループで学習サイクルを実行
    i = start_cycle
    attempt = start_attempt
    try:
        while True:
            print('', flush=True)
            print(f'Train {i} (Attempt {attempt}) ====================================', flush=True)
            
            # 動的パラメータの取得（成功したサイクル数iを使用）
            pv_count = get_dynamic_pv_count(i)
            lr = get_dynamic_learning_rate(i)
            game_count = get_dynamic_game_count(i)
            
            print(f'>> Successful Cycle: {i}', flush=True)
            print(f'>> Total Attempts: {attempt}', flush=True)
            print(f'>> Game Count: {game_count}', flush=True)
            print(f'>> MCTS Simulations: {pv_count}', flush=True)
            print(f'>> Learning Rate: {lr}', flush=True)
            
            # セルフプレイ部（並列化設定に応じて実行）
            if USE_PARALLEL:
                # 並列版: ワーカー数を設定
                if isinstance(WORKER_MODE, int):
                    # 手動指定
                    self_play(pv_evaluate_count=pv_count, game_count=game_count, num_workers=WORKER_MODE)
                elif WORKER_MODE == "aggressive":
                    # 積極的な設定
                    self_play(pv_evaluate_count=pv_count, game_count=game_count, aggressive=True)
                else:
                    # 自動（保守的）
                    self_play(pv_evaluate_count=pv_count, game_count=game_count)
            else:
                # シリアル版
                self_play(pv_evaluate_count=pv_count, game_count=game_count)
            
            # パラメータ更新部分（動的学習率）
            print(f'>> Train {i}', flush=True)
            train_network(learning_rate=lr)

            # 新パラメータ評価部（勝率55%以上で更新）
            model_updated = evaluate_network()

            # ベストプレイヤーの評価（常に実行）
            evaluate_best_player()
            
            # モデルが更新された場合のみサイクル数をインクリメント
            if model_updated:
                i += 1
                print(f'>> ✅ Cycle {i-1} SUCCESS! Model updated. Moving to cycle {i}.', flush=True)
            else:
                print(f'>> ⚠️ Attempt {attempt}: Model NOT updated. Retrying cycle {i}.', flush=True)
            
            # 試行回数をインクリメント（評価後）
            attempt += 1
            
            # チェックポイント保存（サイクル数と試行回数の両方を保存）
            with open(checkpoint_file, 'w') as f:
                json.dump({'cycle': i, 'attempt': attempt}, f)
            
            print(f'>> Checkpoint saved (cycle: {i}, attempt: {attempt}).', flush=True)
            
    except KeyboardInterrupt:
        print('', flush=True)
        print(f'>> Training interrupted at cycle {i} (attempt {attempt})', flush=True)
        print(f'>> Progress saved. Resume with: python train_cycle.py', flush=True)
        # 現在のサイクル番号と試行回数を保存
        with open(checkpoint_file, 'w') as f:
            json.dump({'cycle': i, 'attempt': attempt}, f)