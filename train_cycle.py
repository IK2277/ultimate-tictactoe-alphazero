# ====================
# 学習サイクルの実行
# ====================

# パッケージのインポート
from dual_network import dual_network
import json
from pathlib import Path

# C++バックエンドの利用可能性をチェック
try:
    import uttt_cpp
    from self_play_cpp import self_play
    print(">> Using C++ backend for maximum speed!")
except ImportError:
    from self_play_hybrid import self_play_hybrid as self_play
    print(">> Using hybrid backend (C++ game logic + Python MCTS)")

from train_network import train_network, get_dynamic_learning_rate
from evaluate_network import evaluate_network
from evaluate_best_player import evaluate_best_player
from pv_mcts import get_dynamic_pv_count

if __name__ == '__main__':
    # デュアルネットワークの作成
    dual_network()

    # チェックポイントファイル
    checkpoint_file = Path('training_checkpoint.json')
    
    # 前回のサイクル番号を読み込み
    start_cycle = 0
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_cycle = checkpoint.get('cycle', 0)
            print(f'>> Resuming from cycle {start_cycle}')
        except:
            print('>> Starting from cycle 0')
    else:
        print('>> Starting from cycle 0')
    
    # 無限ループで学習サイクルを実行
    i = start_cycle
    try:
        while True:
            print('')
            print(f'Train {i} ====================================')
            
            # 動的パラメータの取得
            pv_count = get_dynamic_pv_count(i)
            lr = get_dynamic_learning_rate(i)
            
            print(f'>> MCTS Simulations: {pv_count}')
            print(f'>> Learning Rate: {lr}')
            
            # セルフプレイ部（C++バックエンド自動選択、動的探索回数）
            self_play(pv_evaluate_count=pv_count)
            
            # パラメータ更新部分（動的学習率）
            print(f'>> Train {i}')
            train_network(learning_rate=lr)

            # 新パラメータ評価部
            update_best_player = evaluate_network()

            # ベストプレイヤーの評価
            if update_best_player:
                evaluate_best_player()
            
            # チェックポイント保存
            with open(checkpoint_file, 'w') as f:
                json.dump({'cycle': i + 1}, f)
            
            print(f'>> Cycle {i} completed. Checkpoint saved.')
            
            # 次のサイクルへ
            i += 1
            
    except KeyboardInterrupt:
        print('')
        print(f'>> Training interrupted at cycle {i}')
        print(f'>> Progress saved. Resume with: python train_cycle.py')
        # 現在のサイクル番号を保存
        with open(checkpoint_file, 'w') as f:
            json.dump({'cycle': i}, f)