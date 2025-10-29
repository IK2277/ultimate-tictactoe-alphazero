# ====================
# 学習サイクルの実行
# ====================

# パッケージのインポート
from dual_network import dual_network

# C++バックエンドの利用可能性をチェック
try:
    import uttt_cpp
    from self_play_cpp import self_play
    print(">> Using C++ backend for maximum speed!")
except ImportError:
    from self_play_hybrid import self_play_hybrid as self_play
    print(">> Using hybrid backend (C++ game logic + Python MCTS)")

from train_network import train_network
from evaluate_network import evaluate_network
from evaluate_best_player import evaluate_best_player

if __name__ == '__main__':
    # デュアルネットワークの作成
    dual_network()

    for i in range(10):
        print('Train',i,'====================')
        # セルフプレイ部（C++バックエンド自動選択）
        self_play()
        
        # パラメータ更新部分
        print(f'>> Train {i}')
        train_network()

        # 新パラメータ評価部
        update_best_player = evaluate_network()

        # ベストプレイヤーの評価
        if update_best_player:
            evaluate_best_player()