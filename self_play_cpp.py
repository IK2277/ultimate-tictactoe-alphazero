# ====================
# C++バックエンドを使用した高速セルフプレイ
# ====================

from game import State, state_to_input_tensor
from dual_network import DualNetwork, device
from datetime import datetime
from pathlib import Path
import numpy as np
import pickle
import torch
import os

# C++実装をインポート (利用可能な場合)
try:
    import uttt_cpp
    from pv_mcts_cpp import pv_mcts_action_cpp
    CPP_AVAILABLE = True
    print("Using C++ backend for MCTS")
except ImportError:
    CPP_AVAILABLE = False
    print("C++ backend not available, using Python implementation")
    from pv_mcts import pv_mcts_action

# パラメータの準備
SP_GAME_COUNT = 500 # セルフプレイを行うゲーム数（本家は25000）
SP_TEMPERATURE = 1.0 # ボルツマン分布の温度パラメータ

# PV_EVALUATE_COUNTとMCTS_BATCH_SIZEをC++版に合わせる
PV_EVALUATE_COUNT = 50
MCTS_BATCH_SIZE = 8

# 動的なゲーム数を取得する関数
def get_dynamic_game_count(cycle):
    """
    サイクル数に応じてゲーム数を動的に変更
    cycle 0-9: 500ゲーム (初期学習)
    cycle 10-19: 1000ゲーム (中期学習)
    cycle 20-29: 1500ゲーム (後期学習)
    cycle 30+: 2000ゲーム (微調整)
    """
    if cycle < 10:
        return 500
    elif cycle < 20:
        return 1000
    elif cycle < 30:
        return 1500
    else:
        return 2000

# 1ゲームの実行
def play(model, use_cpp=True, pv_evaluate_count=None):
    # 探索回数の決定
    if pv_evaluate_count is None:
        pv_evaluate_count = PV_EVALUATE_COUNT
    
    # 学習データ
    history = []

    # C++バックエンドの使用判定
    if use_cpp and CPP_AVAILABLE:
        # C++実装を使用
        from pv_mcts_cpp import pv_mcts_scores_cpp
        state = uttt_cpp.State()
    else:
        # Python実装を使用
        from pv_mcts import pv_mcts_scores
        state = State()

    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 入力データの取得
        if use_cpp and CPP_AVAILABLE:
            # C++のStateから直接テンソルを取得
            tensor_flat = np.array(state.to_input_tensor(), dtype=np.float32)
            input_tensor = tensor_flat.reshape(9, 9, 3)
        else:
            input_tensor = state_to_input_tensor(state)

        # MCTSスコアの取得
        if use_cpp and CPP_AVAILABLE:
            scores = pv_mcts_scores_cpp(model, state, SP_TEMPERATURE, pv_evaluate_count, MCTS_BATCH_SIZE)
        else:
            scores = pv_mcts_scores(model, state, SP_TEMPERATURE, pv_evaluate_count)
        
        # 合法手を取得
        legal_actions = state.legal_actions()
        
        # scoresのサイズチェックと正規化
        if len(scores) != len(legal_actions):
            raise ValueError(f"Score size mismatch: scores={len(scores)}, legal_actions={len(legal_actions)}")
        
        scores = np.array(scores, dtype=np.float64)
        if np.sum(scores) == 0:
            scores = np.ones(len(scores)) / len(scores)
        else:
            scores = scores / np.sum(scores)
        
        # 方策を81次元ベクトルに変換
        policy = np.zeros(81)
        for i, action in enumerate(legal_actions):
            policy[action] = scores[i]

        # 行動選択
        action = np.random.choice(legal_actions, p=scores)

        # 学習データに追加
        history.append([input_tensor, policy, None])

        # 次の状態の取得
        state = state.next(action)

    # 学習データに価値を追加
    value = -1 if state.is_lose() else 0 # 終局時の価値

    for i in range(len(history)):
        history[i][2] = value
        value = -value # 手番が変わるので価値を反転

    return history

# セルフプレイ
def self_play(use_cpp=True, pv_evaluate_count=None, game_count=None):
    # 探索回数の決定
    if pv_evaluate_count is None:
        pv_evaluate_count = PV_EVALUATE_COUNT
    
    # ゲーム数の決定
    if game_count is None:
        game_count = SP_GAME_COUNT
    
    # 学習データ
    history = []

    # ベストプレイヤーのモデルの読み込み
    model = DualNetwork().to(device)
    model.load_state_dict(torch.load('./model/best.pth', map_location=device, weights_only=True))
    model.eval()

    # 複数回のゲーム実行
    for i in range(game_count):
        # 1ゲームの実行
        h = play(model, use_cpp, pv_evaluate_count)
        history.extend(h)

        # 出力
        backend = "C++" if (use_cpp and CPP_AVAILABLE) else "Python"
        print(f'\rSelfPlay {i+1}/{game_count} (Backend: {backend}, MCTS: {pv_evaluate_count})', end='')
    print('')

    # 学習データの保存
    now = datetime.now()
    file_name = './data/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    os.makedirs('./data', exist_ok=True)
    with open(file_name, mode='wb') as f:
        pickle.dump(history, f)

# 動作確認
if __name__ == '__main__':
    import os
    
    # C++バックエンドのチェック
    if CPP_AVAILABLE:
        from pv_mcts_cpp import check_cpp_compatibility
        check_cpp_compatibility()
        print("\nStarting self-play with C++ backend...")
        use_cpp = True
    else:
        print("\nC++ backend not available. Starting self-play with Python backend...")
        use_cpp = False
    
    self_play(use_cpp=use_cpp)
