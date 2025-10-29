# ====================
# ハイブリッド実装: C++ゲームロジック + Python MCTS
# ====================

from game import State as PyState, state_to_input_tensor
from dual_network import DualNetwork, device
from datetime import datetime
from pathlib import Path
import numpy as np
import pickle
import torch
import os

# C++実装をインポート (ゲームロジックのみ使用)
try:
    import uttt_cpp
    CPP_AVAILABLE = True
    print("Using C++ backend for game logic")
except ImportError:
    CPP_AVAILABLE = False
    print("C++ backend not available, using Python implementation")

from pv_mcts import pv_mcts_action, pv_mcts_scores

# パラメータの準備
SP_GAME_COUNT = 500 # セルフプレイを行うゲーム数
SP_TEMPERATURE = 1.0 # ボルツマン分布の温度パラメータ

# C++の状態をPython版に変換（MCTS用）
def cpp_to_python_state(cpp_state):
    """C++のStateをPython版のStateに変換"""
    py_state = PyState(
        pieces=[list(board) for board in cpp_state.pieces],
        enemy_pieces=[list(board) for board in cpp_state.enemy_pieces],
        main_board_pieces=list(cpp_state.main_board_pieces),
        main_board_enemy_pieces=list(cpp_state.main_board_enemy_pieces),
        active_board=cpp_state.active_board
    )
    return py_state

# Python版の状態をC++に変換
def python_to_cpp_state(py_state):
    """Python版のStateをC++のStateに変換"""
    cpp_state = uttt_cpp.State(
        py_state.pieces,
        py_state.enemy_pieces,
        py_state.main_board_pieces,
        py_state.main_board_enemy_pieces,
        py_state.active_board
    )
    return cpp_state

# 1ゲームの実行（ハイブリッド版）
def play_hybrid(model):
    """
    C++のゲームロジック + PythonのMCTSでゲームを実行
    """
    history = []
    
    if CPP_AVAILABLE:
        # C++で状態管理
        cpp_state = uttt_cpp.State()
    else:
        cpp_state = None
        py_state = PyState()
    
    # Python版MCTSを使用
    next_action_func = pv_mcts_action(model, SP_TEMPERATURE)
    
    while True:
        # 現在の状態をPython版に変換してMCTSに渡す
        if CPP_AVAILABLE:
            py_state = cpp_to_python_state(cpp_state)
        
        # ゲーム終了チェック
        if py_state.is_done():
            break
        
        # 入力テンソルの取得
        input_tensor = state_to_input_tensor(py_state)
        
        # MCTSでスコア取得
        scores = pv_mcts_scores(model, py_state, SP_TEMPERATURE)
        legal_actions = py_state.legal_actions()
        
        # 方策を81次元ベクトルに変換
        policy = np.zeros(81)
        for i, action in enumerate(legal_actions):
            policy[action] = scores[i]
        
        # 行動選択
        action = np.random.choice(legal_actions, p=scores)
        
        # 学習データに追加
        history.append([input_tensor, policy, None])
        
        # 次の状態へ遷移（C++で高速化）
        if CPP_AVAILABLE:
            cpp_state = cpp_state.next(action)
        else:
            py_state = py_state.next(action)
    
    # 学習データに価値を追加
    if CPP_AVAILABLE:
        py_state = cpp_to_python_state(cpp_state)
    
    value = -1 if py_state.is_lose() else 0
    
    for i in range(len(history)):
        history[i][2] = value
        value = -value
    
    return history

# セルフプレイ（ハイブリッド版）
def self_play_hybrid():
    """
    C++のゲームロジック + PythonのMCTSでセルフプレイ
    """
    history = []
    
    # ベストプレイヤーのモデルの読み込み
    model = DualNetwork().to(device)
    model.load_state_dict(torch.load('./model/best.pth', map_location=device, weights_only=True))
    model.eval()
    
    # 複数回のゲーム実行
    for i in range(SP_GAME_COUNT):
        h = play_hybrid(model)
        history.extend(h)
        
        backend = "C++ game + Python MCTS" if CPP_AVAILABLE else "Python"
        print(f'\rSelfPlay {i+1}/{SP_GAME_COUNT} (Backend: {backend})', end='')
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
    if CPP_AVAILABLE:
        print("Testing hybrid implementation...")
        print("✅ C++ module available for game logic")
    else:
        print("⚠️  C++ module not available, using pure Python")
    
    self_play_hybrid()
