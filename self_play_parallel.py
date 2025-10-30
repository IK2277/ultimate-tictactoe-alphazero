# ====================
# 並列セルフプレイ（マルチプロセス版）
# ====================

from game import State, state_to_input_tensor
from dual_network import DualNetwork, device
from datetime import datetime
from pathlib import Path
import numpy as np
import pickle
import torch
import os
import multiprocessing as mp
from functools import partial

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

# バッチサイズは自動調整（実行時に決定）
try:
    from auto_tune import get_optimal_batch_size
    MCTS_BATCH_SIZE = get_optimal_batch_size()
except:
    MCTS_BATCH_SIZE = 8  # フォールバック

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

# ワーカープロセスで実行される関数
def worker_play_games(model_path, num_games, use_cpp, pv_evaluate_count, worker_id):
    """
    各ワーカープロセスで複数ゲームを実行
    """
    # PyTorchの最適化を有効化
    from optimize_inference import optimize_pytorch
    optimize_pytorch()
    
    # モデルの読み込み（各プロセスで独立にロード）
    model = DualNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # torch.compile()はマルチプロセスで問題を起こすため無効化
    # 代わりに他の最適化（TF32、cuDNN）が有効になっている
    
    history = []
    for i in range(num_games):
        h = play(model, use_cpp, pv_evaluate_count)
        history.extend(h)
        
        # 進捗表示（各ワーカーから）
        if (i + 1) % 10 == 0:
            backend = "C++" if (use_cpp and CPP_AVAILABLE) else "Python"
            print(f'Worker {worker_id}: {i+1}/{num_games} games completed')
    
    return history

# 並列セルフプレイ
def self_play_parallel(use_cpp=True, pv_evaluate_count=None, game_count=None, num_workers=None, aggressive=False):
    """
    マルチプロセスで並列にセルフプレイを実行
    
    Args:
        use_cpp: C++バックエンドを使用するか
        pv_evaluate_count: MCTS探索回数
        game_count: 総ゲーム数
        num_workers: 並列ワーカー数（Noneの場合は自動決定）
        aggressive: Trueの場合、より多くのワーカーを使用（実験的）
    """
    # 探索回数の決定
    if pv_evaluate_count is None:
        pv_evaluate_count = PV_EVALUATE_COUNT
    
    # ゲーム数の決定
    if game_count is None:
        game_count = SP_GAME_COUNT
    
    # ワーカー数の決定（自動最適化）
    # Windowsのプロセス起動コストとIPCオーバーヘッドを考慮
    if num_workers is None:
        try:
            from auto_tune import get_optimal_num_workers
            num_workers = get_optimal_num_workers(aggressive=aggressive)
        except:
            # フォールバック
            num_workers = 8 if aggressive else 4
    
    backend = "C++" if (use_cpp and CPP_AVAILABLE) else "Python"
    print(f'>> Starting parallel self-play with {num_workers} workers ({backend} backend)')
    print(f'>> Total games: {game_count}, Games per worker: {game_count // num_workers}')
    print(f'>> MCTS batch size: {MCTS_BATCH_SIZE}, PV evaluate count: {pv_evaluate_count}')
    
    model_path = './model/best.pth'
    
    # 各ワーカーが実行するゲーム数を計算
    games_per_worker = game_count // num_workers
    remaining_games = game_count % num_workers
    
    # マルチプロセスプールで並列実行
    # Windows対応: spawn方式を明示的に指定
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        # 各ワーカーに割り当てるゲーム数を計算
        worker_args = []
        for i in range(num_workers):
            # 余りのゲームを最初のワーカーに分配
            worker_games = games_per_worker + (1 if i < remaining_games else 0)
            worker_args.append((model_path, worker_games, use_cpp, pv_evaluate_count, i + 1))
        
        try:
            # 並列実行
            results = pool.starmap(worker_play_games, worker_args)
        except KeyboardInterrupt:
            print("\n>> Interrupted by user")
            pool.terminate()
            pool.join()
            raise
    
    # 全ワーカーの結果を統合
    history = []
    for result in results:
        history.extend(result)
    
    print(f'>> Collected {len(history)} training samples from {game_count} games')
    
    # 学習データの保存
    now = datetime.now()
    file_name = './data/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    os.makedirs('./data', exist_ok=True)
    with open(file_name, mode='wb') as f:
        pickle.dump(history, f)
    
    print(f'>> Saved to {file_name}')

# 動作確認
if __name__ == '__main__':
    # Windows対応: freeze_support()を追加
    mp.freeze_support()
    
    import os
    
    # C++バックエンドのチェック
    if CPP_AVAILABLE:
        from pv_mcts_cpp import check_cpp_compatibility
        check_cpp_compatibility()
        print("\nStarting parallel self-play with C++ backend...")
        use_cpp = True
    else:
        print("\nC++ backend not available. Starting parallel self-play with Python backend...")
        use_cpp = False
    
    # 並列実行
    self_play_parallel(use_cpp=use_cpp)
