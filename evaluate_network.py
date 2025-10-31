# ====================
# 新パラメータ評価部
# ====================

# パッケージのインポート
from game import State
from pv_mcts import pv_mcts_action
from dual_network import DualNetwork, device
from pathlib import Path
from shutil import copy
import numpy as np
import torch

# C++ゲームロジックを使用（利用可能な場合）
try:
    import uttt_cpp
    CPP_GAME_AVAILABLE = True
except ImportError:
    CPP_GAME_AVAILABLE = False

# パラメータの準備
EN_GAME_COUNT = 100 # 1評価あたりのゲーム数（50→100に変更）
EN_TEMPERATURE = 0.0 # AlphaZero準拠: 評価時は決定論的（最善手のみ選択）

# 先手プレイヤーのポイント
def first_player_point(ended_state):
    # 1:先手勝利, 0:先手敗北, 0.5:引き分け
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5

# 1ゲームの実行
def play(next_actions):
    # 状態の生成（C++版を優先使用）
    if CPP_GAME_AVAILABLE:
        state = uttt_cpp.State()
    else:
        state = State()

    # ゲーム終了までループ
    while True:
        # ゲーム終了時
        if state.is_done():
            break;

        # 行動の取得
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        # 次の状態の取得
        state = state.next(action)

    # 先手プレイヤーのポイントを返す
    return first_player_point(state)

# ベストプレイヤーの交代
def update_best_player():
    copy('./model/latest.pth', './model/best.pth')
    print('Change BestPlayer', flush=True)

# ネットワークの評価
def evaluate_network():
    # 最新プレイヤーのモデルの読み込み
    model0 = DualNetwork().to(device)
    model0.load_state_dict(torch.load('./model/latest.pth', map_location=device, weights_only=True))
    model0.eval()

    # ベストプレイヤーのモデルの読み込み
    model1 = DualNetwork().to(device)
    model1.load_state_dict(torch.load('./model/best.pth', map_location=device, weights_only=True))
    model1.eval()

    # PV MCTSで行動選択を行う関数の生成
    next_action0 = pv_mcts_action(model0, EN_TEMPERATURE)
    next_action1 = pv_mcts_action(model1, EN_TEMPERATURE)
    next_actions = (next_action0, next_action1)

    # 複数回の対戦を繰り返す
    total_point = 0
    for i in range(EN_GAME_COUNT):
        # 1ゲームの実行
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))

        # 出力
        print('\rEvaluate {}/{}'.format(i + 1, EN_GAME_COUNT), end='', flush=True)
    print('', flush=True)

    # 平均ポイントの計算
    average_point = total_point / EN_GAME_COUNT
    print('AveragePoint', average_point, flush=True)

    # モデルの破棄
    del model0
    del model1

    # ベストプレイヤーの更新（勝率55%以上で更新）
    UPDATE_THRESHOLD = 0.55  # 更新基準を厳格化
    
    if average_point >= UPDATE_THRESHOLD:
        update_best_player()
        print('>> Model updated! (win rate: {:.1%})'.format(average_point), flush=True)
        return True
    else:
        print('>> Model NOT updated (win rate: {:.1%} < {:.1%} threshold)'.format(average_point, UPDATE_THRESHOLD), flush=True)
        return False

# 動作確認
if __name__ == '__main__':
    evaluate_network()
