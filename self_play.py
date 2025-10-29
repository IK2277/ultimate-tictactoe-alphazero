# ====================
# セルフプレイ部
# ====================

# パッケージのインポート
from game import State, state_to_input_tensor
from pv_mcts import pv_mcts_scores
from dual_network import DN_OUTPUT_SIZE, DN_INPUT_SHAPE, DualNetwork, device
from datetime import datetime
from pathlib import Path
import numpy as np
import pickle
import os
import torch

# パラメータの準備
SP_GAME_COUNT = 500 # セルフプレイを行うゲーム数（本家は25000）
SP_TEMPERATURE = 1.0 # ボルツマン分布の温度パラメータ

# 先手プレイヤーの価値
def first_player_value(ended_state):
    # 1:先手勝利, -1:先手敗北, 0:引き分け
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0

# 学習データの保存
def write_data(history):
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True) # フォルダがない時は生成
    path = './data/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(history, f)

# 状態を(H, W, C) = (9, 9, 3)の入力テンソルに変換
# def state_to_input_tensor(state):
#     a, b, c = DN_INPUT_SHAPE # (9, 9, 3)

#     player_pieces = np.zeros((a, b))
#     opponent_pieces = np.zeros((a, b))
#     legal_moves_channel = np.zeros((a, b))

#     # チャンネル 0 (自分) と 1 (相手)
#     for board_idx in range(9):
#         for cell_idx in range(9):
#             R = (board_idx // 3) * 3 + (cell_idx // 3)
#             C = (board_idx % 3) * 3 + (cell_idx % 3)
#             if state.pieces[board_idx][cell_idx] == 1:
#                 player_pieces[R, C] = 1.0
#             if state.enemy_pieces[board_idx][cell_idx] == 1:
#                 opponent_pieces[R, C] = 1.0

#     # チャンネル 2 (合法手)
#     for action in state.legal_actions():
#         board_idx = action // 9
#         cell_idx = action % 9
#         R = (board_idx // 3) * 3 + (cell_idx // 3)
#         C = (board_idx % 3) * 3 + (cell_idx % 3)
#         legal_moves_channel[R, C] = 1.0

#     # 3つのチャンネルを (H, W, C) = (9, 9, 3) の形状にスタック
#     return np.stack([player_pieces, opponent_pieces, legal_moves_channel], axis=-1)

# 1ゲームの実行
def play(model):
    # 学習データ
    history = []

    # 状態の生成
    state = State()

    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 合法手の確率分布の取得
        scores = pv_mcts_scores(model, state, SP_TEMPERATURE)

        # 学習データに状態と方策を追加
        policies = [0] * DN_OUTPUT_SIZE
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        # 状態を(9, 9, 3)のテンソルに変換して保存
        input_tensor = state_to_input_tensor(state)
        history.append([input_tensor, policies, None])

        # 行動の取得
        action = np.random.choice(state.legal_actions(), p=scores)

        # 次の状態の取得
        state = state.next(action)

    # 学習データに価値を追加
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value
    return history

# セルフプレイ
def self_play():
    # 学習データ
    history = []

    # ベストプレイヤーのモデルの読み込み
    model = DualNetwork().to(device)
    model.load_state_dict(torch.load('./model/best.pth', map_location=device, weights_only=True))
    model.eval()

    # 複数回のゲームの実行
    for i in range(SP_GAME_COUNT):
        # 1ゲームの実行
        h = play(model)
        history.extend(h)

        # 出力
        print('\rSelfPlay {}/{}'.format(i+1, SP_GAME_COUNT), end='')
    print('')

    # 学習データの保存
    write_data(history)

    # モデルの破棄
    del model

# 動作確認
if __name__ == '__main__':
    self_play()
