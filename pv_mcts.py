# ====================
# モンテカルロ木探索の作成
# ====================

# パッケージのインポート
from game import State, state_to_input_tensor
from dual_network import DN_INPUT_SHAPE, DualNetwork, device
from math import sqrt
from pathlib import Path
import numpy as np
import torch

# パラメータの準備
PV_EVALUATE_COUNT = 50 # 1推論あたりのシミュレーション回数（本家は1600）
MCTS_BATCH_SIZE = 8 # 高速化対応。MCTSの推論バッチサイズ（1〜PV_EVALUATE_COUNT）

# 動的な探索回数を取得する関数
def get_dynamic_pv_count(cycle):
    """
    サイクル数に応じて探索回数を動的に変更
    cycle 0-9: 100回
    cycle 10-19: 200回
    cycle 20-29: 400回
    cycle 30+: 800回
    """
    if cycle < 10:
        return 100
    elif cycle < 20:
        return 200
    elif cycle < 30:
        return 400
    else:
        return 800

# バッチ推論
def predict_batch(model, states_batch):
    # 推論のための入力データのシェイプの変換
    # 複数の state を (N, H, W, C) = (N, 9, 9, 3) のテンソルにまとめる
    inputs_batch = [state_to_input_tensor(state) for state in states_batch]
    x = np.stack(inputs_batch, axis=0)
    
    # NumPy (N, H, W, C) -> PyTorch (N, C, H, W)
    x = np.transpose(x, (0, 3, 1, 2))
    x = torch.FloatTensor(x).to(device)

    # 推論モード
    model.eval()
    with torch.no_grad():
        policies, values = model(x)
    
    # GPU -> CPU -> NumPy
    policies = policies.cpu().numpy()
    values = values.cpu().numpy()
    
    results = []
    for i in range(len(states_batch)):
        state = states_batch[i]
        legal_actions = state.legal_actions()

        # 方策の取得
        policy = policies[i]
        # 合法手のみを抽出し、合計1の確率分布に変換
        legal_policies = policy[legal_actions] if legal_actions else []
        legal_policies_sum = np.sum(legal_policies)
        if legal_policies_sum > 0:
            legal_policies /= legal_policies_sum
        else:
            # 合法手がないか、すべての方策が0だった場合 (稀)
            # 合法手があれば、均等な確率を割り当てる
            if legal_actions:
                legal_policies = np.ones_like(legal_actions, dtype=float) / len(legal_actions)
            else:
                legal_policies = [] # 合法手なし

        # 価値の取得
        value = values[i][0]
        results.append((legal_policies, value))
        
    return results

# ノードのリストを試行回数のリストに変換
def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores

# モンテカルロ木探索のスコアの取得
def pv_mcts_scores(model, state, temperature, pv_evaluate_count=None):
    # 探索回数の決定（指定がなければデフォルト値を使用）
    if pv_evaluate_count is None:
        pv_evaluate_count = PV_EVALUATE_COUNT

    # モンテカルロ木探索のノードの定義
    class Node:
        # ノードの初期化
        def __init__(self, state, p):
            self.state = state # 状態
            self.p = p # 方策
            self.w = 0 # 累計価値
            self.n = 0 # 試行回数
            self.child_nodes = None  # 子ノード群

        # リーフノードを探索する関数
        def search_leaf(self, path):
            path.append(self)
            
            # ゲーム終了時
            if self.state.is_done():
                value = -1 if self.state.is_lose() else 0 # 負けは-1、引き分けは0
                return self, -value # 相手から見た価値を返す

            # 子ノードが存在しない時 (リーフ)
            if not self.child_nodes:
                return self, 0 # 価値はまだ不明

            # 子ノードが存在する時
            else:
                # アーク評価値が最大の子ノードを探索
                node = self.next_child_node()
                return node.search_leaf(path) # 再帰的に探索

        # ノードを展開し、(policy) を設定する関数
        def expand(self, policies):
            # 子ノードの展開
            self.child_nodes = []
            for action, policy in zip(self.state.legal_actions(), policies):
                self.child_nodes.append(Node(self.state.next(action), policy))

        # 価値をツリーの上位に伝播させる関数
        def backpropagate(self, path, value):
            for node in reversed(path):
                node.w += value
                node.n += 1
                value = -value # 親ノードから見た価値に反転

        # アーク評価値が最大の子ノードを取得
        def next_child_node(self):
            # アーク評価値の計算
            C_PUCT = 1.0
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) +
                    C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))

            # アーク評価値が最大の子ノードを返す
            return self.child_nodes[np.argmax(pucb_values)]

    # 現在の局面のノードの作成
    root_node = Node(state, 0)

    # バッチ評価用のキュー
    leaves_to_eval = [] # 評価対象のリーフノード
    paths_to_leaves = [] # そこに至るまでのパス

    for i in range(pv_evaluate_count):
        # 1. リーフノードを探索
        path = [] # 探索パス
        leaf, value = root_node.search_leaf(path)
        
        # 2. ゲーム終了ノードの場合
        if leaf.state.is_done():
            leaf.backpropagate(path, value) # 価値をそのまま伝播
            continue # このシミュレーションは完了

        # 3. 未評価のリーフノードの場合
        if leaf.n == 0:
            leaves_to_eval.append(leaf)
            paths_to_leaves.append(path)
        else:
            pass 

        # 4. バッチサイズに達したか、最後のシミュレーションの場合
        if len(leaves_to_eval) >= MCTS_BATCH_SIZE or i == PV_EVALUATE_COUNT - 1:
            if leaves_to_eval:
                # 5. バッチ推論を実行
                states_batch = [leaf.state for leaf in leaves_to_eval]
                results_batch = predict_batch(model, states_batch)

                # 6. 結果を各リーフに配布し、展開と伝播を行う
                for leaf, path, (policies, value) in zip(leaves_to_eval, paths_to_leaves, results_batch):
                    leaf.expand(policies)
                    leaf.backpropagate(path, value) # 価値を伝播
                
                # キューをクリア
                leaves_to_eval.clear()
                paths_to_leaves.clear()

    # 合法手の確率分布
    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0: # 最大値のみ1
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else: # ボルツマン分布でバラつき付加
        scores = boltzman(scores, temperature)
    return scores

# モンテカルロ木探索で行動選択
def pv_mcts_action(model, temperature=0, pv_evaluate_count=None):
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature, pv_evaluate_count)
        return np.random.choice(state.legal_actions(), p=scores)
    return pv_mcts_action

# ボルツマン分布
def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]

# 動作確認
if __name__ == '__main__':
    # モデルの読み込み
    path = sorted(Path('./model').glob('*.pth'))[-1]
    model = DualNetwork().to(device)
    model.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
    model.eval()

    # 状態の生成
    state = State()

    # モンテカルロ木探索で行動取得を行う関数の生成
    next_action = pv_mcts_action(model, 1.0)

    # ゲーム終了までループ
    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 行動の取得
        action = next_action(state)

        # 次の状態の取得
        state = state.next(action)

        # 文字列表示
        print(state)
