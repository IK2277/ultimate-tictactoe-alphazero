# ====================
# Ultimate-tic-tac-toe (究極の三目並べ)
# ====================

# パッケージのインポート
import random
import math
import numpy as np
from dual_network import DN_INPUT_SHAPE

# ゲーム状態
class State:
    # 初期化
    def __init__(self, pieces=None, enemy_pieces=None, main_board_pieces=None, main_board_enemy_pieces=None, active_board=-1):
        # pieces[i][j] = i番目の小盤面のj番目のマスにある、現プレイヤーの石
        self.pieces = pieces if pieces is not None else [[0] * 9 for _ in range(9)]
        # enemy_pieces[i][j] = i番目の小盤面のj番目のマスにある、相手プレイヤーの石
        self.enemy_pieces = enemy_pieces if enemy_pieces is not None else [[0] * 9 for _ in range(9)]
        
        # main_board_pieces[i] = 現プレイヤーが i番目の小盤面で勝利したか
        self.main_board_pieces = main_board_pieces if main_board_pieces is not None else [0] * 9
        # main_board_enemy_pieces[i] = 相手プレイヤーが i番目の小盤面で勝利したか
        self.main_board_enemy_pieces = main_board_enemy_pieces if main_board_enemy_pieces is not None else [0] * 9
        
        # 次にプレイすべき小盤面のインデックス (0-8)。-1は任意。
        self.active_board = active_board

    # 9マスの盤面 (list) を受け取り、勝利しているか判定する (元のis_loseのロジック)
    def _check_win(self, board_pieces):
        # 3並びかどうか
        def is_comp(x, y, dx, dy):
            for k in range(3):
                if y < 0 or 2 < y or x < 0 or 2 < x or \
                    board_pieces[x+y*3] == 0:
                    return False
                x, y = x+dx, y+dy
            return True

        if is_comp(0, 0, 1, 1) or is_comp(0, 2, 1, -1):
            return True
        for i in range(3):
            if is_comp(0, i, 1, 0) or is_comp(i, 0, 0, 1):
                return True
        return False

    # 石の総数の取得
    def piece_count(self, pieces_list):
        count = 0
        for board in pieces_list:
            for cell in board:
                if cell == 1:
                    count +=  1
        return count

    # 負けかどうか (メインボードで相手が勝ったか)
    def is_lose(self):
        return self._check_win(self.main_board_enemy_pieces)

    # 引き分けかどうか
    def is_draw(self):
        # 負けでなく、かつ合法手がない
        return not self.is_lose() and len(self.legal_actions()) == 0

    # ゲーム終了かどうか
    def is_done(self):
        return self.is_lose() or self.is_draw()

    # 次の状態の取得
    def next(self, action):
        # action (0-80) から board_idx (0-8) と cell_idx (0-8) を計算
        board_idx = action // 9
        cell_idx = action % 9

        # 1. プレイヤーを交代し、盤面をディープコピー
        new_pieces = [b.copy() for b in self.enemy_pieces]
        new_enemy_pieces = [b.copy() for b in self.pieces]
        new_main_pieces = self.main_board_enemy_pieces.copy()
        new_main_enemy_pieces = self.main_board_pieces.copy()

        # 2. (元)プレイヤーの石を (新)相手の盤面に配置
        new_enemy_pieces[board_idx][cell_idx] = 1

        # 3. 配置した小盤面(board_idx)で勝利したか判定
        small_board_win = self._check_win(new_enemy_pieces[board_idx])
        
        if small_board_win:
            new_main_enemy_pieces[board_idx] = 1 # (新)相手の勝利
        else:
            # 小盤面が引き分けか判定
            board_full = True
            for j in range(9):
                if new_pieces[board_idx][j] == 0 and new_enemy_pieces[board_idx][j] == 0:
                    board_full = False
                    break
            if board_full:
                # C++の実装に倣い、引き分けは両方の勝利としてマークし、プレイ不可にする
                new_main_pieces[board_idx] = 1
                new_main_enemy_pieces[board_idx] = 1 

        # 4. 次のアクティブボードを決定
        # 今置いた cell_idx が次の board_idx になる
        next_active_board = cell_idx

        # ただし、その盤面が既に終了している場合は -1 (任意) にする
        target_board_finished = (new_main_pieces[next_active_board] == 1 or 
                                 new_main_enemy_pieces[next_active_board] == 1)

        if target_board_finished:
            next_active_board = -1
        
        # 5. 新しい状態を返す
        return State(new_pieces, new_enemy_pieces, 
                     new_main_pieces, new_main_enemy_pieces, 
                     next_active_board)

    # 合法手のリストの取得
    def legal_actions(self):
        actions = []
        if self.is_lose():
            return []

        # 1. プレイ可能な盤面(candidate_boards)のリストを作成
        candidate_boards = []
        if self.active_board == -1:
            # 任意: 終了していない全ての盤面が候補
            for i in range(9):
                if self.main_board_pieces[i] == 0 and self.main_board_enemy_pieces[i] == 0:
                    candidate_boards.append(i)
        else:
            # 強制: active_boardが候補
            # ただし、active_boardが既に終了している場合は、任意(-1)と同じルールになる
            if self.main_board_pieces[self.active_board] == 0 and self.main_board_enemy_pieces[self.active_board] == 0:
                candidate_boards.append(self.active_board)
            else:
                # 送り込まれた先が既に終了していた場合
                for i in range(9):
                    if self.main_board_pieces[i] == 0 and self.main_board_enemy_pieces[i] == 0:
                        candidate_boards.append(i)

        # 2. 候補の盤面(candidate_boards)から、空いているセルを探す
        for board_idx in candidate_boards:
            for cell_idx in range(9):
                if self.pieces[board_idx][cell_idx] == 0 and self.enemy_pieces[board_idx][cell_idx] == 0:
                    # アクションを 0-80 の整数として追加
                    actions.append(board_idx * 9 + cell_idx)
        
        return actions

    # 先手かどうか
    def is_first_player(self):
        # 全ての盤面の石の数を合計して比較
        return self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)

    # 文字列表示
    def __str__(self):
        # ox[0] = 現プレイヤーの石, ox[1] = 相手の石
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        s = ""
        # r = メイン行(0-2), c = 小盤面行(0-2)
        # i = メイン列(0-2), j = 小盤面列(0-2)
        for r in range(3):
            for c in range(3):
                for i in range(3):
                    board_idx = r * 3 + i
                    for j in range(3):
                        cell_idx = c * 3 + j
                        
                        p = '-'
                        if self.pieces[board_idx][cell_idx] == 1:
                            p = ox[0] # 現プレイヤー
                        elif self.enemy_pieces[board_idx][cell_idx] == 1:
                            p = ox[1] # 相手
                        
                        s += p + " "
                    if i < 2: s += "| "
                s += "\n"
            if r < 2: s += "---------------------\n"
        
        s += "\nMain Board Status:\n"
        for i in range(9):
            mb_p = '.'
            if self.main_board_pieces[i] == 1 and self.main_board_enemy_pieces[i] == 1:
                mb_p = 'D' # Draw
            elif self.main_board_pieces[i] == 1:
                mb_p = ox[0]
            elif self.main_board_enemy_pieces[i] == 1:
                mb_p = ox[1]
            s += mb_p
            if i % 3 == 2: s += '\n'
        
        next_player = ox[0]
        s += f"Next Player: {next_player}\n"
        s += f"Active Board: {'Any' if self.active_board == -1 else self.active_board}\n"
        return s

# ----------------------------------------------------------------
# 変更: ここから下を追加 (self_play.py から state_to_input_tensor を移設)
# ----------------------------------------------------------------

# 状態を(H, W, C) = (9, 9, 3)の入力テンソルに変換
def state_to_input_tensor(state):
    a, b, c = DN_INPUT_SHAPE # (9, 9, 3)

    player_pieces = np.zeros((a, b))
    opponent_pieces = np.zeros((a, b))
    legal_moves_channel = np.zeros((a, b))

    # チャンネル 0 (自分) と 1 (相手)
    for board_idx in range(9):
        for cell_idx in range(9):
            R = (board_idx // 3) * 3 + (cell_idx // 3)
            C = (board_idx % 3) * 3 + (cell_idx % 3)
            if state.pieces[board_idx][cell_idx] == 1:
                player_pieces[R, C] = 1.0
            if state.enemy_pieces[board_idx][cell_idx] == 1:
                opponent_pieces[R, C] = 1.0

    # チャンネル 2 (合法手)
    for action in state.legal_actions():
        board_idx = action // 9
        cell_idx = action % 9
        R = (board_idx // 3) * 3 + (cell_idx // 3)
        C = (board_idx % 3) * 3 + (cell_idx % 3)
        legal_moves_channel[R, C] = 1.0

    # 3つのチャンネルを (H, W, C) = (9, 9, 3) の形状にスタック
    return np.stack([player_pieces, opponent_pieces, legal_moves_channel], axis=-1)

# ----------------------------------------------------------------
# 変更: ここまで追加
# ----------------------------------------------------------------

# ランダムで行動選択
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]

# アルファベータ法で状態価値計算
# (注意: Ultimate-tic-tac-toeでは探索空間が広すぎるため、実用的な時間では終わりません)
def alpha_beta(state, alpha, beta):
    # 負けは状態価値-1
    if state.is_lose():
        return -1

    # 引き分けは状態価値0
    if state.is_draw():
        return  0

    # 合法手の状態価値の計算
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -beta, -alpha)
        if score > alpha:
            alpha = score

        # 現ノードのベストスコアが親ノードを超えたら探索終了
        if alpha >= beta:
            return alpha

    # 合法手の状態価値の最大値を返す
    return alpha

# アルファベータ法で行動選択
def alpha_beta_action(state):
    # 合法手の状態価値の計算
    best_action = 0
    alpha = -float('inf')
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha)
        if score > alpha:
            best_action = action
            alpha = score

    # 合法手の状態価値の最大値を持つ行動を返す
    return best_action

# プレイアウト
def playout(state):
    # 負けは状態価値-1
    if state.is_lose():
        return -1

    # 引き分けは状態価値0
    if state.is_draw():
        return  0

    # 次の状態の状態価値
    return -playout(state.next(random_action(state)))

# 最大値のインデックスを返す
def argmax(collection):
    return collection.index(max(collection))

# モンテカルロ木探索の行動選択
def mcts_action(state):
    # モンテカルロ木探索のノード
    class node:
        # 初期化
        def __init__(self, state):
            self.state = state # 状態
            self.w = 0 # 累計価値
            self.n = 0 # 試行回数
            self.child_nodes = None  # 子ノード群

        # 評価
        def evaluate(self):
            # ゲーム終了時
            if self.state.is_done():
                # 勝敗結果で価値を取得
                value = -1 if self.state.is_lose() else 0 # 負けは-1、引き分けは0

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

            # 子ノードが存在しない時
            if not self.child_nodes:
                # プレイアウトで価値を取得
                value = playout(self.state)

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1

                # 子ノードの展開 (元の10回から変更)
                if self.n == 1: # 1回試行したらすぐ展開する
                    self.expand()
                return value

            # 子ノードが存在する時
            else:
                # UCB1が最大の子ノードの評価で価値を取得
                value = -self.next_child_node().evaluate()

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

        # 子ノードの展開
        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(node(self.state.next(action)))

        # UCB1が最大の子ノードを取得
        def next_child_node(self):
             # 試行回数nが0の子ノードを返す
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node

            # UCB1の計算
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                # UCB1の定数を調整 (元の 2*... から 0.5*... へ)
                ucb1_values.append(-child_node.w/child_node.n + 0.5 * (2*math.log(t)/child_node.n)**0.5)

            # UCB1が最大の子ノードを返す
            return self.child_nodes[argmax(ucb1_values)]

    # ルートノードの生成
    root_node = node(state)
    root_node.expand()

    # ルートノードを1000回評価 (UTTTは複雑なので100回から増やす)
    for _ in range(1000):
        root_node.evaluate()

    # 試行回数の最大値を持つ行動を返す
    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]

# 動作確認
if __name__ == '__main__':
    # 状態の生成
    state = State()
    
    # ゲーム終了までのループ
    while True:
        # ゲーム終了時
        if state.is_done():
            # 最終盤面を表示
            print(state)
            if state.is_lose():
                # is_lose()は「現プレイヤーの負け」なので、
                # 勝利したのは「相手」＝ox[1]
                ox = ('o', 'x') if state.is_first_player() else ('x', 'o')
                print(f"Winner: {ox[1]}")
            else:
                print("Draw")
            break

        # 現プレイヤーの手番
        if state.is_first_player():
            # 'o' (先手) はMCTS
            action = mcts_action(state)
            print("Player 'o' (MCTS) moved.")
        else:
            # 'x' (後手) はランダム
            action = random_action(state)
            print("Player 'x' (Random) moved.")

        # 次の状態の取得
        state = state.next(action)

        # 文字列表示
        print(state)
        print("====================="
)