# ====================
# 人とAIの対戦 (Ultimate-tic-tac-toe版)
# ====================

# パッケージのインポート
from game import State
from pv_mcts import pv_mcts_action
from dual_network import DualNetwork, device
from pathlib import Path
from threading import Thread
import tkinter as tk
import sys
import torch

# ベストプレイヤーのモデルの読み込み
model_path = './model/best.pth'
if not Path(model_path).exists():
    print(f"エラー: モデルファイルが見つかりません: {model_path}")
    print("先に train_cycle.py を実行してモデルを生成してください。")
    sys.exit()
    
model = DualNetwork().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# --- 定数の定義 ---
CANVAS_SIZE = 450 # キャンバス全体のサイズ
BOARD_SIZE = 450  # 盤面全体のサイズ
SMALL_BOARD_SIZE = BOARD_SIZE / 3 # 150
CELL_SIZE = SMALL_BOARD_SIZE / 3  # 50
PIECE_MARGIN = 5 # 石の描画マージン

# ゲームUIの定義
class GameUI(tk.Frame):
    # 初期化
    def __init__(self, master=None, model=None):
        tk.Frame.__init__(self, master)
        self.master.title('Ultimate Tic-Tac-Toe')

        # ゲーム状態の生成
        self.state = State()

        # PV MCTSで行動選択を行う関数の生成
        self.next_action = pv_mcts_action(model, 0.0)

        # キャンバスの生成 (9x9)
        self.c = tk.Canvas(self, width = CANVAS_SIZE, height = CANVAS_SIZE, highlightthickness = 0)
        self.c.bind('<Button-1>', self.turn_of_human)
        self.c.pack()

        # 描画の更新
        self.on_draw()

    # 人間のターン
    def turn_of_human(self, event):
        # ゲーム終了時
        if self.state.is_done():
            self.state = State()
            self.on_draw()
            return

        # AIのターン (先手でない) 時
        if not self.state.is_first_player():
            return

        # 1. クリック位置を行動(0-80)に変換
        x, y = event.x, event.y
        if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
            return # キャンバス外

        main_col = int(x / SMALL_BOARD_SIZE) # メインボードの列 (0-2)
        main_row = int(y / SMALL_BOARD_SIZE) # メインボードの行 (0-2)
        board_idx = main_row * 3 + main_col   # 小盤面のインデックス (0-8)

        cell_col = int((x % SMALL_BOARD_SIZE) / CELL_SIZE) # 小盤面内の列 (0-2)
        cell_row = int((y % SMALL_BOARD_SIZE) / CELL_SIZE) # 小盤面内の行 (0-2)
        cell_idx = cell_row * 3 + cell_col    # 小盤面内のセル (0-8)

        action = board_idx * 9 + cell_idx # 最終的なアクション (0-80)

        # 2. 合法手でない時
        if not (action in self.state.legal_actions()):
            return

        # 3. 次の状態の取得
        self.state = self.state.next(action)
        self.on_draw()

        # 4. AIのターン
        self.master.after(1, self.turn_of_ai)

    # AIのターン
    def turn_of_ai(self):
        # ゲーム終了時
        if self.state.is_done():
            self.on_draw() # 最終盤面を描画
            return

        # 行動の取得
        action = self.next_action(self.state)

        # 次の状態の取得
        self.state = self.state.next(action)
        self.on_draw()

    # 石の描画 (board_idx: 0-8, cell_idx: 0-8, first_player: bool)
    def draw_piece(self, board_idx, cell_idx, first_player):
        main_col = board_idx % 3
        main_row = board_idx // 3
        cell_col = cell_idx % 3
        cell_row = cell_idx // 3

        # セルの左上座標
        x0 = (main_col * SMALL_BOARD_SIZE) + (cell_col * CELL_SIZE)
        y0 = (main_row * SMALL_BOARD_SIZE) + (cell_row * CELL_SIZE)
        
        # マージンを適用
        x1, y1 = x0 + PIECE_MARGIN, y0 + PIECE_MARGIN
        x2, y2 = x0 + CELL_SIZE - PIECE_MARGIN, y0 + CELL_SIZE - PIECE_MARGIN
        
        color = '#FFFFFF' if first_player else '#5D5D5D' # O: 白, X: グレー

        if first_player: # 'o'
            self.c.create_oval(x1, y1, x2, y2, width = 4.0, outline = color)
        else: # 'x'
            self.c.create_line(x1, y1, x2, y2, width = 4.0, fill = color)
            self.c.create_line(x1, y2, x2, y1, width = 4.0, fill = color)

    # メインボードの勝者を描画 (board_idx: 0-8, first_player: bool)
    def draw_main_winner(self, board_idx, first_player):
        main_col = board_idx % 3
        main_row = board_idx // 3
        
        # 小盤面の左上座標
        x0 = main_col * SMALL_BOARD_SIZE
        y0 = main_row * SMALL_BOARD_SIZE
        # 小盤面の右下座標
        x1 = x0 + SMALL_BOARD_SIZE
        y1 = y0 + SMALL_BOARD_SIZE
        
        color = '#FFFFFF' if first_player else '#5D5D5D'
        
        # 盤面全体を半透明の色で覆う
        # (Tkinterは半透明をサポートしないため、代わりに大きな記号を描画)
        
        if first_player: # 'o'
            self.c.create_oval(x0+10, y0+10, x1-10, y1-10, width = 10.0, outline = color)
        else: # 'x'
            self.c.create_line(x0+10, y0+10, x1-10, y1-10, width = 10.0, fill = color)
            self.c.create_line(x0+10, y1-10, x1-10, y0+10, width = 10.0, fill = color)
            
    # 描画の更新
    def on_draw(self):
        self.c.delete('all')
        
        # 背景
        BG_COLOR = '#00A0FF' # 青
        LINE_COLOR = '#0077BB' # 濃い青
        ACTIVE_COLOR = '#00C0FF' # 明るい青 (ハイライト)
        
        self.c.create_rectangle(0, 0, CANVAS_SIZE, CANVAS_SIZE, width = 0.0, fill = BG_COLOR)

        # --- 1. アクティブボードのハイライト ---
        if not self.state.is_done():
            # 合法手からプレイ可能な盤面インデックスを取得
            legal_boards = set()
            for action in self.state.legal_actions():
                legal_boards.add(action // 9)

            for board_idx in legal_boards:
                main_col = board_idx % 3
                main_row = board_idx // 3
                x0 = main_col * SMALL_BOARD_SIZE
                y0 = main_row * SMALL_BOARD_SIZE
                self.c.create_rectangle(x0, y0, x0 + SMALL_BOARD_SIZE, y0 + SMALL_BOARD_SIZE, 
                                        width = 0.0, fill = ACTIVE_COLOR)

        # --- 2. グリッド線の描画 ---
        for i in range(1, 9):
            width = 5.0 if (i % 3 == 0) else 2.0 # 3の倍数(小盤面の境界)は太線
            
            # 縦線
            x = i * CELL_SIZE
            self.c.create_line(x, 0, x, CANVAS_SIZE, width = width, fill = LINE_COLOR)
            
            # 横線
            y = i * CELL_SIZE
            self.c.create_line(0, y, CANVAS_SIZE, y, width = width, fill = LINE_COLOR)

        # --- 3. 石の描画 ---
        # ox[0] = 現プレイヤー, ox[1] = 相手
        is_first = self.state.is_first_player()
        
        for board_idx in range(9):
            # メインボードが決着しているか
            main_win_p1 = self.state.main_board_pieces[board_idx] == 1
            main_win_p2 = self.state.main_board_enemy_pieces[board_idx] == 1
            
            if main_win_p1 and main_win_p2: # 引き分け
                # (ここでは描画しないが、必要なら 'D' などを描画)
                pass
            elif main_win_p1: # 現プレイヤーの勝利
                self.draw_main_winner(board_idx, is_first)
            elif main_win_p2: # 相手プレイヤーの勝利
                self.draw_main_winner(board_idx, not is_first)
            else:
                # 盤面が決着していない場合のみ、中の石を描画
                for cell_idx in range(9):
                    if self.state.pieces[board_idx][cell_idx] == 1:
                        self.draw_piece(board_idx, cell_idx, is_first)
                    if self.state.enemy_pieces[board_idx][cell_idx] == 1:
                        self.draw_piece(board_idx, cell_idx, not is_first)
        
        # --- 4. ゲーム終了時の勝者表示 ---
        if self.state.is_done():
            winner_text = ""
            if self.state.is_lose():
                # is_lose() は「現プレイヤーの負け」
                winner = 'X' if self.state.is_first_player() else 'O'
                winner_text = f"Winner: {winner}"
            else:
                winner_text = "Draw"
            
            # 画面中央にテキストを表示
            self.c.create_rectangle(CANVAS_SIZE/2 - 100, CANVAS_SIZE/2 - 30, 
                                    CANVAS_SIZE/2 + 100, CANVAS_SIZE/2 + 30, 
                                    fill='black', outline='white', width=2)
            self.c.create_text(CANVAS_SIZE/2, CANVAS_SIZE/2, text=winner_text, 
                               fill='white', font=("Arial", 24, "bold"))

# ゲームUIの実行
f = GameUI(model=model)
f.pack()
f.mainloop()