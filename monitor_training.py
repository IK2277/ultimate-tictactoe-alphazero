"""
リアルタイムで訓練ログを記録し、進捗を可視化
TensorBoard風のシンプルな可視化
"""

import matplotlib
matplotlib.use('TkAgg')  # Windowsでグラフを表示するため
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
import time
from pathlib import Path
import re

class TrainingMonitor:
    """
    訓練の進捗をリアルタイムで監視・可視化
    """
    def __init__(self, log_file="training_log.txt", max_points=1000):
        self.log_file = log_file
        self.max_points = max_points
        
        # データ保存用
        self.epochs = deque(maxlen=max_points)
        self.losses = deque(maxlen=max_points)
        self.lrs = deque(maxlen=max_points)
        self.eval_scores = []
        self.cycles = []
        # セルフプレイ進捗
        self.total_games = None
        self.games_per_worker = None
        self.worker_progress = {}
        self.selfplay_progress = 0.0
        
        # ファイルの最後の位置
        self.last_position = 0
        self.last_mtime = 0
        self.first_read = True  # 初回読み込みフラグ
        
        # グラフの初期化
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 9))
        self.fig.suptitle('Real-time Training Monitor', 
                         fontsize=16, fontweight='bold')
        
    def read_new_lines(self):
        """
        ログファイルから新しい行を読み込む
        初回はファイル全体を読み込み、2回目以降は追加分のみ
        """
        if not Path(self.log_file).exists():
            return []
        
        # エンコーディングの自動検出
        encodings = ['utf-8', 'utf-16-le', 'cp932']
        p = Path(self.log_file)
        try:
            stat = p.stat()
            # ファイルがローテーション/短縮された場合に先頭から読み直す
            if stat.st_size < self.last_position:
                self.last_position = 0
                self.first_read = True
            self.last_mtime = stat.st_mtime
        except Exception:
            pass
        
        for encoding in encodings:
            try:
                with open(self.log_file, 'r', encoding=encoding, errors='ignore') as f:
                    # 初回は全体を読み込む
                    if self.first_read:
                        new_lines = f.readlines()
                        self.last_position = f.tell()
                        self.first_read = False
                        print(f">> Initial read: {len(new_lines)} lines loaded", flush=True)
                    else:
                        # 2回目以降は追加分のみ
                        f.seek(self.last_position)
                        new_lines = f.readlines()
                        self.last_position = f.tell()
                return new_lines
            except (UnicodeDecodeError, IOError):
                continue
        
        # すべて失敗した場合
        return []
    
    def parse_lines(self, lines):
        """
        新しい行からデータを抽出
        """
        for line in lines:
            # エポックと損失（新しい形式に対応）
            # 形式: "Epoch 1/100, Loss: 3.4748, LR: 0.001000"
            epoch_match = re.search(r'Epoch (\d+)/\d+, Loss: ([\d.]+), LR: ([\d.e-]+)', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                loss = float(epoch_match.group(2))
                lr = float(epoch_match.group(3))
                
                total_epoch = len(self.epochs)
                self.epochs.append(total_epoch)
                self.losses.append(loss)
                self.lrs.append(lr)
            
            # 評価スコア（新しい形式に対応）
            # 形式: "AveragePoint 0.54" または "Average Score: 0.54"
            eval_match = re.search(r'(?:AveragePoint|Average[:\s]+(?:Score)?)[:\s]+([\d.]+)', line)
            if eval_match:
                score = float(eval_match.group(1))
                # スコアが0-1の範囲にあることを確認
                if 0 <= score <= 1:
                    self.eval_scores.append(score)
                    self.cycles.append(len(self.eval_scores))
            
            # サイクル開始（新しい形式に対応）
            # 形式: "Train 0 ====" または "Train 0"
            cycle_match = re.search(r'Train (\d+)', line)
            if cycle_match:
                cycle_num = int(cycle_match.group(1))
                print(f">> Cycle {cycle_num} started")

            # セルフプレイ総ゲーム数
            # 形式: ">> Total games: 500, Games per worker: 62"
            games_info = re.search(r'Total games:\s*(\d+),\s*Games per worker:\s*(\d+)', line)
            if games_info:
                self.total_games = int(games_info.group(1))
                self.games_per_worker = int(games_info.group(2))
                print(f">> Detected: Total games={self.total_games}, Per worker={self.games_per_worker}", flush=True)

            # ワーカー進捗
            # 形式: "Worker 3: 20/62 games completed"
            worker_prog = re.search(r'Worker\s+(\d+):\s+(\d+)/(\d+)\s+games completed', line)
            if worker_prog:
                wid = int(worker_prog.group(1))
                done = int(worker_prog.group(2))
                total = int(worker_prog.group(3))
                self.worker_progress[wid] = (done, total)
                # 全体進捗の推定
                total_done = sum(d for d, t in self.worker_progress.values())
                total_all = sum(t for d, t in self.worker_progress.values())
                if total_all > 0:
                    self.selfplay_progress = total_done / total_all
    
    def update_plots(self, frame):
        """
        グラフを更新
        """
        # 新しいデータを読み込む
        new_lines = self.read_new_lines()
        if new_lines:
            self.parse_lines(new_lines)
        
        # グラフをクリア
        for ax in self.axes.flat:
            ax.clear()
        
        # 1. 損失の推移
        ax1 = self.axes[0, 0]
        if self.losses:
            ax1.plot(list(self.epochs), list(self.losses), 'b-', linewidth=1.5, alpha=0.7)
            
            # 移動平均
            if len(self.losses) >= 10:
                window = 10
                moving_avg = np.convolve(self.losses, np.ones(window)/window, mode='valid')
                ax1.plot(list(self.epochs)[window-1:], moving_avg, 'r-', 
                        linewidth=2, label=f'Moving Avg ({window})')
            
            ax1.set_xlabel('Epoch', fontsize=10)
            ax1.set_ylabel('Loss', fontsize=10)
            ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 最新の値を表示
            if self.losses:
                latest_loss = self.losses[-1]
                ax1.text(0.02, 0.98, f'Latest: {latest_loss:.4f}',
                        transform=ax1.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', 
                        facecolor='wheat', alpha=0.5))
        
        # 2. 学習率
        ax2 = self.axes[0, 1]
        if self.lrs:
            ax2.plot(list(self.epochs), list(self.lrs), 'g-', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=10)
            ax2.set_ylabel('Learning Rate', fontsize=10)
            ax2.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        # 3. 評価スコア
        ax3 = self.axes[1, 0]
        if self.eval_scores:
            ax3.plot(self.cycles, self.eval_scores, 'mo-', 
                    linewidth=2, markersize=10, markerfacecolor='cyan')
            ax3.axhline(y=0.5, color='r', linestyle='--', 
                       label='50% (Baseline)', linewidth=2)
            ax3.set_xlabel('Training Cycle', fontsize=10)
            ax3.set_ylabel('Win Rate', fontsize=10)
            ax3.set_title('Model Evaluation Score', fontsize=12, fontweight='bold')
            ax3.set_ylim([0, 1])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 改善を強調
            for i, score in enumerate(self.eval_scores):
                if score > 0.5:
                    ax3.plot(self.cycles[i], score, 'g*', markersize=20, 
                            markeredgecolor='darkgreen', markeredgewidth=2)
        
        # 4. 統計情報
        ax4 = self.axes[1, 1]
        ax4.axis('off')
        
        stats_text = "== Training Statistics ==\n"
        stats_text += "=" * 30 + "\n\n"
        
        if self.losses:
            stats_text += f">> Current Loss: {self.losses[-1]:.4f}\n"
            stats_text += f">> Epochs Completed: {len(self.losses)}\n"
            if len(self.losses) > 1:
                improvement = self.losses[0] - self.losses[-1]
                stats_text += f">> Loss Improvement: {improvement:.4f}\n"
        
        stats_text += "\n"
        # セルフプレイ進捗
        stats_text += ">> Self-play Status:\n"
        if self.total_games:
            stats_text += f"   Total games: {self.total_games}\n"
        if self.worker_progress:
            stats_text += f"   Overall: {self.selfplay_progress*100:.1f}%\n"
            # 最大3ワーカーまで詳細表示
            shown = 0
            for wid in sorted(self.worker_progress.keys()):
                d, t = self.worker_progress[wid]
                stats_text += f"   Worker {wid}: {d}/{t}\n"
                shown += 1
                if shown >= 3:
                    break
        else:
            stats_text += "   (waiting for self-play data...)\n"
        stats_text += "\n"
        
        if self.eval_scores:
            stats_text += f">> Cycles Completed: {len(self.eval_scores)}\n"
            stats_text += f">> Best Score: {max(self.eval_scores):.2%}\n"
            improvements = sum(1 for s in self.eval_scores if s > 0.5)
            stats_text += f">> Model Improvements: {improvements}\n"
        
        stats_text += "\n"
        stats_text += f">> Last Update: {time.strftime('%H:%M:%S')}\n"
        if self.last_mtime:
            stats_text += f">> Log mtime: {time.strftime('%H:%M:%S', time.localtime(self.last_mtime))}\n"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
    
    def start(self):
        """
        モニタリングを開始
        """
        print(">> Starting real-time training monitor...", flush=True)
        print(f">> Monitoring: {self.log_file}", flush=True)
        print(">> Press Ctrl+C to stop", flush=True)
        print()
        
        # アニメーションを開始（2秒ごとに更新）
        self.ani = FuncAnimation(self.fig, self.update_plots, 
                          interval=2000, cache_frame_data=False)
        plt.show()

def main():
    """
    メイン処理
    """
    import sys
    
    log_file = "training_log.txt"
    
    # コマンドライン引数でログファイルを指定可能
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    # ログファイルの存在確認
    if not Path(log_file).exists():
        print(f"XX Log file '{log_file}' not found")
        print("\n>> Usage:")
        print("   1. Start training in another terminal:")
        print("      python -u train_cycle.py *>&1 | Tee-Object -FilePath training_log.txt")
        print("   2. Run this script:")
        print("      python monitor_training.py")
        print("\n>> Note: Make sure training_log.txt exists before starting the monitor")
        return
    
    # モニターを開始
    monitor = TrainingMonitor(log_file)
    
    try:
        monitor.start()
    except KeyboardInterrupt:
        print("\n\n>> Monitoring stopped")
        print(">> Final plot saved to 'training_progress.png'")
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
