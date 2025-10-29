"""
リアルタイムで訓練ログを記録し、進捗を可視化
TensorBoard風のシンプルな可視化
"""

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
        
        # ファイルの最後の位置
        self.last_position = 0
        
        # グラフの初期化
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 9))
        self.fig.suptitle('Real-time Training Monitor', 
                         fontsize=16, fontweight='bold')
        
    def read_new_lines(self):
        """
        ログファイルから新しい行を読み込む
        """
        if not Path(self.log_file).exists():
            return []
        
        # Windows PowerShellのTee-ObjectはUTF-16 LEを使用
        try:
            with open(self.log_file, 'r', encoding='utf-16-le') as f:
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()
            return new_lines
        except UnicodeDecodeError:
            # UTF-8でも試す
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    f.seek(self.last_position)
                    new_lines = f.readlines()
                    self.last_position = f.tell()
                return new_lines
            except:
                return []
    
    def parse_lines(self, lines):
        """
        新しい行からデータを抽出
        """
        for line in lines:
            # エポックと損失
            epoch_match = re.search(r'Epoch (\d+)/\d+, Loss: ([\d.]+), LR: ([\d.]+)', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                loss = float(epoch_match.group(2))
                lr = float(epoch_match.group(3))
                
                total_epoch = len(self.epochs)
                self.epochs.append(total_epoch)
                self.losses.append(loss)
                self.lrs.append(lr)
            
            # 評価スコア
            eval_match = re.search(r'AveragePoint ([\d.]+)', line)
            if eval_match:
                score = float(eval_match.group(1))
                self.eval_scores.append(score)
                self.cycles.append(len(self.eval_scores))
            
            # サイクル開始
            cycle_match = re.search(r'Train (\d+) =', line)
            if cycle_match:
                cycle_num = int(cycle_match.group(1))
                print(f">> Cycle {cycle_num} started")
    
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
        
        if self.eval_scores:
            stats_text += f">> Cycles Completed: {len(self.eval_scores)}\n"
            stats_text += f">> Best Score: {max(self.eval_scores):.2%}\n"
            improvements = sum(1 for s in self.eval_scores if s > 0.5)
            stats_text += f">> Model Improvements: {improvements}\n"
        
        stats_text += "\n"
        stats_text += f">> Last Update: {time.strftime('%H:%M:%S')}\n"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
    
    def start(self):
        """
        モニタリングを開始
        """
        print(">> Starting real-time training monitor...")
        print(f">> Monitoring: {self.log_file}")
        print(">> Press Ctrl+C to stop")
        print()
        
        # アニメーションを開始（2秒ごとに更新）
        ani = FuncAnimation(self.fig, self.update_plots, 
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
        print("      python train_cycle.py | Tee-Object -FilePath training_log.txt")
        print("   2. Run this script:")
        print("      python monitor_training.py")
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
