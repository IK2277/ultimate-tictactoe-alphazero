"""
シンプルなリアルタイム学習モニター
ログファイルを読んで損失と評価スコアをグラフ表示
"""
import matplotlib.pyplot as plt
from pathlib import Path
import re
import time

class SimpleMonitor:
    def __init__(self, log_file="training.log"):
        self.log_file = log_file
        self.losses = []
        self.epochs = []
        self.eval_scores = []
        self.cycles = []
        self.last_size = 0  # ファイルサイズを追跡
        
        # グラフ設定
        plt.ion()  # インタラクティブモード
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle('Training Monitor', fontsize=14, fontweight='bold')
        
    def read_log(self):
        """ログファイルを読み込んでデータを抽出"""
        if not Path(self.log_file).exists():
            return
        
        # ファイルサイズをチェック（変更がない場合はスキップ）
        try:
            current_size = Path(self.log_file).stat().st_size
            if current_size == self.last_size:
                return  # 変更なし
            self.last_size = current_size
        except:
            return
        
        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except:
            return
        
        # データをクリアして再読み込み
        self.losses.clear()
        self.epochs.clear()
        self.eval_scores.clear()
        self.cycles.clear()
        
        epoch_count = 0
        current_cycle = 0
        
        for line in lines:
            # サイクル番号
            cycle_match = re.search(r'Train (\d+)', line)
            if cycle_match:
                current_cycle = int(cycle_match.group(1))
            
            # エポックと損失
            # 形式: "Epoch 1/100, Loss: 3.4748, LR: 0.001000"
            loss_match = re.search(r'Epoch \d+/\d+, Loss: ([\d.]+)', line)
            if loss_match:
                loss = float(loss_match.group(1))
                self.losses.append(loss)
                self.epochs.append(epoch_count)
                epoch_count += 1
            
            # 評価スコア
            # 形式: "AveragePoint 0.54"
            eval_match = re.search(r'AveragePoint\s+([\d.]+)', line)
            if eval_match:
                score = float(eval_match.group(1))
                if 0 <= score <= 1:
                    self.eval_scores.append(score)
                    self.cycles.append(current_cycle)
    
    def update_plot(self):
        """グラフを更新"""
        # 左側: 損失
        self.ax1.clear()
        if self.losses:
            self.ax1.plot(self.epochs, self.losses, 'b-', linewidth=1.5, alpha=0.7)
            self.ax1.set_xlabel('Epoch')
            self.ax1.set_ylabel('Loss')
            self.ax1.set_title('Training Loss')
            self.ax1.grid(True, alpha=0.3)
            
            # 最新の損失を表示
            latest_loss = self.losses[-1]
            self.ax1.text(0.02, 0.98, f'Latest: {latest_loss:.4f}',
                         transform=self.ax1.transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 右側: 評価スコア
        self.ax2.clear()
        if self.eval_scores:
            self.ax2.plot(self.cycles, self.eval_scores, 'go-', 
                         linewidth=2, markersize=8, markerfacecolor='cyan')
            self.ax2.axhline(y=0.5, color='r', linestyle='--', 
                           label='50% (Baseline)', linewidth=2)
            self.ax2.set_xlabel('Training Cycle')
            self.ax2.set_ylabel('Win Rate')
            self.ax2.set_title('Model Evaluation')
            self.ax2.set_ylim([0, 1])
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
            
            # 最新のスコアを表示
            latest_score = self.eval_scores[-1]
            self.ax2.text(0.02, 0.98, f'Latest: {latest_score:.2%}',
                         transform=self.ax2.transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def run(self):
        """モニタリングを開始"""
        print("=" * 50)
        print("シンプル学習モニター")
        print("=" * 50)
        print(f"ログファイル: {self.log_file}")
        print("Ctrl+C で終了")
        print()
        
        # 初回読み込み
        self.read_log()
        self.update_plot()
        
        try:
            update_count = 0
            while True:
                self.read_log()
                self.update_plot()
                
                # 進捗表示
                update_count += 1
                if update_count % 10 == 0:
                    print(f"更新回数: {update_count}, 損失数: {len(self.losses)}, 評価数: {len(self.eval_scores)}", flush=True)
                
                time.sleep(1)  # 1秒ごとに更新（より頻繁に）
        except KeyboardInterrupt:
            print("\n\n監視を終了します")
            plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
            print(f"グラフを training_progress.png に保存しました")
            plt.close()

if __name__ == '__main__':
    import sys
    
    # コマンドライン引数でログファイル名を指定可能
    log_file = sys.argv[1] if len(sys.argv) > 1 else "training.log"
    
    # ログファイルの存在確認
    if not Path(log_file).exists():
        print(f"エラー: {log_file} が見つかりません")
        print()
        print("使い方:")
        print("1. 別のターミナルで学習を開始:")
        print("   python train_cycle.py > training.log 2>&1")
        print()
        print("2. このスクリプトを実行:")
        print("   python monitor_simple.py")
        sys.exit(1)
    
    monitor = SimpleMonitor(log_file)
    monitor.run()
