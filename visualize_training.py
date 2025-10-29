"""
学習の進捗を可視化するスクリプト
訓練ログから損失の推移やモデルの強さをグラフ化
"""

import re
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def parse_training_log(log_file="training_log.txt"):
    """
    訓練ログファイルから情報を抽出
    """
    if not Path(log_file).exists():
        print(f"XX Log file '{log_file}' not found")
        print("   train_cycle.py の出力をファイルにリダイレクトしてください:")
        print("   python train_cycle.py > training_log.txt")
        return None
    
    # Windows PowerShellのTee-ObjectはUTF-16 LEを使用
    try:
        with open(log_file, 'r', encoding='utf-16-le') as f:
            content = f.read()
    except UnicodeDecodeError:
        # UTF-8でも試す
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
    
    data = {
        'cycles': [],
        'epochs': [],
        'losses': [],
        'learning_rates': [],
        'eval_scores': [],
        'vs_random': []
    }
    
    # 各サイクルの情報を抽出
    cycle_pattern = r'Train (\d+) ='
    cycles = re.findall(cycle_pattern, content)
    
    # エポックと損失を抽出
    epoch_pattern = r'Epoch (\d+)/\d+, Loss: ([\d.]+), LR: ([\d.]+)'
    matches = re.findall(epoch_pattern, content)
    
    current_cycle = 0
    for epoch_str, loss_str, lr_str in matches:
        epoch = int(epoch_str)
        if epoch == 1:
            current_cycle += 1
        
        data['cycles'].append(current_cycle)
        data['epochs'].append(epoch)
        data['losses'].append(float(loss_str))
        data['learning_rates'].append(float(lr_str))
    
    # 評価スコアを抽出
    eval_pattern = r'AveragePoint ([\d.]+)'
    eval_scores = re.findall(eval_pattern, content)
    data['eval_scores'] = [float(s) for s in eval_scores]
    
    # ランダム対戦結果を抽出
    random_pattern = r'VS_Random ([\d.]+)'
    vs_random = re.findall(random_pattern, content)
    data['vs_random'] = [float(s) for s in vs_random]
    
    return data

def plot_training_progress(data):
    """
    訓練の進捗をグラフ化
    """
    if data is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Ultimate Tic-Tac-Toe Training Progress', fontsize=16, fontweight='bold')
    
    # 1. 損失の推移
    ax1 = axes[0, 0]
    if data['losses']:
        # 全エポックの損失
        ax1.plot(data['losses'], alpha=0.6, linewidth=0.5, color='blue', label='Loss per Epoch')
        
        # サイクルごとの平均損失
        unique_cycles = sorted(set(data['cycles']))
        avg_losses = []
        for cycle in unique_cycles:
            cycle_losses = [data['losses'][i] for i, c in enumerate(data['cycles']) if c == cycle]
            avg_losses.append(np.mean(cycle_losses))
        
        ax1.plot(np.linspace(0, len(data['losses']), len(avg_losses)), 
                avg_losses, 'r-', linewidth=2, label='Average per Cycle', marker='o')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 学習率の推移
    ax2 = axes[0, 1]
    if data['learning_rates']:
        ax2.plot(data['learning_rates'], 'g-', linewidth=1)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    
    # 3. モデル評価スコア
    ax3 = axes[1, 0]
    if data['eval_scores']:
        cycles = range(1, len(data['eval_scores']) + 1)
        ax3.plot(cycles, data['eval_scores'], 'bo-', linewidth=2, markersize=8)
        ax3.axhline(y=0.5, color='r', linestyle='--', label='50% (Random)')
        ax3.set_xlabel('Training Cycle')
        ax3.set_ylabel('Win Rate vs Previous Best')
        ax3.set_title('Model Evaluation (Latest vs Best)')
        ax3.set_ylim([0, 1])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 改善を示すマーカー
        for i, score in enumerate(data['eval_scores']):
            if score > 0.5:
                ax3.plot(i+1, score, 'g*', markersize=15, label='Improved' if i == 0 else '')
    
    # 4. ランダムプレイヤー対戦結果
    ax4 = axes[1, 1]
    if data['vs_random']:
        cycles = range(1, len(data['vs_random']) + 1)
        win_rates = [1 - score for score in data['vs_random']]  # VS_Randomは負け率
        ax4.plot(cycles, win_rates, 'mo-', linewidth=2, markersize=8)
        ax4.set_xlabel('Training Cycle')
        ax4.set_ylabel('Win Rate vs Random')
        ax4.set_title('Performance vs Random Player')
        ax4.set_ylim([0, 1])
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    print("✅ グラフを 'training_progress.png' に保存しました")
    plt.show()

def print_summary(data):
    """
    訓練のサマリーを表示
    """
    if data is None:
        return
    
    print("\n" + "="*50)
    print("== Training Summary ==")
    print("="*50)
    
    if data['losses']:
        print(f"\n🎯 Loss:")
        print(f"   Initial: {data['losses'][0]:.4f}")
        print(f"   Final:   {data['losses'][-1]:.4f}")
        print(f"   Improvement: {data['losses'][0] - data['losses'][-1]:.4f}")
    
    if data['eval_scores']:
        print(f"\n>> Model Evaluation:")
        print(f"   Cycles Completed: {len(data['eval_scores'])}")
        print(f"   Best Win Rate: {max(data['eval_scores']):.2%}")
        improvements = sum(1 for score in data['eval_scores'] if score > 0.5)
        print(f"   Model Improvements: {improvements}/{len(data['eval_scores'])}")
    
    if data['vs_random']:
        print(f"\n>> vs Random Player:")
        win_rates = [1 - score for score in data['vs_random']]
        print(f"   Latest Win Rate: {win_rates[-1]:.2%}")
        print(f"   Average Win Rate: {np.mean(win_rates):.2%}")
    
    print("\n" + "="*50)

def main():
    """
    メイン処理
    """
    print("📈 Training Progress Visualization")
    print("="*50)
    
    # ログファイルのパス（カスタマイズ可能）
    log_file = "training_log.txt"
    
    # データを解析
    print(f"\n📂 Reading log file: {log_file}")
    data = parse_training_log(log_file)
    
    if data is None:
        print("\n💡 Tip: 次回の学習時にログを保存するには:")
        print("   python train_cycle.py | Tee-Object -FilePath training_log.txt")
        return
    
    # サマリーを表示
    print_summary(data)
    
    # グラフを作成
    print("\n>> Creating visualization...")
    plot_training_progress(data)

if __name__ == '__main__':
    main()
