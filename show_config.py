#!/usr/bin/env python
"""
初期設定を表示するスクリプト
"""
import torch
from auto_tune import get_optimal_batch_size, get_optimal_num_workers
from train_network import get_dynamic_learning_rate, RN_EPOCHS, BATCH_SIZE
from self_play_cpp import get_dynamic_game_count
from pv_mcts import get_dynamic_pv_count
import os
from pathlib import Path

print('\n' + '='*60)
print('🎮 システム情報')
print('='*60)
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'VRAM: {mem_gb:.1f} GB')
    print(f'CUDA Version: {torch.version.cuda}')
else:
    print('GPU: CPU Only')
print(f'PyTorch Version: {torch.__version__}')

print('\n' + '='*60)
print('⚙️  トレーニング設定 (Cycle 0)')
print('='*60)
print(f'学習率 (LR): {get_dynamic_learning_rate(0)} (AlphaZero準拠: OLIVAW)')
print(f'オプティマイザ: SGD (Momentum 0.9)')
print(f'学習エポック数: {RN_EPOCHS}')
print(f'トレーニングバッチサイズ: {BATCH_SIZE}')
print(f'L2正則化: 1e-4 (weight_decay)')
print(f'Dropout: 0.3 (Policy/Value Head)')

print('\n' + '='*60)
print('🎲 MCTS & Self-Play設定 (Cycle 0)')
print('='*60)
print(f'MCTSバッチサイズ: {get_optimal_batch_size()}')
print(f'MCTS探索回数: {get_dynamic_pv_count(0)}')
print(f'セルフプレイゲーム数: {get_dynamic_game_count(0)}')
print(f'並列ワーカー数: {get_optimal_num_workers(aggressive=True)} (aggressive mode)')
print(f'温度スケジューリング (AlphaZero準拠):')
print(f'  訓練時: 0-29手 τ=1.0, 30手以降 τ→0 (決定論的)')
print(f'  評価時: τ=0.0 (最善手のみ)')

print('\n' + '='*60)
print('📊 学習率スケジュール')
print('='*60)
print('【サイクルベース】')
for cycle in [0, 10, 20, 30]:
    lr = get_dynamic_learning_rate(cycle)
    games = get_dynamic_game_count(cycle)
    mcts = get_dynamic_pv_count(cycle)
    print(f'  Cycle {cycle:2d}+: LR={lr:7.5f}, Games={games:4d}, MCTS={mcts:3d}')

print('\n【エポックベース (各サイクル内)】')
print('  Epoch  1-69:  ×1.0 (フル学習率)')
print('  Epoch 70-89:  ×0.2 (1/5に減衰)')
print('  Epoch 90-100: ×0.1 (1/10に減衰)')

print('\n' + '='*60)
print('🎯 評価設定')
print('='*60)
print('評価試合数: 100試合')
print('対戦相手: ランダムプレイヤー')
print('更新閾値: 55% (勝率)')
print('評価温度: τ=0.0 (決定論的、AlphaZero準拠)')

print('\n' + '='*60)
print('✅ ファイル状態')
print('='*60)
data_files = len(list(Path('data').glob('*.history')))
model_files = len(list(Path('model').glob('*.pth')))
checkpoint = os.path.exists('training_checkpoint.json')
log_exists = os.path.exists('training.log')

print(f'データファイル (.history): {data_files}')
print(f'モデルファイル (.pth): {model_files}')
print(f'チェックポイント: {"存在" if checkpoint else "なし (新規開始)"}')
print(f'ログファイル: {"存在" if log_exists else "なし (新規作成)"}')

print('\n' + '='*60)
print('🚀 準備完了！学習を開始できます')
print('='*60)
print('【推奨】ログ付きで実行:')
print('  python train_with_log.py')
print()
print('【シンプル】直接実行:')
print('  python train_cycle.py')
print()
print('【モニタリング】別ターミナルで:')
print('  python monitor_simple.py  # グラフ表示')
print('  python monitor_gpu.py     # GPU監視')
print('='*60 + '\n')
