# Ultimate Tic-Tac-Toe AlphaZero

AlphaZeroアルゴリズムを使用したUltimate Tic-Tac-Toeの強化学習実装（PyTorch + C++最適化版）

## 📋 目次

- [概要](#概要)
- [主な特徴](#主な特徴)
- [システム要件](#システム要件)
- [クイックスタート](#クイックスタート)
- [トレーニング](#トレーニング)
- [パフォーマンス設定](#パフォーマンス設定)
- [ファイル構成](#ファイル構成)
- [トラブルシューティング](#トラブルシューティング)
- [技術詳細](#技術詳細)

---

## 概要

AlphaZeroスタイルの自己対戦型強化学習を用いて、Ultimate Tic-Tac-Toe（究極の三目並べ）をプレイするAIを訓練します。

### 主な特徴

- ✅ **ResNetベースのDual Network** (Policy + Value Head、1.2M parameters)
- ✅ **Monte Carlo Tree Search (MCTS)** による探索
- ✅ **C++実装による高速化** (ゲームロジック20x、MCTS 50x高速)
- ✅ **マルチプロセス並列化** (8ワーカー、5-6倍高速)
- ✅ **GPU対応** (NVIDIA CUDA 12.1+、PyTorch最適化)
- ✅ **動的パラメータ調整** (MCTS探索回数、学習率、ゲーム数)
- ✅ **リアルタイム訓練監視** (matplotlib可視化)
- ✅ **チェックポイント機能** (いつでも中断・再開可能)

---

## システム要件

### ソフトウェア

- **Python** 3.11以降
- **CUDA** 12.1以降 (GPU使用時、推奨)
- **Visual Studio 2019以降** (Windows、C++拡張ビルド用)
- **Git** (バージョン管理)

### ハードウェア（推奨）

- **GPU**: NVIDIA GeForce RTX 4070 Ti以上 (12GB VRAM)
- **CPU**: 12コア以上
- **メモリ**: 32GB以上
- **ストレージ**: 5GB以上の空き容量

---

## クイックスタート

### 1. 自動セットアップ（Windows推奨）

```powershell
# PowerShellで実行
.\setup.ps1
```

このスクリプトは以下を自動実行します：
- CUDA対応PyTorchのインストール
- 必要なパッケージのインストール
- C++拡張のビルド
- GPUの動作確認
- 初期モデルの作成

### 2. トレーニング開始

```powershell
# 並列トレーニング（8ワーカー、推奨）
python -u train_cycle.py *>&1 | Tee-Object -FilePath training_log.txt

# リアルタイムモニタリング（別ターミナル）
python monitor_training.py
```

### 3. 人間vs AI対戦

```powershell
python human_play.py
```

---

## トレーニング

### 基本コマンド

```powershell
# トレーニング開始（ログ出力付き）
python -u train_cycle.py *>&1 | Tee-Object -FilePath training_log.txt
```

### 動的パラメータ

トレーニングは自動的にパラメータを調整します：

| サイクル | ゲーム数 | MCTS探索 | 学習率 | 推定時間/サイクル |
|---------|---------|----------|--------|------------------|
| 0-9 | 500 | 100 | 0.001 | 約1分 |
| 10-19 | 1000 | 200 | 0.0005 | 約4分 |
| 20-29 | 1500 | 400 | 0.0002 | 約11分 |
| 30+ | 2000 | 800 | 0.0001 | 約29分 |

**50サイクルの総時間**: 約17時間（8ワーカー）

### 中断と再開

- **中断**: `Ctrl+C` で安全に停止
- **再開**: 再度 `python train_cycle.py` で自動的にチェックポイントから再開

### 進捗確認

```powershell
# リアルタイムモニタリング
python monitor_training.py

# トレーニング後の可視化
python visualize_training.py
```

---

## パフォーマンス設定

### ワーカー数の設定

`train_cycle.py`の先頭で設定：

```python
# ========================================
# 並列化設定
# ========================================
USE_PARALLEL = True  # 並列化ON/OFF

# ワーカー数の設定
WORKER_MODE = "aggressive"  # 推奨（8ワーカー）
# WORKER_MODE = "auto"      # 保守的（4ワーカー）
# WORKER_MODE = 10          # 手動指定（10ワーカー）
```

### パフォーマンス比較

| 設定 | ワーカー数 | Cycle 0-9 | 50サイクル | 高速化 |
|------|-----------|-----------|-----------|--------|
| シリアル | 1 | 3.6分 | 93時間 | 1.0x |
| 保守的 | 4 | 約1分 | 約26時間 | 3.6x |
| **積極的** | **8** | **約40秒** | **約17時間** | **5.4x** ⭐ |
| 最大 | 10 | 約35秒 | 約15時間 | 6.2x |

### システムリソース確認

```powershell
python auto_tune.py
```

出力例：
```
=== System Resources ===
CPU Cores: 12 physical, 20 logical
RAM: 31.8GB total, 16.3GB available
GPU: NVIDIA GeForce RTX 4070 Ti
VRAM: 12.0GB total

>> Recommended settings:
   MCTS Batch Size: 32
   Parallel Workers (Conservative): 4
   Parallel Workers (Aggressive): 8
```

---

## ファイル構成

### コアファイル

```
train_cycle.py           # メイントレーニングループ
dual_network.py          # ニューラルネットワーク定義
game.py                  # Ultimate Tic-Tac-Toeゲームロジック
pv_mcts.py              # MCTS実装（Python）
pv_mcts_cpp.py          # MCTS実装（C++ラッパー）
self_play_cpp.py        # セルフプレイ（C++バックエンド）
self_play_parallel.py   # セルフプレイ（並列版）
train_network.py        # ネットワークトレーニング
evaluate_network.py     # ネットワーク評価
evaluate_best_player.py # ベストプレイヤー評価
```

### ツール

```
auto_tune.py            # システムリソース自動調整
optimize_inference.py   # PyTorch推論最適化
compare_workers.py      # ワーカー数比較ベンチマーク
test_speed.py          # 速度テスト
monitor_training.py    # リアルタイム進捗監視
visualize_training.py  # トレーニング結果可視化
human_play.py          # 人間vs AI対戦
check_gpu.py           # GPU動作確認
```

### C++拡張

```
cpp/
  uttt_game.cpp/h      # ゲームロジック（C++）
  uttt_mcts.cpp/h      # MCTS実装（C++）
  python_bindings.cpp  # Pythonバインディング
  setup.py             # ビルドスクリプト
build_cpp.ps1          # C++拡張ビルドスクリプト
```

### データ・モデル

```
model/
  best.pth             # ベストモデル（PyTorch）
data/
  *.history            # トレーニングデータ
training_checkpoint.json  # チェックポイント
training_log.txt          # トレーニングログ
training_progress.png     # 進捗グラフ
```

---

## トラブルシューティング

### GPU関連

**問題**: CUDA Out of Memory
```python
# train_cycle.py または self_play_parallel.py で調整
WORKER_MODE = 4  # ワーカー数を減らす

# または MCTS_BATCH_SIZE を減らす（self_play_parallel.py）
MCTS_BATCH_SIZE = 16  # 32から16に
```

**問題**: GPUが認識されない
```powershell
python check_gpu.py  # GPU確認
# CUDA 12.1+ がインストールされているか確認
```

### メモリ不足

**問題**: メモリ不足エラー
```python
# ワーカー数を減らす
WORKER_MODE = "auto"  # 4ワーカー

# またはシリアル版を使用
USE_PARALLEL = False
```

### プロセス起動エラー（Windows）

**問題**: マルチプロセスエラー
- `if __name__ == '__main__':` ブロック内でコードを実行
- `mp.freeze_support()` が呼ばれているか確認

### トレーニングが遅い

1. **ワーカー数を増やす**:
   ```python
   WORKER_MODE = "aggressive"  # 8ワーカー
   # または
   WORKER_MODE = 10  # 10ワーカー
   ```

2. **システムリソースを確認**:
   ```powershell
   python auto_tune.py
   ```

3. **他のアプリケーションを閉じる**

### C++拡張のビルドエラー

```powershell
# C++拡張を再ビルド
.\build_cpp.ps1
```

Visual Studio 2019以降がインストールされているか確認してください。

---

## 技術詳細

### アーキテクチャ

#### ニューラルネットワーク
- **タイプ**: ResNet with Dual Head
- **入力**: 9×9×3 (盤面の状態)
- **出力**: 
  - Policy Head: 81次元（各マスの確率）
  - Value Head: 1次元（勝率予測）
- **パラメータ数**: 約1.2M
- **ブロック数**: 16 Residual Blocks
- **フィルタ数**: 128

#### MCTS（Monte Carlo Tree Search）
- **シミュレーション回数**: 100→800（動的調整）
- **バッチサイズ**: 32（GPU最適化）
- **探索戦略**: PUCT（Predictor + Upper Confidence Bound）
- **温度パラメータ**: 1.0（セルフプレイ時）

#### トレーニング
- **オプティマイザ**: AdamW
- **学習率**: 0.001→0.0001（動的調整）
- **バッチサイズ**: 128
- **エポック数**: 100/サイクル
- **損失関数**: 
  - Policy: カスタムクロスエントロピー
  - Value: MSE

### パフォーマンス最適化

#### C++バックエンド
- **ゲームロジック**: 20倍高速化
- **MCTS**: 50倍高速化
- **言語**: C++17
- **ライブラリ**: pybind11

#### GPU最適化
- **TensorFloat-32 (TF32)**: 有効化
- **cuDNN Benchmark**: 有効化
- **torch.compile()**: 動的最適化（PyTorch 2.0+）

#### マルチプロセス並列化
- **方式**: spawn（Windows対応）
- **ワーカー数**: 4-10（自動調整）
- **通信**: Queue（IPC）

### 動的パラメータスケーリング

サイクル数に応じて自動調整：

```python
# MCTS探索回数
def get_dynamic_pv_count(cycle):
    if cycle < 10: return 100
    elif cycle < 20: return 200
    elif cycle < 30: return 400
    else: return 800

# 学習率
def get_dynamic_learning_rate(cycle):
    if cycle < 10: return 0.001
    elif cycle < 20: return 0.0005
    elif cycle < 30: return 0.0002
    else: return 0.0001

# ゲーム数
def get_dynamic_game_count(cycle):
    if cycle < 10: return 500
    elif cycle < 20: return 1000
    elif cycle < 30: return 1500
    else: return 2000
```

### ベンチマーク結果

#### 実測値（RTX 4070 Ti、12コアCPU、32GB RAM）

**シリアル版（C++）**:
- 速度: 約2.3ゲーム/秒
- Cycle 0-9: 約3.6分

**並列版（8ワーカー）**:
- 速度: 約12ゲーム/秒
- Cycle 0-9: 約40秒
- **高速化**: 5.4倍

**総合**:
- バッチサイズ増加: 1.7倍
- PyTorch最適化: 1.25倍
- 並列化: 2.5倍
- **合計**: 約5倍の高速化

---

## ライセンス

このプロジェクトは教育目的で作成されています。

## 参考文献

- [AlphaZero論文](https://arxiv.org/abs/1712.01815)
- [AlphaGo Zero論文](https://www.nature.com/articles/nature24270)
- [PyTorch公式ドキュメント](https://pytorch.org/docs/)

## 謝辞

このプロジェクトは、書籍「AlphaZero 深層学習・強化学習・探索 人工知能プログラミング実践入門」を参考に、Ultimate Tic-Tac-Toeに適用し、PyTorch + C++最適化を施したものです。

---

## クイックコマンド一覧

```powershell
# セットアップ
.\setup.ps1

# トレーニング開始
python -u train_cycle.py *>&1 | Tee-Object -FilePath training_log.txt

# 進捗監視
python monitor_training.py

# 対戦プレイ
python human_play.py

# システムリソース確認
python auto_tune.py

# ワーカー数比較
python compare_workers.py

# GPU確認
python check_gpu.py

# 可視化
python visualize_training.py
```

---

**バージョン**: 2.0  
**最終更新**: 2025年10月30日
