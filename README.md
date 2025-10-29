# Ultimate Tic-Tac-Toe AlphaZero

AlphaZeroアルゴリズムを使用したUltimate Tic-Tac-Toeの強化学習実装（PyTorch + C++最適化版）

## 概要

このプロジェクトは、AlphaZeroスタイルの自己対戦型強化学習を用いて、Ultimate Tic-Tac-Toe（究極の三目並べ）をプレイするAIを訓練します。

### 主な特徴

- **ResNetベースのDual Network** (Policy + Value Head)
- **Monte Carlo Tree Search (MCTS)** による探索
- **C++実装による高速化** (ゲームロジック20x、MCTS 50x高速)
- **GPU対応** (NVIDIA CUDA 12.1+)
- **PyTorch実装** (最適化済み)
- **リアルタイム訓練監視** (matplotlib可視化)

## システム要件

### ソフトウェア

- **Python** 3.11以降
- **CUDA** 12.1以降 (GPU使用時、推奨)
- **Visual Studio 2019以降** (Windows、C++拡張ビルド用)
- **Git** (バージョン管理)

### ハードウェア

- **推奨GPU**: NVIDIA GeForce RTX 4070 Ti以上 (12GB VRAM)
- **メモリ**: 16GB以上
- **ストレージ**: 5GB以上の空き容量

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

### 2. 手動セットアップ

#### Pythonパッケージのインストール

```bash
# CUDA 12.1対応PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# その他の依存関係
pip install -r requirements.txt
```

#### C++拡張のビルド（推奨）

```powershell
# PowerShellで実行（簡単な方法）
.\build_cpp.ps1
```

または手動でビルド：

```bash
cd cpp
pip install -e .
cd ..
```

#### 初期モデルの作成

```bash
python dual_network.py
```

これにより、`model/best.pth`と`model/latest.pth`が作成されます。

### 3. 動作確認

```bash
# GPUの確認
python check_gpu.py

# C++拡張の確認
python test_cpp_mcts.py
```

## 訓練の実行

### 完全な学習サイクル（推奨）

```powershell
# ログ付きで実行（リアルタイム監視用）
python -u train_cycle.py *>&1 | Tee-Object -FilePath training_log.txt
```

これにより以下が10サイクル実行されます：
1. **セルフプレイ**: 500ゲーム実行（C++バックエンド）
2. **ネットワーク訓練**: 100エポック（GPU使用）
3. **評価**: 最新モデル vs ベストモデル（50試合）
4. **ベストプレイヤー評価**: ランダムプレイヤーとの対戦（10試合）

### リアルタイム監視

別のターミナルで実行：

```bash
python monitor_training.py
```

2秒ごとにグラフが更新され、以下を表示：
- 損失の推移（移動平均付き）
- 学習率スケジュール
- モデル評価スコア
- 統計情報

### 訓練完了後の可視化

```bash
python visualize_training.py
```

詳細な訓練進捗レポートと4パネルのグラフを生成：
- 損失曲線（サイクル平均）
- 学習率推移
- 評価スコア
- ランダムプレイヤーとの勝率

### 個別コンポーネントの実行

```bash
# セルフプレイのみ
python self_play_cpp.py    # C++版（最速・推奨）
python self_play_hybrid.py # ハイブリッド版
python self_play.py        # Python版（テスト用）

# ネットワーク訓練のみ
python train_network.py

# 評価のみ
python evaluate_network.py        # 最新 vs ベスト
python evaluate_best_player.py     # vsランダム
```

## 人間 vs AI 対戦

```bash
python human_play.py
```

コンソールでAIと対戦できます。入力形式：`大盤面番号 小盤面番号`（例：`0 4`）

## プロジェクト構成

```
UTTT/
├── game.py                    # ゲームロジック（Ultimate Tic-Tac-Toe）
├── dual_network.py            # ResNetベースDual Network (PyTorch)
├── pv_mcts.py                 # MCTS実装 (Python版)
├── pv_mcts_cpp.py             # MCTS実装 (C++版、高速)
├── self_play.py               # セルフプレイ (Python版)
├── self_play_cpp.py           # セルフプレイ (C++版、推奨)
├── self_play_hybrid.py        # セルフプレイ (ハイブリッド版)
├── train_network.py           # ニューラルネットワーク訓練
├── train_cycle.py             # 完全学習サイクル
├── evaluate_network.py        # モデル評価 (latest vs best)
├── evaluate_best_player.py    # ベストプレイヤー評価 (vs random)
├── human_play.py              # 人間対戦インターフェース
├── visualize_training.py      # 訓練進捗可視化（事後分析）
├── monitor_training.py        # リアルタイム訓練モニター
├── check_gpu.py               # GPU動作確認
├── test_cpp_mcts.py           # C++実装テスト
├── setup.ps1                  # 自動セットアップスクリプト
├── build_cpp.ps1              # C++拡張ビルドスクリプト
├── run_training.ps1           # 訓練実行スクリプト
├── requirements.txt           # Python依存関係
├── README.md                  # このファイル
├── CPP_GUIDE.md               # C++実装詳細ガイド
├── cpp/                       # C++実装（pybind11）
│   ├── uttt_game.h            # ゲームロジック (C++)
│   ├── uttt_game.cpp          
│   ├── uttt_mcts.h            # MCTS実装 (C++)
│   ├── uttt_mcts.cpp          
│   ├── python_bindings.cpp    # Pythonバインディング
│   ├── setup.py               # C++ビルド設定
│   ├── pyproject.toml         # プロジェクトメタデータ
│   └── README.md              # C++実装ガイド
├── model/                     # 訓練済みモデル
│   ├── best.pth               # ベストモデル (PyTorch)
│   └── latest.pth             # 最新モデル (PyTorch)
├── data/                      # 学習データ（自動生成）
│   └── *.history              # セルフプレイ履歴
└── sample/                    # 元のサンプルコード
    └── sample/
        └── 6_7_tictactoe/     # 通常Tic-Tac-Toeサンプル
```

## パラメータ設定

### ハイパーパラメータ

#### self_play_cpp.py

```python
SP_GAME_COUNT = 500        # セルフプレイゲーム数/サイクル
SP_TEMPERATURE = 1.0       # 探索温度（高いほどランダム）
PV_EVALUATE_COUNT = 50     # MCTSシミュレーション回数
MCTS_BATCH_SIZE = 8        # バッチ推論サイズ（GPU最適化）
```

#### train_network.py

```python
RN_EPOCHS = 100            # 訓練エポック数
BATCH_SIZE = 128           # ミニバッチサイズ
LEARNING_RATE = 0.001      # 初期学習率
MOMENTUM = 0.9             # SGDモーメンタム
WEIGHT_DECAY = 1e-4        # L2正則化
LR_SCHEDULE = [50, 80]     # 学習率減衰ステップ
```

#### dual_network.py

```python
DN_FILTERS = 128           # ResNet フィルタ数
DN_RESIDUAL_NUM = 16       # ResNet ブロック数
DN_INPUT_SHAPE = (3, 9, 9) # 入力形状（チャンネル, H, W）
DN_OUTPUT_SIZE = 81        # 行動数（9x9グリッド）
```
```

### train_network.py

#### evaluate_network.py

```python
EN_GAME_COUNT = 50         # 評価ゲーム数（latest vs best）
EN_TEMPERATURE = 0.0       # 探索温度（評価時は決定的）
```

#### evaluate_best_player.py

```python
EP_GAME_COUNT = 10         # ベンチマークゲーム数
```

### モデルアーキテクチャ

ニューラルネットワークは**ResNetベースのDual Network**：

```
Input (3, 9, 9)
    ↓
Conv2D (128 filters, 3x3)
    ↓
ResBlock x 16
    ├─ Conv2D (128, 3x3)
    ├─ BatchNorm + ReLU
    ├─ Conv2D (128, 3x3)
    ├─ BatchNorm
    └─ Skip Connection + ReLU
    ↓
┌───────────────┴───────────────┐
│ Policy Head                   │ Value Head
│ Conv2D (2, 1x1)               │ Conv2D (1, 1x1)
│ BatchNorm + ReLU              │ BatchNorm + ReLU
│ Flatten                       │ Flatten
│ Dense (81)                    │ Dense (256) + ReLU
│ Softmax                       │ Dense (1) + Tanh
└─────────────────────────────────┘
```

- **パラメータ数**: 約1,200万
- **Policy Head**: 81次元（全行動の確率分布）
- **Value Head**: [-1, 1]（局面評価値）

## パフォーマンス

### 実測ベンチマーク（RTX 4070 Ti + Ryzen 7 5800X）

| コンポーネント | Python版 | C++版 | 高速化率 |
|--------------|---------|------|---------|
| ゲームロジック（1手） | 100 μs | 5 μs | **20x** |
| MCTS（50シミュレーション） | 2.5 s | 50 ms | **50x** |
| セルフプレイ（500ゲーム） | 60 分 | **2 分** | **30x** |
| 完全学習サイクル | 70 分 | **4 分** | **17x** |

### GPU加速効果

| 環境 | 訓練時間/エポック | 10サイクル合計 |
|------|-----------------|--------------|
| CPU のみ | 120秒 | **~20時間** |
| RTX 4070 Ti | **2秒** | **~40分** |

**推奨環境**: GPU + C++バックエンドで最大50倍の高速化

## トラブルシューティング

### GPU関連

**問題**: GPUが認識されない

```bash
# CUDAバージョン確認
nvidia-smi

# CUDA対応PyTorchの再インストール
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 確認
python check_gpu.py
```

### C++拡張関連

**問題**: C++拡張のビルドエラー

```powershell
# Visual Studioの確認（"C++によるデスクトップ開発"が必要）
# https://visualstudio.microsoft.com/ja/downloads/

# pybind11の再インストール
pip install --upgrade pybind11

# ビルドのクリーンアップ
cd cpp
Remove-Item -Recurse -Force build,*.pyd,*.so -ErrorAction SilentlyContinue
python setup.py build_ext --inplace
cd ..
```

**問題**: ImportError: DLL load failed

```powershell
# Visual C++ 再頒布可能パッケージのインストール
# https://aka.ms/vs/17/release/vc_redist.x64.exe
```

### エンコーディング関連

**問題**: UnicodeEncodeError (Windows PowerShell)

すでに修正済み：すべての絵文字をASCII文字に置換済み

### 訓練関連

**問題**: 損失が減少しない

- 学習率を調整：`train_network.py`の`LEARNING_RATE`を0.0001に減少
- バッチサイズを増加：`BATCH_SIZE = 256`
- セルフプレイゲーム数を増加：`SP_GAME_COUNT = 1000`

**問題**: メモリ不足

- バッチサイズを削減：`BATCH_SIZE = 64`
- MCTSシミュレーション回数を削減：`PV_EVALUATE_COUNT = 30`
- ResNetブロック数を削減：`DN_RESIDUAL_NUM = 8`

### Windows関連

**問題**: マルチプロセッシングエラー

すでに修正済み：`train_network.py`で`NUM_WORKERS=0`に設定済み

**問題**: PowerShellでTee-Objectが遅い

UTF-16エンコーディングの問題。以下を使用：

```powershell
python -u train_cycle.py *>&1 | Tee-Object -FilePath training_log.txt
```

## 学習の進め方

### 推奨トレーニングスケジュール

1. **初回実行（5-10サイクル）**
   - 基本的な戦略を学習
   - ランダムプレイヤーに対して70-80%勝率を目指す

2. **中期訓練（10-20サイクル）**
   - 戦術的なプレイを学習
   - 勝率85%以上を目指す

3. **長期訓練（20-50サイクル）**
   - 高度な戦略を習得
   - 一貫して90%以上の勝率

### 学習の評価指標

- **損失**: 3.6 → 3.5以下に減少（最初の数サイクル）
- **評価スコア**: 0.5以上（モデル改善の指標）
- **vs Random**: 0.7以上（良好な学習）、0.8以上（優秀）

## 技術詳細

### AlphaZero アルゴリズム

1. **セルフプレイ**: ニューラルネットワーク + MCTSで自己対戦
2. **訓練データ生成**: 各局面の（状態、方策、価値）を記録
3. **教師あり学習**: 方策と価値の同時学習
4. **モデル評価**: 新モデルと旧モデルを対戦させて選択
5. **反復**: 1-4を繰り返す

### MCTS（Monte Carlo Tree Search）

- **PUCT（Predictor + Upper Confidence bounds applied to Trees）**
- バッチ推論による高速化（GPU効率向上）
- C++実装で探索速度50倍向上

### ゲームのエンコーディング

**入力**: (3, 9, 9) テンソル
- チャンネル0: 現プレイヤーの駒
- チャンネル1: 相手プレイヤーの駒
- チャンネル2: 有効な大盤面（プレイ可能領域）

**出力**:
- Policy: 81次元ベクトル（各マスの選択確率）
- Value: スカラー（勝率予測、-1〜1）

## 参考資料

### 論文

- [Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)
- [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://science.sciencemag.org/content/362/6419/1140)
- [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)](https://arxiv.org/abs/1911.08265)

### リンク

- [Ultimate Tic-Tac-Toe ルール](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe)
- [PyTorch 公式ドキュメント](https://pytorch.org/docs/stable/index.html)
- [pybind11 ドキュメント](https://pybind11.readthedocs.io/)

## 開発履歴

### v2.0（現在のバージョン）

- PyTorchへの完全移行（TensorFlow/Kerasから）
- C++バックエンドの実装（30-50倍高速化）
- GPU最適化（cudnn.benchmark、pin_memory）
- Windows互換性の改善
- リアルタイム訓練監視の追加
- エンコーディング問題の修正

### v1.0（初期バージョン）

- TensorFlow/Kerasベースの実装
- Python版MCTS
- 基本的なAlphaZeroアルゴリズム

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します！以下の改善アイデアがあります：

### TODO

- [ ] マルチGPU対応
- [ ] 分散学習（複数マシン）
- [ ] より深いネットワーク（32ブロック）
- [ ] 開局データベースの統合
- [ ] Web UIの実装（React/Flask）
- [ ] モデルの量子化（推論高速化）
- [ ] TensorBoard統合
- [ ] Docker化

## 連絡先

質問・提案・バグ報告は[GitHub Issues](https://github.com/IK2277/ultimate-tictactoe-alphazero/issues)でお願いします。

---

**Author**: IK2277  
**Repository**: https://github.com/IK2277/ultimate-tictactoe-alphazero  
**Last Updated**: 2025年10月29日
