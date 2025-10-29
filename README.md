# Ultimate Tic-Tac-Toe AlphaZero

AlphaZeroアルゴリズムを使用したUltimate Tic-Tac-Toeの強化学習実装

## 🎮 概要

このプロジェクトは、AlphaZeroスタイルの自己対戦型強化学習を用いて、Ultimate Tic-Tac-Toe（究極の三目並べ）をプレイするAIを訓練します。

### 主な特徴

- 🧠 **ResNetベースのDual Network** (Policy + Value)
- 🔍 **Monte Carlo Tree Search (MCTS)** による探索
- 🚀 **C++実装による高速化** (30-50倍高速)
- 💻 **GPU対応** (NVIDIA CUDA)
- 🎯 **PyTorch実装** (TensorFlowから完全移行)

## 📋 必要要件

### ソフトウェア

- Python 3.8以降
- CUDA 12.1以降 (GPU使用時)
- Visual Studio 2019以降 (Windows、C++拡張用)
- Git

### Pythonパッケージ

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pybind11
```

## 🚀 セットアップ

### 1. リポジトリのクローン

```bash
git clone <your-repo-url>
cd UTTT
```

### 2. 初期モデルの作成

```bash
python dual_network.py
```

これにより、`model/best.pth`と`model/latest.pth`が作成されます。

### 3. C++拡張のビルド（オプションだが推奨）

```bash
cd cpp
pip install -e .
cd ..
```

または

```bash
pip install pybind11
cd cpp
python setup.py build_ext --inplace
cd ..
```

### 4. 動作確認

```bash
# GPUの確認
python check_gpu.py

# C++拡張の確認（ビルドした場合）
python pv_mcts_cpp.py
```

## 🎓 学習の実行

### 完全な学習サイクル

```bash
python train_cycle.py
```

これにより以下が10サイクル実行されます：
1. **セルフプレイ**: 500ゲーム実行
2. **ネットワーク訓練**: 100エポック
3. **評価**: 最新モデル vs ベストモデル（50試合）
4. **ベストプレイヤー評価**: ランダム/MCTS vs AI

### 個別の実行

```bash
# セルフプレイのみ
python self_play_cpp.py    # C++版（高速）
python self_play_hybrid.py # ハイブリッド版
python self_play.py        # Python版

# ネットワーク訓練のみ
python train_network.py

# 評価のみ
python evaluate_network.py
python evaluate_best_player.py
```

## 🎮 人間 vs AI 対戦

```bash
python human_play.py
```

## 📁 プロジェクト構成

```
UTTT/
├── game.py                    # ゲームロジック
├── dual_network.py            # ニューラルネットワーク
├── pv_mcts.py                # MCTS (Python版)
├── pv_mcts_cpp.py            # MCTS (C++ラッパー)
├── self_play.py              # セルフプレイ (Python版)
├── self_play_cpp.py          # セルフプレイ (C++版)
├── self_play_hybrid.py       # セルフプレイ (ハイブリッド版)
├── train_network.py          # ネットワーク訓練
├── train_cycle.py            # 学習サイクル全体
├── evaluate_network.py       # モデル評価
├── evaluate_best_player.py   # ベストプレイヤー評価
├── human_play.py             # 人間対戦GUI
├── check_gpu.py              # GPU確認ユーティリティ
├── test_cpp_mcts.py          # C++実装のテスト
├── cpp/                      # C++実装
│   ├── uttt_game.h/cpp       # ゲームロジック (C++)
│   ├── uttt_mcts.h/cpp       # MCTS (C++)
│   ├── python_bindings.cpp   # Pythonバインディング
│   ├── setup.py              # ビルド設定
│   └── README.md             # C++実装ガイド
├── model/                    # 訓練済みモデル
│   ├── best.pth              # ベストモデル
│   └── latest.pth            # 最新モデル
├── data/                     # 学習データ
│   └── *.history             # セルフプレイ履歴
└── sample/                   # サンプルコード

```

## ⚙️ パラメータ調整

### self_play.py / self_play_cpp.py

```python
SP_GAME_COUNT = 500        # セルフプレイゲーム数
SP_TEMPERATURE = 1.0       # 探索の温度
PV_EVALUATE_COUNT = 50     # MCTSシミュレーション回数
MCTS_BATCH_SIZE = 8        # バッチ推論サイズ
```

### train_network.py

```python
BATCH_SIZE = 128           # 訓練バッチサイズ
EPOCHS = 100               # エポック数
LEARNING_RATE = 0.001      # 学習率
```

### evaluate_network.py

```python
EN_GAME_COUNT = 50         # 評価ゲーム数
EN_TEMPERATURE = 1.0       # 探索の温度
```

## 🚀 パフォーマンス

### Python版 vs C++版

| 処理 | Python | C++ | 高速化率 |
|------|--------|-----|----------|
| ゲームロジック | 100 μs | 5 μs | **20x** |
| MCTS (50回) | 2.5 s | 50 ms | **50x** |
| セルフプレイ (500ゲーム) | 60 分 | 2 分 | **30x** |
| 学習1サイクル | 60 分 | **2 分** | **30x** |

### GPU使用時

- RTX 4070 Ti: ~2分/サイクル
- CPU のみ: ~60分/サイクル

## 🔧 トラブルシューティング

### C++拡張のビルドエラー

```bash
# Visual Studioのインストールを確認
# "C++によるデスクトップ開発"ワークロードが必要

# pybind11の再インストール
pip install --upgrade pybind11

# ビルドのクリーンアップ
cd cpp
rm -rf build *.pyd *.so
python setup.py build_ext --inplace
```

### GPU が認識されない

```bash
# CUDA対応PyTorchの再インストール
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 確認
python check_gpu.py
```

### Windows マルチプロセッシングエラー

`train_network.py`で`NUM_WORKERS=0`に設定されています（既に対応済み）

## 📚 参考文献

- [AlphaGo Zero論文](https://www.nature.com/articles/nature24270)
- [AlphaZero論文](https://arxiv.org/abs/1712.01815)
- [Ultimate Tic-Tac-Toe ルール](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe)

## 📝 ライセンス

MIT License

## 👥 貢献

プルリクエストを歓迎します！

## 🎯 TODO

- [ ] マルチスレッド対応
- [ ] より大きなネットワーク
- [ ] 開局データベース
- [ ] Web UIの実装
- [ ] モデルの圧縮

## 📧 連絡先

質問や提案がある場合は、Issueを作成してください。
