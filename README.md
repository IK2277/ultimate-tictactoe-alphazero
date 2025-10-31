# Ultimate Tic-Tac-Toe AlphaZero (ver2.0)

AlphaZeroアルゴリズムを使用したUltimate Tic-Tac-Toeの強化学習実装  
**PyTorch + C++最適化 + 並列処理 + リアルタイム可視化 + バッチサイズ最適化**

---

##  主な特徴

-  **ResNetベースのDual Network** (Policy + Value Head, 1.2M parameters)
-  **Monte Carlo Tree Search (MCTS)** による探索
-  **C++実装による高速化** (ゲームロジック30-50倍高速)
-  **マルチプロセス並列化** (8ワーカー, 5-6倍高速)
-  **GPU対応** (NVIDIA CUDA 12.1+, TF32cuDNN最適化)
-  **バッチサイズ最適化** (RTX 3090で5.5倍高速化達成) 🆕
-  **動的パラメータ調整** (MCTS探索回数, 学習率, ゲーム数)
-  **リアルタイム可視化** (損失評価スコアのグラフ表示)
-  **GPUモニタリング** (リアルタイムでGPU使用状況を表示) 🆕
-  **チェックポイント機能** (いつでも中断再開可能)

---

##  システム要件

### 必須環境
- **Python**: 3.11以降
- **OS**: Windows 10/11 (PowerShell)
- **ストレージ**: 2GB以上の空き容量

### 推奨環境
- **GPU**: NVIDIA GeForce RTX 3090 / 4070 Ti以上 (12-24GB VRAM)
- **CUDA**: 12.1以降
- **CPU**: 12コア以上 (並列処理用)
- **メモリ**: 32GB以上
- **Visual Studio**: 2019以降 (C++拡張ビルド用)

### テスト済み環境
- **RTX 3090 (24GB)**: バッチサイズ64で5.5倍高速化達成 ⭐
- **RTX 4070 Ti (12GB)**: バッチサイズ32-48で安定動作

---

##  クイックスタート

### 1. 環境構築

```powershell
# 自動セットアップスクリプト実行
.\setup.ps1
```

このスクリプトは以下を自動実行します：
- CUDA対応PyTorchのインストール
- 依存パッケージのインストール
- C++拡張のビルド
- GPUの動作確認

### 2. トレーニング開始

#### 方法1: リアルタイム可視化あり（推奨）

**ターミナル1: トレーニング実行**
```powershell
python train_with_log.py
```

**ターミナル2: グラフ表示**
```powershell
python monitor_simple.py
```

#### 方法2: シンプル実行

```powershell
python train_cycle.py
```

#### 方法3: GPU使用状況の監視（推奨）

**ターミナル3: GPUモニタリング** 🆕
```powershell
python monitor_gpu.py
```
リアルタイムでGPU温度、使用率、メモリ、電力を表示します。

### 3. 人間 vs AI 対戦

```powershell
python human_play.py
```

---

##  トレーニングの仕組み

### 学習サイクル

```
Train 0 ====================================
  
 自己対戦（500ゲーム, 8ワーカー並列）
  
 学習（100エポック）
  
 評価（100試合, 最新 vs ベスト）
  
 モデル更新（勝率に関係なく常に更新）
  
Train 1 ====================================
```

### 動的パラメータ調整

トレーニングの進行に応じて自動調整：

| サイクル | ゲーム数 | MCTS探索回数 | 学習率 |
|---------|---------|-------------|--------|
| 0-9     | 500     | 100         | 0.001  |
| 10-19   | 1000    | 200         | 0.0005 |
| 20-29   | 1500    | 400         | 0.0002 |
| 30+     | 2000    | 800         | 0.0001 |

---

##  リアルタイム可視化

### グラフ表示内容

- **左側**: 損失（Loss）の推移
- **右側**: 評価スコア（Win Rate）

### 更新頻度
- 1秒ごとに自動更新
- 最新の値を表示
- Ctrl+C で終了すると `training_progress.png` に保存

### 進捗確認

ターミナルに以下のような出力が表示されます：

```
==================================================
シンプル学習モニター
==================================================
ログファイル: training.log
Ctrl+C で終了

更新回数: 10, 損失数: 50, 評価数: 0
更新回数: 20, 損失数: 100, 評価数: 1
```

---

##  パフォーマンス設定

### 並列ワーカー数の調整

`train_cycle.py` の設定を変更：

```python
# aggressive: 8ワーカー（推奨）
WORKER_MODE = "aggressive"

# auto: 4-6ワーカー（保守的）
WORKER_MODE = "auto"

# 手動指定
WORKER_MODE = 6
```

### GPU最適化 🆕

自動的に有効化される最適化：
-  **TF32**: Tensor Float 32（NVIDIA Ampere以降）
-  **cuDNN benchmark**: 最適な畳み込みアルゴリズム自動選択
-  **MCTS バッチサイズ**: GPU別に自動最適化
  - RTX 3090 (24GB): **64** (5.5倍高速化) ⚡
  - RTX 4070 Ti (12GB): **48**
  - RTX 3080 (10-12GB): **32-48**
  - RTX 3060 (8GB): **24**

### バッチサイズのベンチマーク 🆕

システムに最適なバッチサイズを測定：

```powershell
python quick_batch_test.py
```

詳細なベンチマーク結果は `BATCH_SIZE_OPTIMIZATION_REPORT.md` を参照。

---

##  ファイル構成

### コアファイル

```
train_cycle.py           # メインの学習ループ
train_with_log.py        # ログ付き学習実行ラッパー
monitor_simple.py        # リアルタイム可視化ツール

dual_network.py          # ResNetベースのニューラルネットワーク
train_network.py         # ネットワークの訓練
evaluate_network.py      # モデル評価（100試合）

self_play_parallel.py    # 並列自己対戦（8ワーカー）
pv_mcts.py              # Python版MCTS
pv_mcts_cpp.py          # C++版MCTS（高速）

game.py                 # ゲームロジック
human_play.py           # 人間対戦モード
```

### C++拡張

```
cpp/
  uttt_game.cpp         # ゲームロジック（C++）
  uttt_mcts.cpp         # MCTS実装（C++）
  python_bindings.cpp   # Python連携
```

### 設定・モニタリングツール 🆕

```
auto_tune.py            # システムリソース自動検出
optimize_inference.py   # PyTorch最適化設定
monitor_gpu.py          # GPUリアルタイムモニタリング 🆕
quick_batch_test.py     # バッチサイズベンチマーク 🆕
setup.ps1              # 自動セットアップスクリプト
requirements.txt       # 依存パッケージ
```

### データ・モデル

```
model/
  best.pth             # ベストモデル
  latest.pth           # 最新モデル

data/
  *.history            # 自己対戦データ

training.log           # トレーニングログ（Git管理対象）
training_progress.png  # グラフ画像（終了時）
```

---

##  複数デバイス間での作業

### training.logの共有

`training.log` はGit管理されているため、別のデバイスでも学習状況を確認できます。

#### 別デバイスでの確認方法

```powershell
# 最新の学習ログを取得
git pull origin ver2.0

# ログの確認
Get-Content training.log -Tail 20  # 最新20行を表示

# リアルタイム監視（別デバイスでも可能）
python monitor_simple.py
```

#### 学習実行デバイスでの更新

```powershell
# 定期的にログを共有（例: 10サイクルごと）
git add training.log
git commit -m "update: Training log Cycle XX"
git push origin ver2.0
```

**メリット:**
- 複数デバイスで学習状況を同期
- GitHubで学習履歴を管理
- 別デバイスからもグラフ表示可能

---

##  トラブルシューティング

### GPU が認識されない

```powershell
# GPU確認
python check_gpu.py

# CUDA対応PyTorchを再インストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### C++拡張のビルドエラー

```powershell
# Visual Studio Build Toolsがインストールされているか確認
# 再ビルド
cd cpp
python setup.py build_ext --inplace
```

### 出力がリアルタイムで表示されない

- `train_with_log.py` を使用してください
- PowerShellのリダイレクト（`>`）はバッファリングされるため非推奨

### メモリ不足エラー

- ワーカー数を減らす: `WORKER_MODE = 4`
- バッチサイズを減らす（`self_play_parallel.py`内）

---

##  期待される学習速度

### RTX 4070 Ti + 12コアCPU の場合

| フェーズ | 時間（サイクルあたり） |
|---------|----------------------|
| Cycle 0-9 (500ゲーム) | 約40-60秒 |
| Cycle 10-19 (1000ゲーム) | 約2-4分 |
| Cycle 20-29 (1500ゲーム) | 約6-11分 |
| Cycle 30+ (2000ゲーム) | 約12-20分 |

### スピードアップ効果

- **C++ゲームロジック**: 30-50倍高速
- **8ワーカー並列化**: 5-6倍高速
- **総合**: ベースライン比 **150-300倍高速**

---

##  ベンチマークテスト

### 速度テスト

```powershell
# 基本速度テスト
python test_speed.py

# ワーカー数比較
python compare_workers.py

# 自己対戦ベンチマーク
python benchmark_selfplay.py
```

### C++実装のテスト

```powershell
python test_cpp_mcts.py
```

---

##  技術詳細

### ニューラルネットワーク

- **アーキテクチャ**: ResNet (9x9入力, 9ブロック)
- **Policy Head**: 81次元出力（全マス）
- **Value Head**: 1次元出力（勝率予測）
- **パラメータ数**: 約1.2M
- **最適化**: Adam optimizer, 動的学習率

### MCTS設定

- **探索回数**: 100-800（動的調整）
- **バッチサイズ**: GPU別に自動最適化 (8-64) 🆕
- **温度パラメータ**: 1.0
- **UCB定数**: 1.0

### 評価方法

- **試合数**: 100試合
- **対戦相手**: latest vs best
- **更新ポリシー**: 勝率に関係なく常に更新

---

##  開発履歴（ver2.0の主要変更）

### 最適化

-  8ワーカー並列化（aggressive mode）
-  C++バックエンド統合
-  TF32cuDNN最適化
-  バッファリング問題解決（flush=True）
-  **バッチサイズ最適化** (5.5倍高速化達成) 🆕

### 評価システム

-  評価ゲーム数: 50→100試合
-  モデル更新: 常に更新（勝率条件削除）

### 可視化・モニタリング 🆕

-  シンプルなリアルタイムモニター追加
-  **GPUモニタリングツール** (温度、使用率、メモリ、電力)
-  ログファイル方式に統一（追記モード対応）
-  1秒ごとの高速更新

### ベンチマーク・テストツール 🆕

-  **バッチサイズベンチマークツール** (`quick_batch_test.py`)
-  システムリソース自動検出の改善
-  GPU別最適設定の自動適用

### マルチデバイス対応

-  `training.log` をGit管理対象に追加
-  複数デバイス間での学習状況共有が可能

### その他

-  動的パラメータ調整
-  チェックポイント機能
-  ドキュメント整理
-  詳細なベンチマークレポート作成

---

##  コントリビューション

プルリクエスト歓迎！以下の点にご注意ください：

- コードスタイル: PEP 8準拠
- コミットメッセージ: 英語推奨
- テスト: 変更箇所のテスト追加

---

##  ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

##  参考資料

### 論文
- [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) (AlphaGo Zero)
- [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://www.science.org/doi/10.1126/science.aar6404) (AlphaZero)

### Ultimate Tic-Tac-Toe ルール
- [Wikipedia - Ultimate Tic-Tac-Toe](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe)

---

##  お問い合わせ

質問バグ報告は [Issues](https://github.com/IK2277/ultimate-tictactoe-alphazero/issues) へお願いします。

---

##  更新履歴

### ver2.0.1 (2025年10月31日) 🆕
- バッチサイズ最適化 (RTX 3090で5.5倍高速化)
- GPUモニタリングツール追加
- バッチサイズベンチマークツール追加
- ログ追記モード対応
- 詳細なベンチマークレポート作成
- READMEの大幅更新

### ver2.0 (2025年10月30日)
- 並列処理による高速化
- リアルタイム可視化
- C++バックエンド統合
- 動的パラメータ調整

---

**Happy Training! 🚀**
