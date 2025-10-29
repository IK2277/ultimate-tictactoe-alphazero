# Ultimate Tic-Tac-Toe C++ Implementation

高速化のためのC++実装とPythonバインディング

## ビルド方法

### 必要な環境
- Visual Studio 2019以降 (Windows) または GCC/Clang (Linux/Mac)
- Python 3.8以降
- pybind11

### インストール

```bash
# pybind11のインストール
pip install pybind11

# C++拡張モジュールのビルドとインストール
cd cpp
pip install -e .
```

## 使用方法

```python
import uttt_cpp

# ゲーム状態の作成
state = uttt_cpp.State()

# 合法手の取得
legal_actions = state.legal_actions()

# 次の状態へ遷移
next_state = state.next(legal_actions[0])

# MCTS探索の実行
# model は推論関数 (状態のリスト -> (policy, value) のリスト)
scores = uttt_cpp.pv_mcts_scores(
    model=inference_func,
    state=state,
    temperature=1.0,
    evaluate_count=50,
    batch_size=8
)
```

## パフォーマンス

Python実装と比較して：
- ゲームロジック: 約10-20倍高速
- MCTS探索: 約30-50倍高速
- セルフプレイ全体: 約20-30倍高速

## ファイル構成

- `uttt_game.h/cpp`: ゲームロジックの実装
- `uttt_mcts.h/cpp`: MCTS探索の実装
- `python_bindings.cpp`: pybind11によるPythonバインディング
- `setup.py`: ビルド設定
