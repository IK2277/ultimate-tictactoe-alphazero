# ====================
# C++実装を使ったPV-MCTSラッパー
# ====================

import numpy as np
import torch
try:
    import uttt_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: uttt_cpp module not found. Using Python implementation.")

from dual_network import DN_INPUT_SHAPE, device

# C++バックエンドを使用した高速MCTS
def pv_mcts_scores_cpp(model, state, temperature, evaluate_count=50, batch_size=8):
    """
    C++実装のMCTSを使用してスコアを計算
    
    Args:
        model: PyTorchモデル (DualNetwork)
        state: C++のState オブジェクト
        temperature: ボルツマン分布の温度
        evaluate_count: シミュレーション回数
        batch_size: バッチ推論サイズ
    
    Returns:
        scores: 各合法手の確率分布
    """
    if not CPP_AVAILABLE:
        raise RuntimeError("C++ module is not available. Please build uttt_cpp first.")
    
    model.eval()
    
    # C++から呼ばれる推論関数を定義
    def inference_func(states_list):
        """
        C++から呼ばれるバッチ推論関数
        
        Args:
            states_list: C++ State オブジェクトのリスト
        
        Returns:
            結果のリスト: [(policy, value), ...]
        """
        # C++のStateからテンソルを取得
        tensors_list = []
        for state in states_list:
            # C++の to_input_tensor() は (H*W*C,) のフラット配列を返す
            tensor_flat = np.array(state.to_input_tensor(), dtype=np.float32)
            # (9*9*3,) -> (9, 9, 3) に変形
            tensor = tensor_flat.reshape(9, 9, 3)
            tensors_list.append(tensor)
        
        # (N, 9, 9, 3) のバッチを作成
        x = np.stack(tensors_list, axis=0)
        
        # NumPy (N, H, W, C) -> PyTorch (N, C, H, W)
        x = np.transpose(x, (0, 3, 1, 2))
        x = torch.FloatTensor(x).to(device)
        
        # 推論
        with torch.no_grad():
            policies, values = model(x)
        
        # GPU -> CPU -> NumPy
        policies = policies.cpu().numpy()
        values = values.cpu().numpy()
        
        # 結果をリストで返す
        results = []
        for i in range(len(states_list)):
            policy = policies[i]  # (81,)
            value = float(values[i][0])
            results.append((policy, value))
        
        return results
    
    # C++のMCTSを実行
    scores = uttt_cpp.pv_mcts_scores(
        model=inference_func,
        state=state,
        temperature=temperature,
        evaluate_count=evaluate_count,
        batch_size=batch_size
    )
    
    return np.array(scores)

# C++実装を使った行動選択関数
def pv_mcts_action_cpp(model, temperature=0, evaluate_count=50, batch_size=8):
    """
    C++実装のMCTSを使用した行動選択関数を返す
    
    Args:
        model: PyTorchモデル
        temperature: ボルツマン分布の温度
        evaluate_count: シミュレーション回数
        batch_size: バッチ推論サイズ
    
    Returns:
        action_func: 状態を受け取り行動を返す関数
    """
    def action_func(state):
        # PythonのStateオブジェクトをC++のStateに変換
        if not isinstance(state, uttt_cpp.State):
            # Python実装のStateからC++のStateに変換
            cpp_state = uttt_cpp.State(
                state.pieces,
                state.enemy_pieces,
                state.main_board_pieces,
                state.main_board_enemy_pieces,
                state.active_board
            )
        else:
            cpp_state = state
        
        scores = pv_mcts_scores_cpp(
            model, cpp_state, temperature, evaluate_count, batch_size
        )
        legal_actions = cpp_state.legal_actions()
        
        # scoresとlegal_actionsのサイズが一致しているか確認
        if len(scores) != len(legal_actions):
            raise ValueError(f"Score size mismatch: scores={len(scores)}, legal_actions={len(legal_actions)}")
        
        # スコアの合計が0の場合は均等分布にする
        if np.sum(scores) == 0:
            scores = np.ones(len(scores)) / len(scores)
        else:
            # 正規化
            scores = scores / np.sum(scores)
        
        return np.random.choice(legal_actions, p=scores)
    
    return action_func

# Python実装とC++実装の互換性チェック
def check_cpp_compatibility():
    """C++モジュールが正しくインストールされているか確認"""
    if not CPP_AVAILABLE:
        print("XX C++ module is NOT available")
        print("   Please run: cd cpp && pip install -e .")
        return False
    
    print("✅ C++ module is available")
    
    # 簡単な動作テスト
    try:
        state = uttt_cpp.State()
        legal_actions = state.legal_actions()
        print(f"✅ Game logic working (legal actions: {len(legal_actions)})")
        
        # 次の状態テスト
        if legal_actions:
            next_state = state.next(legal_actions[0])
            print(f"✅ State transition working")
        
        # テンソル変換テスト
        tensor = state.to_input_tensor()
        print(f"✅ Tensor conversion working (shape: {len(tensor)})")
        
        return True
    except Exception as e:
        print(f"XX C++ module test failed: {e}")
        return False

if __name__ == '__main__':
    check_cpp_compatibility()
