"""
C++実装のテストとデバッグ
"""
import numpy as np
import torch
from dual_network import DualNetwork, device

try:
    import uttt_cpp
    print("✅ C++ module imported")
except ImportError as e:
    print(f"❌ Failed to import C++ module: {e}")
    exit(1)

# モデルの読み込み
print("\n[1] Loading model...")
model = DualNetwork().to(device)
model.load_state_dict(torch.load('./model/best.pth', map_location=device, weights_only=True))
model.eval()
print("✅ Model loaded")

# 状態の作成
print("\n[2] Creating initial state...")
state = uttt_cpp.State()
legal_actions = state.legal_actions()
print(f"✅ Initial state created")
print(f"   Legal actions: {len(legal_actions)}")
print(f"   First 5 actions: {legal_actions[:5]}")

# テンソルの取得
print("\n[3] Testing tensor conversion...")
tensor = state.to_input_tensor()
print(f"✅ Tensor shape: {len(tensor)} (expected: 9*9*3=243)")
tensor_reshaped = np.array(tensor, dtype=np.float32).reshape(9, 9, 3)
print(f"   Reshaped: {tensor_reshaped.shape}")

# 推論関数のテスト
print("\n[4] Testing inference function...")
def test_inference(states_list):
    print(f"   Inference called with {len(states_list)} states")
    
    tensors_list = []
    for s in states_list:
        tensor_flat = np.array(s.to_input_tensor(), dtype=np.float32)
        tensor = tensor_flat.reshape(9, 9, 3)
        tensors_list.append(tensor)
    
    x = np.stack(tensors_list, axis=0)
    x = np.transpose(x, (0, 3, 1, 2))
    x = torch.FloatTensor(x).to(device)
    
    with torch.no_grad():
        policies, values = model(x)
    
    policies = policies.cpu().numpy()
    values = values.cpu().numpy()
    
    results = []
    for i in range(len(states_list)):
        policy = policies[i]  # (81,)
        value = float(values[i][0])
        print(f"   State {i}: policy shape={policy.shape}, value={value:.4f}")
        results.append((policy, value))
    
    return results

# 1回の推論テスト
print("\n[5] Single inference test...")
results = test_inference([state])
print(f"✅ Inference successful")
print(f"   Policy sum: {np.sum(results[0][0]):.4f}")
print(f"   Value: {results[0][1]:.4f}")

# MCTS実行テスト
print("\n[6] Testing MCTS (small scale)...")
try:
    scores = uttt_cpp.pv_mcts_scores(
        model=test_inference,
        state=state,
        temperature=1.0,
        evaluate_count=10,  # 少ないシミュレーション回数でテスト
        batch_size=2
    )
    print(f"✅ MCTS completed")
    print(f"   Scores type: {type(scores)}")
    print(f"   Scores length: {len(scores)}")
    print(f"   Legal actions: {len(legal_actions)}")
    print(f"   Scores sum: {np.sum(scores):.4f}")
    
    if len(scores) > 0:
        print(f"   First 5 scores: {scores[:5]}")
        print(f"   Max score: {np.max(scores):.4f}")
        print(f"   Min score: {np.min(scores):.4f}")
    
    # サイズチェック
    if len(scores) == len(legal_actions):
        print("✅ Score size matches legal actions!")
    else:
        print(f"❌ Size mismatch: scores={len(scores)}, legal_actions={len(legal_actions)}")
        
except Exception as e:
    print(f"❌ MCTS failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[7] All tests completed!")
