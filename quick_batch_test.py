"""
簡易バッチサイズテスト - 1ゲームだけ実行
"""

import torch
import time
from pathlib import Path

print("="*60)
print("Quick Batch Size Test")
print("="*60)

# GPU確認
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"\nGPU: {gpu_name}")
    print(f"VRAM: {total_memory:.1f} GB")
else:
    print("\nNo GPU available!")
    exit(1)

# C++バックエンド確認
try:
    import uttt_cpp
    from pv_mcts_cpp import pv_mcts_scores_cpp
    print("✅ C++ backend available")
except ImportError as e:
    print(f"❌ C++ backend not available: {e}")
    exit(1)

# モデルロード
print("\nLoading model...")
from dual_network import DualNetwork

model_path = Path('./model/best.pth')
if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    exit(1)

device_obj = torch.device('cuda')
model = DualNetwork()
model.load_state_dict(torch.load(model_path, map_location=device_obj, weights_only=False))
model.to(device_obj)
model.eval()
print("✅ Model loaded")

# GPU メモリ確認
allocated = torch.cuda.memory_allocated(device) / 1024**3
print(f"GPU Memory allocated: {allocated:.2f} GB")

# バッチサイズをテスト
batch_sizes = [8, 16, 24, 32, 48, 64, 96, 128]

print("\n" + "="*60)
print("Testing different batch sizes...")
print("="*60)

for batch_size in batch_sizes:
    print(f"\n>>> Batch Size: {batch_size}")
    
    try:
        torch.cuda.empty_cache()
        
        # 1ゲーム実行
        start_time = time.time()
        
        state = uttt_cpp.State()
        move_count = 0
        
        while not state.is_done() and move_count < 50:  # 最大50手まで
            scores = pv_mcts_scores_cpp(
                model, state,
                temperature=1.0,
                evaluate_count=100,
                batch_size=batch_size
            )
            
            action = torch.multinomial(torch.tensor(scores), 1).item()
            state = state.next(action)
            move_count += 1
        
        elapsed = time.time() - start_time
        
        # メモリ使用量
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        
        print(f"  ✅ Success")
        print(f"  Time: {elapsed:.2f}s ({move_count} moves)")
        print(f"  GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  ❌ OUT OF MEMORY")
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            print(f"  GPU Memory at failure: {allocated:.2f} GB")
            break
        else:
            print(f"  ❌ Error: {e}")
            break
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        break

print("\n" + "="*60)
print("Test complete!")
print("="*60)
