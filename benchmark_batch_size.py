"""
バッチサイズの最適化ベンチマーク
段階的にバッチサイズを増やしてパフォーマンスとメモリ使用量を測定
"""

import torch
import time
import psutil
import subprocess
import sys
from pathlib import Path
from dual_network import DN_INPUT_SHAPE

def get_gpu_memory_usage():
    """GPU メモリ使用量を取得"""
    if not torch.cuda.is_available():
        return 0, 0, 0
    
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    
    return total, allocated, reserved

def get_ram_usage():
    """RAM使用量を取得"""
    memory = psutil.virtual_memory()
    return memory.total / 1024**3, memory.used / 1024**3, memory.percent

def monitor_nvidia_smi():
    """nvidia-smiでGPU使用率を確認"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None

def test_batch_size(batch_size, num_games=10):
    """
    指定されたバッチサイズでセルフプレイをテストする
    
    Args:
        batch_size: テストするMCTSバッチサイズ
        num_games: テストするゲーム数
    
    Returns:
        (成功, 平均時間, GPU使用量)
    """
    print(f"\n{'='*60}")
    print(f"Testing Batch Size: {batch_size}")
    print(f"{'='*60}")
    
    # 初期メモリ状態
    torch.cuda.empty_cache()
    time.sleep(1)
    
    gpu_total, gpu_allocated_before, gpu_reserved_before = get_gpu_memory_usage()
    ram_total, ram_used_before, ram_percent_before = get_ram_usage()
    
    print(f"\n[Before Test]")
    print(f"GPU Memory: {gpu_allocated_before:.2f}/{gpu_total:.1f} GB allocated")
    print(f"RAM: {ram_used_before:.1f}/{ram_total:.1f} GB used ({ram_percent_before:.1f}%)")
    
    # nvidia-smiで確認
    nvidia_info = monitor_nvidia_smi()
    if nvidia_info:
        print(f"nvidia-smi: {nvidia_info}")
    
    try:
        # モデルのロード
        from dual_network import DN_INPUT_SHAPE
        from pathlib import Path
        import pickle
        
        # C++バックエンドの確認
        try:
            import uttt_cpp
            from pv_mcts_cpp import pv_mcts_scores_cpp
            cpp_available = True
            print("\n>> Using C++ backend")
        except ImportError:
            cpp_available = False
            print("\n>> C++ backend not available")
            return False, 0, 0
        
        # モデルのロード
        model_path = Path('./model/best.pth')
        if not model_path.exists():
            print("Error: model/best.pth not found")
            return False, 0, 0
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # デュアルネットワークをインポート
        from dual_network import DualNetwork
        model = DualNetwork()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        print(f"Model loaded on {device}")
        
        # メモリ使用量（モデルロード後）
        gpu_total, gpu_allocated_after_model, gpu_reserved_after_model = get_gpu_memory_usage()
        print(f"\n[After Model Load]")
        print(f"GPU Memory: {gpu_allocated_after_model:.2f}/{gpu_total:.1f} GB allocated")
        print(f"Model size: {(gpu_allocated_after_model - gpu_allocated_before):.2f} GB")
        
        # セルフプレイのテスト
        game_times = []
        
        print(f"\n[Running {num_games} games with batch_size={batch_size}]")
        
        for i in range(num_games):
            start_time = time.time()
            
            # 1ゲーム実行
            state = uttt_cpp.State()
            move_count = 0
            
            while not state.is_done():
                # MCTSでスコアを取得
                scores = pv_mcts_scores_cpp(
                    model, state, 
                    temperature=1.0, 
                    evaluate_count=100,  # Train 0-9の設定
                    batch_size=batch_size
                )
                
                # 行動を選択
                action = torch.multinomial(torch.tensor(scores), 1).item()
                state = state.next(action)
                move_count += 1
            
            elapsed = time.time() - start_time
            game_times.append(elapsed)
            
            # リアルタイム表示
            print(f"  Game {i+1}/{num_games}: {elapsed:.2f}s ({move_count} moves)", end='', flush=True)
            
            # GPU使用量を表示
            _, gpu_allocated, gpu_reserved = get_gpu_memory_usage()
            print(f" | GPU: {gpu_allocated:.2f}GB", flush=True)
        
        # 統計
        avg_time = sum(game_times) / len(game_times)
        min_time = min(game_times)
        max_time = max(game_times)
        
        # 最終メモリ状態
        gpu_total, gpu_allocated_final, gpu_reserved_final = get_gpu_memory_usage()
        ram_total, ram_used_final, ram_percent_final = get_ram_usage()
        
        print(f"\n[After Test]")
        print(f"GPU Memory: {gpu_allocated_final:.2f}/{gpu_total:.1f} GB allocated")
        print(f"RAM: {ram_used_final:.1f}/{ram_total:.1f} GB used ({ram_percent_final:.1f}%)")
        
        # 結果サマリー
        print(f"\n{'='*60}")
        print(f"Results for Batch Size {batch_size}:")
        print(f"{'='*60}")
        print(f"✅ SUCCESS")
        print(f"Average time per game: {avg_time:.2f}s")
        print(f"Min/Max time: {min_time:.2f}s / {max_time:.2f}s")
        print(f"Games per second: {1/avg_time:.2f}")
        print(f"Peak GPU Memory: {gpu_allocated_final:.2f} GB / {gpu_total:.1f} GB")
        print(f"GPU Memory increase: {(gpu_allocated_final - gpu_allocated_before):.2f} GB")
        
        # nvidia-smi最終確認
        nvidia_info = monitor_nvidia_smi()
        if nvidia_info:
            print(f"\nnvidia-smi: {nvidia_info}")
        
        return True, avg_time, gpu_allocated_final
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ OUT OF MEMORY ERROR")
            print(f"Batch size {batch_size} is too large!")
            
            # メモリ状態を表示
            gpu_total, gpu_allocated, gpu_reserved = get_gpu_memory_usage()
            print(f"GPU Memory at failure: {gpu_allocated:.2f}/{gpu_total:.1f} GB")
            
            return False, 0, gpu_allocated
        else:
            print(f"\n❌ ERROR: {e}")
            return False, 0, 0
    
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0
    
    finally:
        # クリーンアップ
        torch.cuda.empty_cache()

def main():
    """メイン関数"""
    print("="*60)
    print("Batch Size Optimization Benchmark")
    print("="*60)
    
    # システム情報
    print("\n[System Information]")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {total_memory:.1f} GB")
    else:
        print("GPU: Not available")
        return
    
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.total / 1024**3:.1f} GB")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical")
    
    # テストするバッチサイズのリスト
    batch_sizes = [16, 24, 32, 48, 64]
    num_test_games = 5  # 各バッチサイズでテストするゲーム数
    
    print(f"\nTesting batch sizes: {batch_sizes}")
    print(f"Games per test: {num_test_games}")
    
    input("\nPress Enter to start benchmark...")
    
    # 結果を保存
    results = []
    
    for batch_size in batch_sizes:
        success, avg_time, gpu_memory = test_batch_size(batch_size, num_test_games)
        
        results.append({
            'batch_size': batch_size,
            'success': success,
            'avg_time': avg_time,
            'gpu_memory': gpu_memory
        })
        
        if not success:
            print(f"\n⚠️  Batch size {batch_size} failed. Stopping tests.")
            break
        
        # 次のテストの前に少し待機
        time.sleep(2)
    
    # 最終レポート
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n{'Batch Size':<12} {'Status':<10} {'Avg Time':<12} {'Speed':<12} {'GPU Mem':<10}")
    print("-" * 60)
    
    successful_results = [r for r in results if r['success']]
    
    for r in results:
        status = "✅ OK" if r['success'] else "❌ FAILED"
        avg_time_str = f"{r['avg_time']:.2f}s" if r['success'] else "N/A"
        speed_str = f"{1/r['avg_time']:.2f} g/s" if r['success'] else "N/A"
        gpu_mem_str = f"{r['gpu_memory']:.2f} GB" if r['gpu_memory'] > 0 else "N/A"
        
        print(f"{r['batch_size']:<12} {status:<10} {avg_time_str:<12} {speed_str:<12} {gpu_mem_str:<10}")
    
    if successful_results:
        # 最速のバッチサイズを推奨
        fastest = min(successful_results, key=lambda x: x['avg_time'])
        
        print("\n" + "="*60)
        print("RECOMMENDATION")
        print("="*60)
        print(f"✨ Optimal Batch Size: {fastest['batch_size']}")
        print(f"   Average time: {fastest['avg_time']:.2f}s per game")
        print(f"   Speed: {1/fastest['avg_time']:.2f} games/second")
        print(f"   GPU Memory: {fastest['gpu_memory']:.2f} GB")
        
        # auto_tune.pyの更新を提案
        print("\n📝 To apply this setting, update auto_tune.py:")
        print(f"   Change the batch size for your GPU to: {fastest['batch_size']}")

if __name__ == '__main__':
    main()
