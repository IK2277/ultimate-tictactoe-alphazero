# ====================
# セルフプレイ速度の比較テスト
# ====================

import time
from pathlib import Path

def test_serial_selfplay():
    """従来のシリアル版セルフプレイのテスト"""
    print("\n=== Testing Serial Self-Play ===")
    
    try:
        import uttt_cpp
        from self_play_cpp import self_play
        print("Using C++ backend (serial)")
    except ImportError:
        print("C++ backend not available, skipping serial test")
        return None
    
    start_time = time.time()
    
    # 少量のゲームでテスト（デフォルトは50ゲーム）
    test_game_count = 50
    test_pv_count = 100
    
    print(f"Running {test_game_count} games with {test_pv_count} MCTS simulations...")
    self_play(pv_evaluate_count=test_pv_count, game_count=test_game_count)
    
    elapsed = time.time() - start_time
    print(f"Serial time: {elapsed:.2f} seconds")
    print(f"Games per second: {test_game_count / elapsed:.2f}")
    
    return elapsed

def test_parallel_selfplay():
    """並列版セルフプレイのテスト"""
    print("\n=== Testing Parallel Self-Play ===")
    
    try:
        import uttt_cpp
        from self_play_parallel import self_play_parallel
        print("Using C++ backend (parallel)")
    except ImportError:
        from self_play_parallel import self_play_parallel
        print("Using Python backend (parallel)")
    
    start_time = time.time()
    
    # 少量のゲームでテスト
    test_game_count = 50
    test_pv_count = 100
    
    print(f"Running {test_game_count} games with {test_pv_count} MCTS simulations...")
    self_play_parallel(pv_evaluate_count=test_pv_count, game_count=test_game_count)
    
    elapsed = time.time() - start_time
    print(f"Parallel time: {elapsed:.2f} seconds")
    print(f"Games per second: {test_game_count / elapsed:.2f}")
    
    return elapsed

def benchmark():
    """速度比較ベンチマーク"""
    print("=" * 60)
    print("Self-Play Performance Benchmark")
    print("=" * 60)
    
    # システムリソース確認
    try:
        from auto_tune import check_system_resources
        check_system_resources()
    except:
        print("auto_tune.py not available")
    
    print("\n" + "=" * 60)
    print("Starting benchmark tests...")
    print("=" * 60)
    
    # シリアル版テスト
    serial_time = test_serial_selfplay()
    
    # 少し待つ
    time.sleep(2)
    
    # 並列版テスト
    parallel_time = test_parallel_selfplay()
    
    # 結果比較
    if serial_time and parallel_time:
        print("\n" + "=" * 60)
        print("Benchmark Results")
        print("=" * 60)
        print(f"Serial version:   {serial_time:.2f} seconds")
        print(f"Parallel version: {parallel_time:.2f} seconds")
        speedup = serial_time / parallel_time
        print(f"Speedup: {speedup:.2f}x faster")
        print("=" * 60)
    
    # クリーンアップ（テスト用のhistoryファイルを削除）
    import os
    data_dir = Path('./data')
    if data_dir.exists():
        history_files = sorted(data_dir.glob('*.history'))
        if len(history_files) > 0:
            print(f"\nNote: {len(history_files)} test history file(s) created in ./data/")
            print("You may want to delete them before actual training.")

if __name__ == '__main__':
    # Windows対応
    import multiprocessing as mp
    mp.freeze_support()
    
    benchmark()
