# ====================
# シンプルなセルフプレイ速度テスト
# ====================

import time
from pathlib import Path

def test_serial():
    """シリアル版のテスト"""
    print("\n" + "="*60)
    print("Testing SERIAL Self-Play (従来版)")
    print("="*60)
    
    try:
        import uttt_cpp
        from self_play_cpp import self_play
    except ImportError:
        print("C++ backend not available")
        return None
    
    test_games = 25  # 少量でテスト
    test_pv = 100
    
    print(f"Running {test_games} games with {test_pv} MCTS simulations...")
    
    start = time.time()
    self_play(pv_evaluate_count=test_pv, game_count=test_games)
    elapsed = time.time() - start
    
    print(f"\nSerial Time: {elapsed:.2f} seconds")
    print(f"Games/sec: {test_games/elapsed:.2f}")
    print(f"Estimated time for 500 games: {elapsed * 500 / test_games:.1f} seconds")
    
    return elapsed

def test_parallel():
    """並列版のテスト"""
    print("\n" + "="*60)
    print("Testing PARALLEL Self-Play (並列版)")
    print("="*60)
    
    import multiprocessing as mp
    mp.freeze_support()
    
    try:
        import uttt_cpp
        from self_play_parallel import self_play_parallel
    except ImportError:
        from self_play_parallel import self_play_parallel
    
    test_games = 25  # 少量でテスト
    test_pv = 100
    
    print(f"Running {test_games} games with {test_pv} MCTS simulations...")
    
    start = time.time()
    self_play_parallel(pv_evaluate_count=test_pv, game_count=test_games)
    elapsed = time.time() - start
    
    print(f"\nParallel Time: {elapsed:.2f} seconds")
    print(f"Games/sec: {test_games/elapsed:.2f}")
    print(f"Estimated time for 500 games: {elapsed * 500 / test_games:.1f} seconds")
    
    return elapsed

def main():
    print("="*60)
    print("Self-Play Speed Comparison")
    print("="*60)
    
    # システム情報
    try:
        from auto_tune import check_system_resources
        check_system_resources()
    except:
        pass
    
    # シリアル版テスト
    serial_time = test_serial()
    
    if serial_time is None:
        print("\nCannot run tests without C++ backend")
        return
    
    # 少し待つ
    print("\nWaiting 3 seconds...")
    time.sleep(3)
    
    # 並列版テスト
    parallel_time = test_parallel()
    
    # 結果
    if serial_time and parallel_time:
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Serial:   {serial_time:.2f} sec")
        print(f"Parallel: {parallel_time:.2f} sec")
        speedup = serial_time / parallel_time
        print(f"Speedup:  {speedup:.2f}x")
        
        if speedup > 1:
            print(f"\n>> Parallel is {speedup:.2f}x FASTER!")
        else:
            print(f"\n>> Serial is {1/speedup:.2f}x faster (parallel overhead too high)")
        
        print("="*60)
    
    # テストファイルをクリーンアップ
    print("\nCleaning up test files...")
    data_dir = Path('./data')
    if data_dir.exists():
        for f in data_dir.glob('*.history'):
            f.unlink()
            print(f"Deleted: {f.name}")

if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    
    main()
