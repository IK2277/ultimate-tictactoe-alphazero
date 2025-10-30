# ====================
# ワーカー数ごとの速度比較テスト
# ====================

import time
import multiprocessing as mp

def test_workers(num_workers):
    """指定されたワーカー数でテスト"""
    print(f"\n{'='*60}")
    print(f"Testing with {num_workers} workers")
    print('='*60)
    
    try:
        from self_play_parallel import self_play_parallel
    except ImportError:
        print("Cannot import self_play_parallel")
        return None
    
    test_games = 20  # 少量で高速テスト
    test_pv = 100
    
    print(f"Running {test_games} games with {test_pv} MCTS simulations...")
    print(f"Workers: {num_workers}")
    
    start = time.time()
    self_play_parallel(
        pv_evaluate_count=test_pv, 
        game_count=test_games,
        num_workers=num_workers
    )
    elapsed = time.time() - start
    
    games_per_sec = test_games / elapsed
    est_500 = elapsed * 500 / test_games
    
    print(f"\nResults:")
    print(f"  Time: {elapsed:.1f} seconds")
    print(f"  Speed: {games_per_sec:.2f} games/sec")
    print(f"  Estimated for 500 games: {est_500:.0f} seconds ({est_500/60:.1f} min)")
    
    return elapsed

def main():
    mp.freeze_support()
    
    print("="*60)
    print("Worker Count Speed Comparison")
    print("="*60)
    
    # システムリソース確認
    try:
        from auto_tune import check_system_resources
        check_system_resources()
    except:
        pass
    
    # テストするワーカー数
    worker_counts = [4, 6, 8]
    
    results = {}
    
    for workers in worker_counts:
        try:
            elapsed = test_workers(workers)
            if elapsed:
                results[workers] = elapsed
            
            # 少し待つ
            if workers != worker_counts[-1]:
                print("\nWaiting 3 seconds...")
                time.sleep(3)
        except KeyboardInterrupt:
            print("\n>> Interrupted by user")
            break
        except Exception as e:
            print(f"Error with {workers} workers: {e}")
            continue
    
    # 結果比較
    if results:
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        
        baseline = results.get(4)
        
        for workers, elapsed in sorted(results.items()):
            speedup = baseline / elapsed if baseline else 1.0
            print(f"{workers} workers: {elapsed:.1f}s (speedup: {speedup:.2f}x)")
        
        # 推奨
        fastest = min(results, key=results.get)
        print(f"\n>> Fastest: {fastest} workers")
        
        if fastest >= 8:
            print(">> Recommendation: Use 'aggressive' mode or set WORKER_MODE = 8")
        elif fastest >= 6:
            print(">> Recommendation: Set WORKER_MODE = 6")
        else:
            print(">> Recommendation: Use 'auto' mode or set WORKER_MODE = 4")
        
        print("="*60)
    
    # クリーンアップ
    print("\nCleaning up test files...")
    from pathlib import Path
    data_dir = Path('./data')
    if data_dir.exists():
        for f in data_dir.glob('*.history'):
            f.unlink()
            print(f"Deleted: {f.name}")

if __name__ == '__main__':
    main()
