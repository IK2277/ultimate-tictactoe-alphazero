# ====================
# GPU最適化とバッチサイズの自動調整
# ====================

import torch
import psutil
import os

def get_optimal_batch_size():
    """
    システムのメモリとGPUの状態に基づいて最適なバッチサイズを決定
    """
    if not torch.cuda.is_available():
        # CPU使用時は控えめに
        return 8
    
    try:
        # GPU情報の取得
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        
        # RTX 4070 Tiは12GB VRAM
        # モデルサイズが約50MB、推論時の追加メモリを考慮
        if total_memory > 11 * 1024**3:  # 11GB以上
            # 大容量VRAM: バッチサイズ32
            return 32
        elif total_memory > 7 * 1024**3:  # 7GB以上
            # 中容量VRAM: バッチサイズ16
            return 16
        else:
            # 小容量VRAM: バッチサイズ8
            return 8
    except:
        return 8  # デフォルト値

def get_optimal_num_workers(aggressive=False):
    """
    CPUコア数に基づいて最適なワーカー数を決定
    Windows環境とプロセス起動コストを考慮
    
    Args:
        aggressive: Trueの場合、より多くのワーカーを使用（実験的）
    """
    cpu_count = psutil.cpu_count(logical=False)  # 物理コア数
    
    if cpu_count is None:
        cpu_count = os.cpu_count() or 4
    
    if aggressive:
        # 積極的な設定（より多くのワーカー）
        if torch.cuda.is_available():
            # GPU使用時: 物理コア数の60-75%（最低4、最大10）
            optimal = max(4, min(10, int(cpu_count * 0.7)))
            return optimal
        else:
            # CPU使用時: 物理コア数の75%（最低4、最大12）
            return max(4, min(12, int(cpu_count * 0.75)))
    else:
        # 保守的な設定（安定重視）
        # 経験則: 4-6ワーカーが最適（12コアシステムで）
        if torch.cuda.is_available():
            # GPU使用時: 物理コア数の40-50%（最低2、最大6）
            optimal = max(2, min(6, int(cpu_count * 0.4)))
            return optimal
        else:
            # CPU使用時: 物理コア数の50%（最低2、最大8）
            return max(2, min(8, int(cpu_count * 0.5)))

def check_system_resources():
    """
    システムリソースの状態を表示
    """
    print("=== System Resources ===")
    
    # CPU情報
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    print(f"CPU Cores: {cpu_count_physical} physical, {cpu_count_logical} logical")
    
    # メモリ情報
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.total / 1024**3:.1f}GB total, {memory.available / 1024**3:.1f}GB available ({memory.percent}% used)")
    
    # GPU情報
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {total_memory / 1024**3:.1f}GB total")
        print(f"      {allocated_memory / 1024**3:.2f}GB allocated")
        print(f"      {reserved_memory / 1024**3:.2f}GB reserved")
    else:
        print("GPU: Not available")
    
    print("========================")
    
    # 推奨設定
    batch_size = get_optimal_batch_size()
    num_workers_conservative = get_optimal_num_workers(aggressive=False)
    num_workers_aggressive = get_optimal_num_workers(aggressive=True)
    
    print(f"\n>> Recommended settings:")
    print(f"   MCTS Batch Size: {batch_size}")
    print(f"   Parallel Workers (Conservative): {num_workers_conservative}")
    print(f"   Parallel Workers (Aggressive): {num_workers_aggressive}")
    
    return batch_size, num_workers_conservative

if __name__ == '__main__':
    check_system_resources()
