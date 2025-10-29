# ====================
# GPU情報確認スクリプト
# ====================

import torch

print("=" * 60)
print("PyTorch & CUDA 情報")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print()
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  - Compute Capability: {props.major}.{props.minor}")
        print(f"  - Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  - Multi Processor Count: {props.multi_processor_count}")
        
        if i == torch.cuda.current_device():
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  - Memory Allocated: {mem_allocated:.2f} GB")
            print(f"  - Memory Reserved: {mem_reserved:.2f} GB")
        print()
    
    # cuDNN最適化設定
    print("cuDNN Settings:")
    print(f"  - Benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  - Deterministic: {torch.backends.cudnn.deterministic}")
else:
    print("CUDA is not available. Running on CPU.")

print("=" * 60)
