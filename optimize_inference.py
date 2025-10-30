# ====================
# PyTorchモデル推論の最適化設定
# ====================

import torch
import os

def optimize_pytorch():
    """
    PyTorchの推論パフォーマンスを最適化
    """
    # CUDAが利用可能な場合の最適化
    if torch.cuda.is_available():
        # TensorFloat-32（TF32）を有効化（Ampere世代以降のGPU）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # cuDNNのベンチマークモードを有効化（入力サイズが固定の場合に有効）
        torch.backends.cudnn.benchmark = True
        
        # 決定論的アルゴリズムを無効化（速度優先）
        torch.backends.cudnn.deterministic = False
        
        print(">> CUDA optimizations enabled:")
        print(f"   - TF32: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"   - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    
    # CPUの最適化
    # OpenMPスレッド数の設定（環境変数）
    if 'OMP_NUM_THREADS' not in os.environ:
        num_threads = max(1, torch.get_num_threads() // 2)  # ハイパースレッディング考慮
        torch.set_num_threads(num_threads)
        print(f">> PyTorch CPU threads: {num_threads}")
    
    # メモリアロケータの最適化
    if torch.cuda.is_available():
        # PyTorchのメモリキャッシュを有効活用
        torch.cuda.empty_cache()
    
    return True

def compile_model(model, disable_for_multiprocessing=False):
    """
    PyTorch 2.0+のtorch.compile()を使用してモデルを最適化
    （利用可能な場合）
    
    注意: torch.compile()はマルチプロセスで問題を起こす場合があります。
    並列処理では無効化を推奨します。
    
    Args:
        model: コンパイルするモデル
        disable_for_multiprocessing: マルチプロセス環境では無効化（推奨: True）
    """
    if disable_for_multiprocessing:
        print(">> torch.compile() disabled for multiprocessing safety")
        return model
    
    try:
        # PyTorch 2.0以降でtorch.compileが利用可能
        if hasattr(torch, 'compile'):
            print(">> Compiling model with torch.compile()...")
            # 推論モード用の最適化
            compiled_model = torch.compile(
                model,
                mode='reduce-overhead',  # 推論時のオーバーヘッド削減
                fullgraph=True,           # グラフ全体をコンパイル
                dynamic=False             # 入力サイズ固定で最適化
            )
            print(">> Model compilation successful!")
            return compiled_model
        else:
            print(">> torch.compile() not available (PyTorch < 2.0)")
            return model
    except Exception as e:
        print(f">> Model compilation failed: {e}")
        print(">> Using uncompiled model")
        return model

# 自動最適化を実行
if __name__ == '__main__':
    print("Testing PyTorch optimizations...")
    optimize_pytorch()
    
    # モデルのコンパイルテスト
    from dual_network import DualNetwork, device
    model = DualNetwork().to(device)
    model.eval()
    
    compiled_model = compile_model(model)
    print(f"Model type: {type(compiled_model)}")
