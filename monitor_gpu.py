"""
GPUとシステムリソースのリアルタイムモニタリング
トレーニング中に別ターミナルで実行して使用状況を監視
"""

import subprocess
import time
import sys
import os

def clear_screen():
    """画面をクリア"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_gpu_info():
    """nvidia-smiでGPU情報を取得"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw,power.limit',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 10:
                    gpus.append({
                        'index': parts[0],
                        'name': parts[1],
                        'temp': parts[2],
                        'gpu_util': parts[3],
                        'mem_util': parts[4],
                        'mem_total': parts[5],
                        'mem_used': parts[6],
                        'mem_free': parts[7],
                        'power_draw': parts[8],
                        'power_limit': parts[9]
                    })
            return gpus
    except Exception as e:
        return None

def format_memory(mb):
    """メモリサイズを読みやすい形式に変換"""
    try:
        mb = float(mb)
        if mb >= 1024:
            return f"{mb/1024:.1f} GB"
        else:
            return f"{mb:.0f} MB"
    except:
        return mb

def draw_bar(value, max_value=100, width=30):
    """プログレスバーを描画"""
    try:
        value = float(value)
        filled = int((value / max_value) * width)
        bar = '█' * filled + '░' * (width - filled)
        return bar
    except:
        return '?' * width

def get_color_code(value, thresholds=(50, 80)):
    """値に応じた色コードを返す（ANSI）"""
    try:
        value = float(value)
        if value < thresholds[0]:
            return '\033[92m'  # 緑
        elif value < thresholds[1]:
            return '\033[93m'  # 黄
        else:
            return '\033[91m'  # 赤
    except:
        return '\033[0m'

def monitor_gpu(interval=1):
    """GPUを継続的にモニタリング"""
    
    print("="*70)
    print("GPU & System Resource Monitor")
    print("="*70)
    print("Press Ctrl+C to stop")
    print()
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            # GPU情報を取得
            gpus = get_gpu_info()
            
            if gpus is None:
                print("❌ nvidia-smi not available or error occurred")
                print("   Make sure NVIDIA drivers are installed")
                time.sleep(5)
                continue
            
            # 画面をクリア（更新回数が多い場合のみ）
            if iteration > 1:
                clear_screen()
            
            # ヘッダー
            print("="*70)
            print(f"GPU Monitor - Update #{iteration} - {time.strftime('%H:%M:%S')}")
            print("="*70)
            print()
            
            # 各GPUの情報を表示
            for gpu in gpus:
                print(f"🎮 GPU {gpu['index']}: {gpu['name']}")
                print("-"*70)
                
                # 温度
                temp = float(gpu['temp'])
                temp_color = get_color_code(temp, (70, 85))
                print(f"🌡️  Temperature:  {temp_color}{temp}°C\033[0m")
                
                # GPU使用率
                gpu_util = float(gpu['gpu_util'])
                gpu_color = get_color_code(gpu_util)
                gpu_bar = draw_bar(gpu_util)
                print(f"⚡ GPU Util:     {gpu_color}{gpu_util:>5.1f}%\033[0m [{gpu_bar}]")
                
                # メモリ使用率
                mem_used = float(gpu['mem_used'])
                mem_total = float(gpu['mem_total'])
                mem_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0
                mem_color = get_color_code(mem_percent)
                mem_bar = draw_bar(mem_percent)
                
                print(f"💾 Memory:       {mem_color}{mem_percent:>5.1f}%\033[0m [{mem_bar}]")
                print(f"   Used/Total:   {format_memory(mem_used)} / {format_memory(mem_total)}")
                print(f"   Free:         {format_memory(gpu['mem_free'])}")
                
                # 電力
                try:
                    power_draw = float(gpu['power_draw'])
                    power_limit = float(gpu['power_limit'])
                    power_percent = (power_draw / power_limit) * 100 if power_limit > 0 else 0
                    power_color = get_color_code(power_percent)
                    power_bar = draw_bar(power_percent)
                    print(f"⚡ Power:        {power_color}{power_draw:>6.1f}W / {power_limit:.1f}W\033[0m [{power_bar}]")
                except:
                    print(f"⚡ Power:        {gpu['power_draw']}W / {gpu['power_limit']}W")
                
                print()
            
            # 追加情報
            print("="*70)
            print("💡 Tips:")
            print("  - GPU Util should be >80% during training for optimal performance")
            print("  - Memory usage shows how much VRAM is being used")
            print("  - Temperature should stay below 85°C")
            print("="*70)
            
            # 更新間隔
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")
        sys.exit(0)

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor GPU usage in real-time')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Update interval in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    monitor_gpu(interval=args.interval)

if __name__ == '__main__':
    main()
