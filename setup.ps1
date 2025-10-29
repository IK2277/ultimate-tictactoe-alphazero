# Ultimate Tic-Tac-Toe セットアップスクリプト
# 新しいPCでの初期セットアップを自動化

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ultimate Tic-Tac-Toe セットアップ" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Python バージョン確認
Write-Host "`n[1/6] Pythonバージョン確認..." -ForegroundColor Yellow
python --version

# 2. CUDA 確認
Write-Host "`n[2/6] CUDA確認..." -ForegroundColor Yellow
try {
    nvidia-smi
    $CUDA_AVAILABLE = $true
} catch {
    Write-Host "⚠️  NVIDIA GPU not found. CPU mode will be used." -ForegroundColor Yellow
    $CUDA_AVAILABLE = $false
}

# 3. Pythonパッケージのインストール
Write-Host "`n[3/6] Pythonパッケージのインストール..." -ForegroundColor Yellow

if ($CUDA_AVAILABLE) {
    Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Green
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "Installing PyTorch (CPU only)..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio
}

pip install numpy pybind11

# 4. 初期モデルの作成
Write-Host "`n[4/6] 初期モデルの作成..." -ForegroundColor Yellow
if (-not (Test-Path "model/best.pth")) {
    python dual_network.py
    Write-Host "✅ 初期モデルを作成しました" -ForegroundColor Green
} else {
    Write-Host "✅ モデルは既に存在します" -ForegroundColor Green
}

# 5. C++拡張のビルド（オプション）
Write-Host "`n[5/6] C++拡張のビルド（オプション）..." -ForegroundColor Yellow
$response = Read-Host "C++拡張をビルドしますか？（30-50倍高速化） [Y/n]"

if ($response -eq "" -or $response -eq "Y" -or $response -eq "y") {
    Write-Host "Building C++ extension..." -ForegroundColor Green
    cd cpp
    try {
        pip install -e .
        Write-Host "✅ C++拡張のビルドに成功しました" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  C++拡張のビルドに失敗しました。Python版を使用します。" -ForegroundColor Yellow
        Write-Host "Visual Studio 2019以降が必要です。" -ForegroundColor Yellow
    }
    cd ..
} else {
    Write-Host "C++拡張をスキップしました。Python版を使用します。" -ForegroundColor Yellow
}

# 6. 動作確認
Write-Host "`n[6/6] 動作確認..." -ForegroundColor Yellow

Write-Host "`nGPU確認:" -ForegroundColor Cyan
python check_gpu.py

if (Test-Path "cpp/uttt_cpp*.pyd") {
    Write-Host "`nC++拡張確認:" -ForegroundColor Cyan
    python -c "import uttt_cpp; print('✅ C++拡張が利用可能です')"
}

# セットアップ完了
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "✅ セットアップ完了！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "`n次のステップ:" -ForegroundColor Cyan
Write-Host "  1. 学習開始: python train_cycle.py" -ForegroundColor White
Write-Host "  2. 人間対戦: python human_play.py" -ForegroundColor White
Write-Host "  3. テスト実行: python test_cpp_mcts.py" -ForegroundColor White

Write-Host "`n詳細はREADME.mdを参照してください。" -ForegroundColor Yellow
