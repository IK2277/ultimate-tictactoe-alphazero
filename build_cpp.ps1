# Build and Install C++ Extension
# C++拡張モジュールのビルドとインストール

Write-Host "Building C++ extension for Ultimate Tic-Tac-Toe..." -ForegroundColor Cyan

# 1. pybind11のインストール確認
Write-Host "`n[1/4] Checking pybind11..." -ForegroundColor Yellow
try {
    python -c "import pybind11; print(f'pybind11 version: {pybind11.__version__}')"
} catch {
    Write-Host "Installing pybind11..." -ForegroundColor Yellow
    pip install pybind11
}

# 2. Visual Studio Build Toolsの確認
Write-Host "`n[2/4] Checking Visual Studio Build Tools..." -ForegroundColor Yellow
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -property installationPath
    Write-Host "Visual Studio found at: $vsPath" -ForegroundColor Green
} else {
    Write-Host "Warning: Visual Studio not found. You may need to install Visual Studio 2019 or later with C++ support." -ForegroundColor Red
    Write-Host "Download from: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Yellow
}

# 3. C++拡張のビルド
Write-Host "`n[3/4] Building C++ extension..." -ForegroundColor Yellow
Set-Location cpp
try {
    python setup.py build_ext --inplace
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Build successful!" -ForegroundColor Green
    } else {
        Write-Host "Build failed. Please check error messages above." -ForegroundColor Red
        Set-Location ..
        exit 1
    }
} catch {
    Write-Host "Build error: $_" -ForegroundColor Red
    Set-Location ..
    exit 1
}

# 4. インストール
Write-Host "`n[4/4] Installing extension..." -ForegroundColor Yellow
pip install -e .

Set-Location ..

# 5. 動作確認
Write-Host "`n[5/5] Testing installation..." -ForegroundColor Yellow
python -c "import uttt_cpp; print('✅ C++ module imported successfully'); state = uttt_cpp.State(); print(f'✅ Legal actions: {len(state.legal_actions())}'); print('✅ All tests passed!')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "✅ C++ extension installed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "`nYou can now use:" -ForegroundColor Cyan
    Write-Host "  - uttt_cpp.State() for fast game logic" -ForegroundColor White
    Write-Host "  - pv_mcts_cpp.py for fast MCTS" -ForegroundColor White
    Write-Host "  - self_play_cpp.py for fast self-play" -ForegroundColor White
} else {
    Write-Host "`n❌ Installation test failed" -ForegroundColor Red
}
