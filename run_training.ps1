# UTF-8でログを記録しながら訓練を実行
# Tee-ObjectのUTF-16問題を回避

# UTF-8コンソール出力を設定
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

Write-Host ">> Starting training with UTF-8 log output..." -ForegroundColor Green
Write-Host ">> Log file: training_log.txt" -ForegroundColor Cyan
Write-Host ""

# Pythonの出力をUTF-8でキャプチャ
python train_cycle.py 2>&1 | ForEach-Object {
    # コンソールに表示
    Write-Host $_
    
    # UTF-8でファイルに追記
    Add-Content -Path "training_log.txt" -Value $_ -Encoding UTF8
}

Write-Host ""
Write-Host ">> Training completed!" -ForegroundColor Green
