$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "CoinCast Python bulunamadı: $pythonPath"
}

Set-Location -LiteralPath $projectRoot
& $pythonPath -m src.run_daily_report
exit $LASTEXITCODE
