$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$frontendRoot = Join-Path $projectRoot 'coincast-pulse\dist'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "CoinCast Python bulunamadı: $pythonPath"
}

function Test-CoinCastPort {
    param([int]$Port)
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $connection = $client.ConnectAsync('127.0.0.1', $Port)
        if (-not $connection.Wait(700)) { return $false }
        return $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

if (-not (Test-CoinCastPort -Port 8000)) {
    Start-Process -FilePath $pythonPath `
        -ArgumentList @('-m', 'uvicorn', 'ml_backend.main:app', '--host', '127.0.0.1', '--port', '8000') `
        -WorkingDirectory $projectRoot -WindowStyle Hidden
}

if (-not (Test-CoinCastPort -Port 5173)) {
    Start-Process -FilePath $pythonPath `
        -ArgumentList @('-m', 'http.server', '5173', '--bind', '127.0.0.1', '--directory', 'coincast-pulse\dist') `
        -WorkingDirectory $projectRoot -WindowStyle Hidden
}

$trackerPidPath = Join-Path $projectRoot 'data\forecast_tracker.pid'
$trackerRunning = $false
if (Test-Path -LiteralPath $trackerPidPath) {
    try {
        $trackerPid = [int](Get-Content -LiteralPath $trackerPidPath -Raw)
        $trackerRunning = $null -ne (Get-Process -Id $trackerPid -ErrorAction Stop)
    }
    catch {
        Remove-Item -LiteralPath $trackerPidPath -Force -ErrorAction SilentlyContinue
    }
}

if (-not $trackerRunning) {
    Start-Process -FilePath $pythonPath `
        -ArgumentList @('-m', 'src.run_forecast_tracker', '--every-seconds', '3600') `
        -WorkingDirectory $projectRoot -WindowStyle Hidden
}
