$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $projectRoot
$previousDatabase = $env:DATABASE_PATH
$previousPublicMode = $env:PULSE_PUBLIC_MODE
try {
    $env:DATABASE_PATH = Join-Path $projectRoot '.data/test-stand-2026-11-04/test.db'
    $env:PULSE_PUBLIC_MODE = '0'
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
} finally {
    $env:DATABASE_PATH = $previousDatabase
    $env:PULSE_PUBLIC_MODE = $previousPublicMode
}
