$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $projectRoot
$runStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$evidencePath = Join-Path $projectRoot ('.data/testing-debt-runs/' + $runStamp)
New-Item -ItemType Directory -Path $evidencePath -Force | Out-Null
$previousEvidence = $env:PULSE_TEST_EVIDENCE_DIR
try {
    $env:PULSE_TEST_EVIDENCE_DIR = Join-Path $evidencePath 'evidence'
    python -m pytest tests/test_practice_cases.py -v --junitxml="$evidencePath/practice.xml" 2>&1 | Tee-Object -FilePath "$evidencePath/pytest.txt"
    if ($LASTEXITCODE -ne 0) { throw 'A case failed. See the saved log.' }
} finally {
    $env:PULSE_TEST_EVIDENCE_DIR = $previousEvidence
}
Write-Output $evidencePath
