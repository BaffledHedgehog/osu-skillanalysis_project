$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $projectRoot
$evidencePath = Join-Path $projectRoot '.data/test-evidence/2026-11-04'
New-Item -ItemType Directory -Path $evidencePath -Force | Out-Null
$previousEvidence = $env:PULSE_TEST_EVIDENCE_DIR
try {
    $env:PULSE_TEST_EVIDENCE_DIR = Join-Path $evidencePath 'automated'
    python -m pytest tests/test_practice_cases.py -v --junitxml="$evidencePath/practice.xml" 2>&1 | Tee-Object -FilePath "$evidencePath/pytest.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Some cases failed. See pytest.txt.' }
} finally {
    $env:PULSE_TEST_EVIDENCE_DIR = $previousEvidence
}
