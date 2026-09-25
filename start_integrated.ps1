param([string]$Python = 'D:\CODEX\project-Grinding.CAE\.venv\Scripts\python.exe')
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python runtime missing: $Python" }
$integratedOldPath = $env:PYTHONPATH
$integratedOldUtf8 = $env:PYTHONUTF8
$integratedOldBytecode = $env:PYTHONDONTWRITEBYTECODE
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    & $Python -m grindcae.gui
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $integratedOldPath
    $env:PYTHONUTF8 = $integratedOldUtf8
    $env:PYTHONDONTWRITEBYTECODE = $integratedOldBytecode
}
