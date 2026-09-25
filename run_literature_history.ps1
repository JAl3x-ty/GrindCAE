param(
    [Parameter(Mandatory=$true)][string]$Case,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [string]$Python = 'D:\CODEX\project-Grinding.CAE\.venv\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
$historyOldPath = $env:PYTHONPATH
$historyOldUtf8 = $env:PYTHONUTF8
$historyOldBytecode = $env:PYTHONDONTWRITEBYTECODE
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    & $Python -m grindcae.literature_history $Case --output-dir $OutputDirectory
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $historyOldPath
    $env:PYTHONUTF8 = $historyOldUtf8
    $env:PYTHONDONTWRITEBYTECODE = $historyOldBytecode
}
