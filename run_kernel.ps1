param(
    [Parameter(Mandatory=$true)][string]$Case,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [string]$Python = 'D:\CODEX\project-Grinding.CAE\.venv\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python runtime missing: $Python. Supply -Python with an existing compatible environment."
}
$oldPath380 = $env:PYTHONPATH
$oldBackend380 = $env:MPLBACKEND
$oldUtf8380 = $env:PYTHONUTF8
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    $env:MPLBACKEND = 'Agg'
    $env:PYTHONUTF8 = '1'
    & $Python -m grindcae380 $Case --output-dir $OutputDirectory
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $oldPath380
    $env:MPLBACKEND = $oldBackend380
    $env:PYTHONUTF8 = $oldUtf8380
}
