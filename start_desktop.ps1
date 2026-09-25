param([string]$Python = 'D:\CODEX\project-Grinding.CAE\.venv\Scripts\python.exe')
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python runtime missing: $Python" }
$desktopOldPath = $env:PYTHONPATH
$desktopOldUtf8 = $env:PYTHONUTF8
$desktopOldBytecode = $env:PYTHONDONTWRITEBYTECODE
try {
    $env:PYTHONPATH = (Join-Path $PSScriptRoot 'src') + ';D:\CODEX\project-Grinding.CAE\src'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    & $Python -m grindcae380.desktop
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $desktopOldPath
    $env:PYTHONUTF8 = $desktopOldUtf8
    $env:PYTHONDONTWRITEBYTECODE = $desktopOldBytecode
}
