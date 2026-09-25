param(
    [Parameter(Mandatory=$true)][string]$SourceResult,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [string]$HostWorkspace = 'D:\CODEX\project-Grinding.CAE',
    [string]$HistoryTemplate,
    [string]$MaterialProvenance,
    [string]$Python = 'D:\CODEX\project-Grinding.CAE\.venv\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
$bridgeOldPath = $env:PYTHONPATH
$bridgeOldUtf8 = $env:PYTHONUTF8
$bridgeOldBytecode = $env:PYTHONDONTWRITEBYTECODE
try {
    $env:PYTHONPATH = (Join-Path $PSScriptRoot 'src') + ';' + (Join-Path $HostWorkspace 'src')
    $env:PYTHONUTF8 = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $bridgeArguments = @('-m','grindcae380.legacy_bridge',$SourceResult,'--output-dir',$OutputDirectory)
    if ($HistoryTemplate) {
        $bridgeArguments += @('--history-template',$HistoryTemplate,'--material-provenance',$MaterialProvenance)
    }
    & $Python @bridgeArguments
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $bridgeOldPath
    $env:PYTHONUTF8 = $bridgeOldUtf8
    $env:PYTHONDONTWRITEBYTECODE = $bridgeOldBytecode
}
