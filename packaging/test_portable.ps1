param(
    [Parameter(Mandatory = $true)]
    [string]$DistributionDirectory,
    [string]$ExpectedVersion = "",
    [string]$SmokeOutputDirectory,
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $scriptRoot "..")).Path
$distribution = (Resolve-Path -LiteralPath $DistributionDirectory).Path
if (-not $ExpectedVersion) {
    $versionFile = Join-Path $distribution "VERSION.txt"
    if (-not (Test-Path -LiteralPath $versionFile)) {
        throw "Pass -ExpectedVersion or provide VERSION.txt in the distribution."
    }
    $versionLine = (Get-Content -LiteralPath $versionFile -Encoding UTF8 | Select-Object -First 1)
    if ($versionLine -notmatch '^GrindCAE ([0-9A-Za-z][0-9A-Za-z.+-]*)$') {
        throw "VERSION.txt does not contain a valid GrindCAE version."
    }
    $ExpectedVersion = $Matches[1]
}
if ($ExpectedVersion -notmatch '^[0-9A-Za-z][0-9A-Za-z.+-]*$') {
    throw "ExpectedVersion is invalid: $ExpectedVersion"
}
$expectedName = "GrindCAE-$ExpectedVersion-Windows-x64"
if ((Split-Path -Leaf $distribution) -ne $expectedName) {
    throw "Distribution directory must be named $expectedName."
}

$guiExe = Join-Path $distribution "GrindCAE.exe"
$diagnosticsExe = Join-Path $distribution "GrindCAE-Diagnostics.exe"
foreach ($required in @($guiExe, $diagnosticsExe, (Join-Path $distribution "_internal"))) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required portable artifact is missing: $required"
    }
}

$forbiddenNames = @(
    ".git", ".venv", ".venv-build", "tests", "outputs", "logs", "release",
    "build", "dist", "__pycache__", ".pytest_cache"
)
$forbidden = Get-ChildItem -LiteralPath $distribution -Recurse -Force | Where-Object {
    $forbiddenNames -contains $_.Name
}
if ($forbidden) {
    throw "Forbidden development content found: $($forbidden.FullName -join ', ')"
}

& $diagnosticsExe
if ($LASTEXITCODE -ne 0) {
    throw "Portable diagnostics failed with exit code $LASTEXITCODE."
}

if (-not $SkipSmoke) {
    if (-not $SmokeOutputDirectory) {
        $SmokeOutputDirectory = Join-Path (Split-Path -Parent $distribution) "portable-smoke-output"
    }
    $smoke = [System.IO.Path]::GetFullPath($SmokeOutputDirectory)
    if (Test-Path -LiteralPath $smoke) {
        throw "Smoke output already exists; choose a new directory: $smoke"
    }
    & $diagnosticsExe --smoke-test --output-dir $smoke
    if ($LASTEXITCODE -ne 0) {
        throw "Portable smoke test failed with exit code $LASTEXITCODE."
    }
    $report = Join-Path $smoke "portable_smoke_report.json"
    if (-not (Test-Path -LiteralPath $report)) {
        throw "Portable smoke report was not generated."
    }
    $payload = Get-Content -LiteralPath $report -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $payload.passed) {
        throw "Portable smoke report did not pass."
    }
}

Write-Host "Portable distribution verification PASS: $distribution"
