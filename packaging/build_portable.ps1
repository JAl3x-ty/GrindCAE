param(
    [string]$Python = "",
    [string]$ReleaseRoot = "",
    [switch]$Clean,
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $scriptRoot "..")).Path

if (-not $Python) {
    $candidate = Join-Path $repositoryRoot ".venv-build\Scripts\python.exe"
    if (Test-Path -LiteralPath $candidate) {
        $Python = $candidate
    } else {
        throw "Pass -Python or create .venv-build with Python 3.11 x64."
    }
}
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$pythonInfo = & $pythonPath -c "import platform,sys; print(f'{sys.version_info.major}.{sys.version_info.minor}|{platform.machine()}|{sys.maxsize > 2**32}')"
if ($LASTEXITCODE -ne 0 -or $pythonInfo.Trim() -ne "3.11|AMD64|True") {
    throw "Build Python must be CPython 3.11 x64 AMD64; found $pythonInfo"
}

$sourceVersion = & $pythonPath -c "import sys; sys.path.insert(0, r'$repositoryRoot\src'); import grindcae; print(grindcae.__version__)"
if ($LASTEXITCODE -ne 0 -or -not $sourceVersion) {
    throw "Unable to read the GrindCAE source version."
}
$sourceVersion = $sourceVersion.Trim()
$metadataVersion = & $pythonPath -c "import importlib.metadata as m; print(m.version('grindcae-literature-kernel'))"
if ($LASTEXITCODE -ne 0 -or -not $metadataVersion) {
    throw "Unable to read the installed GrindCAE metadata version."
}
$metadataVersion = $metadataVersion.Trim()
if ($sourceVersion -ne $metadataVersion) {
    throw "Version mismatch: source=$sourceVersion metadata=$metadataVersion"
}

$distributionName = "GrindCAE-$sourceVersion-Windows-x64"
$buildRoot = Join-Path $repositoryRoot "build"
$distRoot = Join-Path $repositoryRoot "dist"
if (-not $ReleaseRoot) {
    $ReleaseRoot = Join-Path $repositoryRoot "release"
}
$release = [System.IO.Path]::GetFullPath($ReleaseRoot)
$distribution = Join-Path $release $distributionName
$zipPath = Join-Path $release "$distributionName.zip"
$shaPath = "$zipPath.sha256"
$manifestPath = Join-Path $release "$distributionName-manifest.json"
$smokeOutput = Join-Path $release "$distributionName-smoke"

$gitStatus = @(git -C $repositoryRoot status --short)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect the Git worktree."
}
if ($gitStatus.Count -gt 0) {
    $details = $gitStatus -join [Environment]::NewLine
    throw "Formal portable build requires a clean Git worktree. Commit or otherwise resolve these files first:`n$details"
}
$gitCommit = (git -C $repositoryRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or -not $gitCommit) {
    throw "Unable to resolve the Git commit for the portable release."
}

$rootResolved = [System.IO.Path]::GetFullPath($repositoryRoot).TrimEnd('\')
$rootPrefix = "$rootResolved\"
$allowedCleanTargets = @($buildRoot, $distRoot, $distribution, $zipPath, $shaPath, $manifestPath) | ForEach-Object {
    [System.IO.Path]::GetFullPath($_)
}
$allowedCleanTargets += [System.IO.Path]::GetFullPath($smokeOutput)
foreach ($target in $allowedCleanTargets) {
    if (-not $target.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing out-of-repository clean target: $target"
    }
}
if ($Clean) {
    foreach ($target in $allowedCleanTargets) {
        if (Test-Path -LiteralPath $target) {
            Remove-Item -LiteralPath $target -Recurse -Force
        }
    }
}
New-Item -ItemType Directory -Force -Path $release | Out-Null

& $pythonPath -m PyInstaller --noconfirm --clean --distpath $distRoot --workpath $buildRoot (Join-Path $scriptRoot "GrindCAE.spec")
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE."
}
$builtDistribution = Join-Path $distRoot $distributionName
if (-not (Test-Path -LiteralPath $builtDistribution)) {
    throw "PyInstaller did not create $builtDistribution"
}
Copy-Item -LiteralPath $builtDistribution -Destination $distribution -Recurse

& $pythonPath (Join-Path $scriptRoot "release_tools.py") prepare --destination $distribution --repository-root $repositoryRoot
if ($LASTEXITCODE -ne 0) {
    throw "Release metadata preparation failed."
}
Copy-Item -LiteralPath (Join-Path $repositoryRoot "docs\WINDOWS_PORTABLE_QUICKSTART.md") -Destination (Join-Path $distribution "README_PORTABLE.md")

$testArguments = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $scriptRoot "test_portable.ps1"),
    "-DistributionDirectory", $distribution, "-ExpectedVersion", $sourceVersion,
    "-SmokeOutputDirectory", $smokeOutput
)
if ($SkipSmoke) {
    $testArguments += "-SkipSmoke"
}
& powershell @testArguments
if ($LASTEXITCODE -ne 0) {
    throw "Portable distribution verification failed."
}

$literatureSmoke = Join-Path $release "$distributionName-literature-smoke"
if (Test-Path -LiteralPath $literatureSmoke) {
    throw "Literature smoke output already exists; choose a new ReleaseRoot."
}
& (Join-Path $distribution "GrindCAE-Diagnostics.exe") --literature-smoke --output-dir $literatureSmoke
if ($LASTEXITCODE -ne 0) {
    throw "Frozen literature/J2 and Tk smoke failed."
}

if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -LiteralPath $distribution -DestinationPath $zipPath -CompressionLevel Optimal
$zipHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
"$zipHash  $(Split-Path -Leaf $zipPath)" | Set-Content -LiteralPath $shaPath -Encoding ascii

& $pythonPath (Join-Path $scriptRoot "release_tools.py") manifest --distribution $distribution --zip $zipPath --output $manifestPath --git-commit $gitCommit --clean-machine-status pending
if ($LASTEXITCODE -ne 0) {
    throw "Release manifest generation failed."
}

Write-Host "Portable release candidate generated: $distribution"
Write-Host "ZIP: $zipPath"
Write-Host "SHA-256: $zipHash"
Write-Host "Clean-machine gate: pending"
