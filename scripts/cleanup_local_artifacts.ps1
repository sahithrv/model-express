param(
    [switch]$Apply,
    [int]$MaxAgeDays = 14,
    [double]$MaxBytesGB = 5
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$cutoff = (Get-Date).AddDays(-[Math]::Abs($MaxAgeDays))
$maxBytes = [int64]($MaxBytesGB * 1GB)

function Get-TreeBytes {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    $item = Get-Item -LiteralPath $Path -Force
    if (-not $item.PSIsContainer) { return [int64]$item.Length }
    $sum = 0L
    Get-ChildItem -LiteralPath $Path -Recurse -Force -File -ErrorAction SilentlyContinue | ForEach-Object {
        $sum += [int64]$_.Length
    }
    return $sum
}

function Test-UnderRoot {
    param([string]$Path, [string]$Root)
    $resolvedPath = Resolve-Path -LiteralPath $Path -ErrorAction SilentlyContinue
    if (-not $resolvedPath) { return $false }
    $resolvedRoot = Resolve-Path -LiteralPath $Root
    return $resolvedPath.Path.StartsWith($resolvedRoot.Path, [System.StringComparison]::OrdinalIgnoreCase)
}

function Add-Candidates {
    param([string]$Root, [ref]$Candidates)
    if (-not (Test-Path -LiteralPath $Root)) { return }
    if (-not (Test-UnderRoot -Path $Root -Root $repoRoot)) { return }
    Get-ChildItem -LiteralPath $Root -Force -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.Name.StartsWith(".")) { return }
        if ($_.Name.EndsWith(".lock")) { return }
        $Candidates.Value += [pscustomobject]@{
            Path = $_.FullName
            LastWriteTime = $_.LastWriteTimeUtc
            Bytes = Get-TreeBytes -Path $_.FullName
        }
    }
}

$candidateList = @()
Add-Candidates -Root (Join-Path $repoRoot ".cache") -Candidates ([ref]$candidateList)
Add-Candidates -Root (Join-Path $repoRoot "services\worker\.cache") -Candidates ([ref]$candidateList)
Add-Candidates -Root (Join-Path $repoRoot "artifacts\logs") -Candidates ([ref]$candidateList)
Add-Candidates -Root (Join-Path $repoRoot "artifacts\exports") -Candidates ([ref]$candidateList)

$tempWorkerRoot = Join-Path ([System.IO.Path]::GetTempPath()) "model-express-worker-datasets"
if (Test-Path -LiteralPath $tempWorkerRoot) {
    Get-ChildItem -LiteralPath $tempWorkerRoot -Force -ErrorAction SilentlyContinue | ForEach-Object {
        $candidateList += [pscustomobject]@{
            Path = $_.FullName
            LastWriteTime = $_.LastWriteTimeUtc
            Bytes = Get-TreeBytes -Path $_.FullName
        }
    }
}

$totalBytes = ($candidateList | Measure-Object -Property Bytes -Sum).Sum
$toRemove = @($candidateList | Where-Object { $_.LastWriteTime -lt $cutoff })
$remaining = @($candidateList | Where-Object { $_.LastWriteTime -ge $cutoff } | Sort-Object LastWriteTime)
$remainingBytes = $totalBytes - (($toRemove | Measure-Object -Property Bytes -Sum).Sum)

foreach ($item in $remaining) {
    if ($remainingBytes -le $maxBytes) { break }
    $toRemove += $item
    $remainingBytes -= $item.Bytes
}

$summary = [pscustomobject]@{
    apply = [bool]$Apply
    max_age_days = $MaxAgeDays
    max_bytes_gb = $MaxBytesGB
    bytes_before = [int64]$totalBytes
    bytes_after = [int64][Math]::Max(0, $remainingBytes)
    count = $toRemove.Count
    paths = @($toRemove | Select-Object -ExpandProperty Path)
}

if ($Apply) {
    foreach ($item in $toRemove) {
        Remove-Item -LiteralPath $item.Path -Recurse -Force -ErrorAction SilentlyContinue
    }
}

$summary | ConvertTo-Json -Depth 4
