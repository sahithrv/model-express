param(
  [Parameter(Mandatory = $true)]
  [string]$DatasetPath,

  [Parameter(Mandatory = $true)]
  [string]$ProjectId,

  [string]$DatasetName = "",
  [string]$Bucket = "model-express",
  [string]$EndpointUrl = "http://localhost:9000"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $DatasetPath -PathType Container)) {
  throw "DatasetPath must be a directory: $DatasetPath"
}

if (-not $DatasetName) {
  $DatasetName = Split-Path -Leaf (Resolve-Path -LiteralPath $DatasetPath)
}

$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) "model-express"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

$zipPath = Join-Path $tempDir "$DatasetName.zip"
Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue
Compress-Archive -LiteralPath (Join-Path $DatasetPath "*") -DestinationPath $zipPath -Force

$checksum = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
$sizeBytes = (Get-Item -LiteralPath $zipPath).Length
$key = "datasets/$ProjectId/$DatasetName.zip"

aws --endpoint-url $EndpointUrl s3 mb "s3://$Bucket" 2>$null | Out-Null
aws --endpoint-url $EndpointUrl s3 cp $zipPath "s3://$Bucket/$key"

[PSCustomObject]@{
  name = $DatasetName
  storage_uri = "s3://$Bucket/$key"
  checksum_sha256 = $checksum
  size_bytes = $sizeBytes
}
