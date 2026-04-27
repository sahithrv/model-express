param(
  [Parameter(Mandatory = $true)]
  [string]$DatasetPath,

  [Parameter(Mandatory = $true)]
  [string]$ProjectId,

  [string]$DatasetName = "",
  [string]$Bucket = "model-express",
  [string]$EndpointUrl = "http://localhost:9000",
  [string]$AccessKey = "model_express",
  [string]$SecretKey = "model_express_password",
  [string]$Region = "us-east-1"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $DatasetPath -PathType Container)) {
  throw "DatasetPath must be a directory: $DatasetPath"
}

$resolvedDatasetPath = (Resolve-Path -LiteralPath $DatasetPath).Path

if (-not $DatasetName) {
  $DatasetName = Split-Path -Leaf $resolvedDatasetPath
}

$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) "model-express"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

$zipPath = Join-Path $tempDir "$DatasetName.zip"
Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $resolvedDatasetPath "*") -DestinationPath $zipPath -Force

if (-not (Test-Path -LiteralPath $zipPath -PathType Leaf)) {
  throw "Failed to create dataset archive: $zipPath"
}

$checksum = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
$sizeBytes = (Get-Item -LiteralPath $zipPath).Length
$key = "datasets/$ProjectId/$DatasetName.zip"

$previousAccessKey = $env:AWS_ACCESS_KEY_ID
$previousSecretKey = $env:AWS_SECRET_ACCESS_KEY
$previousRegion = $env:AWS_DEFAULT_REGION
$previousErrorActionPreference = $ErrorActionPreference

$env:AWS_ACCESS_KEY_ID = $AccessKey
$env:AWS_SECRET_ACCESS_KEY = $SecretKey
$env:AWS_DEFAULT_REGION = $Region

try {
  $ErrorActionPreference = "Continue"

  & aws --endpoint-url $EndpointUrl s3api head-bucket --bucket $Bucket 2>$null | Out-Null
  if ($LASTEXITCODE -ne 0) {
    & aws --endpoint-url $EndpointUrl s3 mb "s3://$Bucket" | Out-Null
    if ($LASTEXITCODE -ne 0) {
      throw "Failed to create bucket '$Bucket'."
    }
  }

  & aws --endpoint-url $EndpointUrl s3 cp $zipPath "s3://$Bucket/$key"
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to upload dataset archive to s3://$Bucket/$key."
  }
}
finally {
  $ErrorActionPreference = $previousErrorActionPreference
  $env:AWS_ACCESS_KEY_ID = $previousAccessKey
  $env:AWS_SECRET_ACCESS_KEY = $previousSecretKey
  $env:AWS_DEFAULT_REGION = $previousRegion
}

[PSCustomObject]@{
  name = $DatasetName
  storage_uri = "s3://$Bucket/$key"
  checksum_sha256 = $checksum
  size_bytes = $sizeBytes
}
