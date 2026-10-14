# Builds the worker image locally (Docker Desktop with the WSL2 backend) and pushes it to Docker Hub.
# Usage:
#   docker login
#   .\runpod-worker\build-and-push.ps1 -Repository yourdockeruser/audvid-whisperx
#   .\runpod-worker\build-and-push.ps1 -Repository yourdockeruser/audvid-whisperx -Deploy
param(
    [Parameter(Mandatory = $true)][string] $Repository,
    [string] $Tag = ("v" + (Get-Date -Format "yyyyMMdd-HHmm")),
    [string] $WhisperModel = "large-v3",
    [string] $AlignLanguages = "en ar",
    [switch] $Deploy
)
$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker not found. Install Docker Desktop (WSL2 backend) or use the GitHub Actions workflow instead."
}
docker info --format "{{.ServerVersion}}" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Docker daemon is not running." }

$image = "${Repository}:${Tag}"
$context = $PSScriptRoot

Write-Host "Building $image (linux/amd64)..."
docker buildx build `
    --platform linux/amd64 `
    --build-arg "WHISPER_MODEL=$WhisperModel" `
    --build-arg "ALIGN_LANGUAGES=$AlignLanguages" `
    --tag $image `
    --tag "${Repository}:latest" `
    --push `
    $context
if ($LASTEXITCODE -ne 0) { throw "docker build/push failed." }

$digest = docker buildx imagetools inspect $image --format "{{json .Manifest.Digest}}" | ConvertFrom-Json
Write-Host "PUSHED $image"
Write-Host "DIGEST $digest"

if ($Deploy) {
    & (Join-Path $PSScriptRoot "deploy-runpod.ps1") -Image $image
}
