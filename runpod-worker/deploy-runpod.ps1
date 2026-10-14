# Points the existing RunPod serverless endpoint at a new worker image.
# Creates a fresh template (keeps HF_TOKEN from the current one) and switches the endpoint to it,
# so the Cloudflare runpod-proxy keeps working without any change.
#
# Usage:
#   $env:RUNPOD_API_KEY = "<key>"      # or you will be prompted
#   .\runpod-worker\deploy-runpod.ps1 -Image yourdockeruser/audvid-whisperx:v20261013-1300
param(
    [Parameter(Mandatory = $true)][string] $Image,
    [string] $EndpointId = "woiyqhh4npzzmr",
    [int] $ContainerDiskGb = 40,
    [int] $ExecutionTimeoutMs = 600000,
    [int] $IdleTimeoutS = 5,
    [string[]] $AllowedCudaVersions = @("12.8", "12.9"),
    [string] $RegistryAuthId = ""
)
$ErrorActionPreference = "Stop"

$key = $env:RUNPOD_API_KEY
if (-not $key) {
    $secure = Read-Host "RunPod API key" -AsSecureString
    $key = [Runtime.InteropServices.Marshal]::PtrToStringAuth([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
}
$headers = @{ Authorization = "Bearer $key"; "Content-Type" = "application/json" }

function Invoke-Gql([string] $Query) {
    $body = @{ query = $Query } | ConvertTo-Json -Compress -Depth 10
    $result = Invoke-RestMethod -Method Post "https://api.runpod.io/graphql" -Headers $headers -Body ([Text.Encoding]::UTF8.GetBytes($body))
    if ($result.errors) { throw ("GraphQL: " + (($result.errors | ForEach-Object { $_.message }) -join " | ")) }
    return $result.data
}
function Quote([string] $Value) { return ($Value | ConvertTo-Json -Compress) }

$read = Invoke-Gql 'query { myself { endpoints { id name template { id name env { key value } } } } }'
$endpoint = $read.myself.endpoints | Where-Object { $_.id -eq $EndpointId }
if (-not $endpoint) { throw "Endpoint $EndpointId not found for this API key." }

$envVars = @($endpoint.template.env | Where-Object { $_.key -in @("HF_TOKEN") })
if (-not $envVars) { Write-Warning "Current template has no HF_TOKEN; diarization will need huggingface_access_token in the request." }
$envVars += [pscustomobject]@{ key = "BATCH_SIZE"; value = "16" }
$envLiteral = ($envVars | ForEach-Object { "{ key: $(Quote $_.key), value: $(Quote ([string]$_.value)) }" }) -join ", "

$templateName = "audvid-whisperx-" + ($Image -replace '^.*:', '') + "-" + (Get-Date -Format "HHmmss")
$fields = @(
    "name: $(Quote $templateName)",
    "imageName: $(Quote $Image)",
    "isServerless: true",
    "containerDiskInGb: $ContainerDiskGb",
    "volumeInGb: 0",
    "dockerArgs: `"`"",
    "env: [$envLiteral]"
)
if ($RegistryAuthId) { $fields += "containerRegistryAuthId: $(Quote $RegistryAuthId)" }
$saved = (Invoke-Gql "mutation { saveTemplate(input: { $($fields -join ', ') }) { id name imageName env { key } } }").saveTemplate
Write-Host "TEMPLATE id=$($saved.id) name=$($saved.name) image=$($saved.imageName) env=$((($saved.env | ForEach-Object { $_.key }) -join ','))"

$patch = [ordered]@{
    templateId          = $saved.id
    executionTimeoutMs  = $ExecutionTimeoutMs
    idleTimeout         = $IdleTimeoutS
    workersMin          = 0
    flashboot           = $true
}
if ($AllowedCudaVersions.Count -gt 0) { $patch.allowedCudaVersions = $AllowedCudaVersions }
$patchJson = $patch | ConvertTo-Json -Compress
$updated = Invoke-RestMethod -Method Patch "https://rest.runpod.io/v1/endpoints/$EndpointId" -Headers $headers -Body ([Text.Encoding]::UTF8.GetBytes($patchJson))
Write-Host "ENDPOINT id=$($updated.id) templateId=$($updated.templateId) timeoutMs=$($updated.executionTimeoutMs) cuda=$((@($updated.allowedCudaVersions) -join ','))"
Write-Host "Old template $($endpoint.template.id) ($($endpoint.template.name)) was left in place for rollback."
