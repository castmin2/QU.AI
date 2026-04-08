$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$py = if (Test-Path ".\\.venv\\Scripts\\python.exe") { ".\\.venv\\Scripts\\python.exe" } else { "python" }

$manifestPath = ".\\ai\\runtime\\chatbox\\manifest.json"
if (Test-Path $manifestPath) {
  try {
    $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
    $bundleId = [string]$manifest.default_bundle
    $bundle = $null
    foreach ($entry in $manifest.bundles) {
      if ([string]$entry.bundle_id -eq $bundleId) {
        $bundle = $entry
        break
      }
    }

    if ($bundle -and $bundle.pack_download) {
      $packRel = [string]$bundle.pack_path
      $packPath = Join-Path $repoRoot ("ai\\runtime\\chatbox\\" + $packRel.Replace('/', '\'))
      $expectedSize = 0
      try { $expectedSize = [int64]$bundle.pack_download.size_bytes } catch { $expectedSize = 0 }
      $needsFetch = $true
      if (Test-Path $packPath) {
        $actualSize = (Get-Item $packPath).Length
        if ($expectedSize -le 0 -or $actualSize -eq $expectedSize) {
          $needsFetch = $false
        }
      }
      if ($needsFetch) {
        Write-Host "Fetching default RFTMW pack bundle: $bundleId"
        & $py "src\\apps\\fetch_chatbox_pack.py" --bundle-id $bundleId --force
        if ($LASTEXITCODE -ne 0) {
          throw "fetch_chatbox_pack.py failed with exit code $LASTEXITCODE"
        }
      }
    }
  } catch {
    Write-Host "Warning: could not prefetch the default RFTMW pack."
    Write-Host $_
  }
}

$env:QUANTONIUM_LOCAL_BACKEND = "rftmw"
$env:QUANTONIUM_LOCAL_ONLY = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"
$env:QUANTONIUM_RFTMW_DISCOVERY_ROOTS = ".\ai\runtime\chatbox"

& ".\qshellchatbox.ps1" @args
