$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$env:QUANTONIUM_LOCAL_BACKEND = "rftmw"
$env:QUANTONIUM_LOCAL_ONLY = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"
$env:QUANTONIUM_RFTMW_DISCOVERY_ROOTS = ".\ai\runtime\chatbox"

& ".\qshellchatbox.ps1" @args
