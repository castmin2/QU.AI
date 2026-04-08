$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$py = if (Test-Path ".\\.venv\\Scripts\\python.exe") { ".\\.venv\\Scripts\\python.exe" } else { "python" }

New-Item -ItemType Directory -Force -Path ".\\logs" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = ".\\logs\\chatbox_${ts}_stdout.log"
$stderr = ".\\logs\\chatbox_${ts}_stderr.log"

$env:PYTHONUTF8 = "1"

& $py -X faulthandler "src\\apps\\qshll_chatbox.py" @args 1> $stdout 2> $stderr
$code = $LASTEXITCODE
if ($code -ne 0) {
  Write-Host "Chatbox exited with code $code."
  Write-Host "See logs: $stderr"
  pause
}

