<#
.SYNOPSIS
  Start the ARGUS central aggregation server.  >>> RUN THIS ON THE LAPTOP <<<

.DESCRIPTION
  This is the machine the cameras send their learning to. It runs a Flower
  server that averages what each node learned, keeps every version of the
  combined model, and exposes a small API and Prometheus metrics.

  Nothing but weights ever arrives here. No video, no images.

  It stays up between sessions: a camera on a schedule connects days
  later, so a server that exited after the first session would not be
  there when it mattered.

.EXAMPLE
  .\scripts\run_central_server.ps1

.EXAMPLE
  .\scripts\run_central_server.ps1 -Rounds 3 -MinClients 2
#>
param(
  [int]$Rounds = 5,
  [int]$MinClients = 1,
  [int]$FlPort = 8080,
  [int]$ApiPort = 8090,
  [string]$Token = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$env:ARGUS_FL_ADDRESS = "0.0.0.0:${FlPort}"
$env:ARGUS_MIN_CLIENTS = "$MinClients"
$env:ARGUS_ROUNDS = "$Rounds"
$env:ARGUS_SERVE_FOREVER = "true"
if ($Token) { $env:ARGUS_AUTH_TOKEN = $Token }

# The Pi needs to reach this machine, so show the address it should use.
$addresses = Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
  Select-Object -ExpandProperty IPAddress

Write-Host ""
Write-Host "ARGUS central server" -ForegroundColor Cyan
Write-Host "  rounds per session : $Rounds"
Write-Host "  clients per round  : $MinClients"
Write-Host "  checkpoints        : central_server\checkpoints"
Write-Host ""
Write-Host "On the PI, point the node at this machine:" -ForegroundColor Yellow
foreach ($ip in $addresses) {
  Write-Host "  export FL_SERVER_URL=${ip}:${FlPort}"
  Write-Host "  export FL_CENTRAL_API=http://${ip}:${ApiPort}"
}
Write-Host ""
Write-Host "Watch it here:"
Write-Host "  http://localhost:${ApiPort}/api/status"
Write-Host "  http://localhost:${ApiPort}/api/training/summary"
Write-Host "  http://localhost:${ApiPort}/metrics"
Write-Host ""

if (-not (Get-NetFirewallRule -DisplayName "ARGUS FL" -ErrorAction SilentlyContinue)) {
  Write-Host "If the Pi cannot connect, Windows Firewall is the usual reason." -ForegroundColor DarkYellow
  Write-Host "Allow it once, from an admin PowerShell:" -ForegroundColor DarkYellow
  Write-Host "  New-NetFirewallRule -DisplayName 'ARGUS FL' -Direction Inbound -Protocol TCP -LocalPort ${FlPort},${ApiPort} -Action Allow"
  Write-Host ""
}

& $python central_server\fl_aggregator.py
