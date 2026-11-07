<#
.SYNOPSIS
  One command to start the Flower aggregator for node day.  >>> RUN ON THE PC <<<

.DESCRIPTION
  Checks everything that has stopped this server before, then starts it:
    1. a Python that has flwr, flask, flask_cors, prometheus_client, numpy
    2. ports 8080 (Flower) and 8090 (REST) free
    3. the Windows Firewall rule "ARGUS FL" (prints the admin command if missing)
    4. -MinClients defaults to the number of nodes in deploy\nodes.txt
  Then prints the exact FL_SERVER_URL / FL_CENTRAL_API lines for each
  node's deploy/node.env, and runs scripts\run_central_server.ps1 in this
  window (Ctrl+C stops it; checkpoints survive in central_server\checkpoints).

.EXAMPLE
  .\scripts\fl_server_up.ps1                    # min clients = nodes in nodes.txt, 1 round per session
.EXAMPLE
  .\scripts\fl_server_up.ps1 -MinClients 2 -Rounds 3
.EXAMPLE
  .\scripts\fl_server_up.ps1 -CheckOnly         # preflight only, do not start
#>
param(
  [int]$MinClients = 0,
  [int]$Rounds = 1,
  [string]$NodesFile = (Join-Path (Split-Path -Parent $PSScriptRoot) 'deploy\nodes.txt'),
  [string]$Python = '',
  [ValidateSet('robust', 'fedavg')] [string]$Aggregation = 'robust',
  [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'ArgusNodes.psm1') -Force
$repo = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent $repo
$FL_PORT = 8080
$API_PORT = 8090
$failed = $false

function Write-Check([string]$label, [bool]$ok, [string]$detail) {
  $tag = if ($ok) { '[ ok ]' } else { '[FAIL]' }
  $color = if ($ok) { 'Green' } else { 'Red' }
  Write-Host ("{0} {1}  {2}" -f $tag, $label, $detail) -ForegroundColor $color
}

# 1. Python with the server's dependencies.
$candidates = @($Python, (Join-Path $workspace '.venv-fl-server\Scripts\python.exe'),
  (Join-Path $repo '.venv\Scripts\python.exe'), (Join-Path $workspace '.venv-gate-b\Scripts\python.exe')) |
  Where-Object { $_ -and (Test-Path $_) }
$py = $null
# PS 5.1 turns a native command's stderr into a terminating error under
# "Stop", even with 2>$null, so probe each interpreter under "Continue".
$ErrorActionPreference = 'Continue'
foreach ($c in $candidates) {
  try { & $c -c 'import flwr, flask, flask_cors, prometheus_client, numpy' 2>$null; $ok = ($LASTEXITCODE -eq 0) }
  catch { $ok = $false }
  if ($ok) { $py = $c; break }
}
$ErrorActionPreference = 'Stop'
if ($py) {
  $flwrVersion = & $py -c 'import flwr; print(flwr.__version__)'
  Write-Check 'python' $true "$py (flwr $flwrVersion; the Pis should match: pip show flwr)"
} else {
  $failed = $true
  Write-Check 'python' $false 'no interpreter with flwr+flask+flask_cors. Fix:'
  Write-Host "         python -m venv $workspace\.venv-fl-server"
  Write-Host "         $workspace\.venv-fl-server\Scripts\python.exe -m pip install flwr==1.28.0 flask flask-cors prometheus-client numpy"
}

# 2. Ports free. Probe by connecting: Windows can list the gRPC server's
# dual-stack socket with no State, so a `-State Listen` filter misses it.
foreach ($p in @($FL_PORT, $API_PORT)) {
  if (Test-ArgusPort '127.0.0.1' $p) {
    $failed = $true
    $owner = @(Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique)
    $cmd = ''
    if ($owner.Count -gt 0) { $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($owner[0])").CommandLine }
    Write-Check "port $p" $false "in use (PID $($owner -join ','): $cmd). An old aggregator? Stop it first."
  } else {
    Write-Check "port $p" $true 'free'
  }
}

# 3. Firewall.
$rule = Get-NetFirewallRule -DisplayName 'ARGUS FL' -ErrorAction SilentlyContinue |
  Where-Object { $_.Enabled -eq 'True' }
if ($rule) {
  Write-Check 'firewall' $true "rule 'ARGUS FL' present"
} else {
  $failed = $true
  Write-Check 'firewall' $false "no 'ARGUS FL' rule: the Pis cannot connect. Run once in an ADMIN PowerShell:"
  Write-Host "         New-NetFirewallRule -DisplayName 'ARGUS FL' -Direction Inbound -Protocol TCP -LocalPort $FL_PORT,$API_PORT -Action Allow"
}

# 4. Nodes and min clients.
$nodes = @()
try { $nodes = @(Get-ArgusNodeList -Path $NodesFile) }
catch { Write-Host "       $($_.Exception.Message)" -ForegroundColor DarkYellow }
if ($MinClients -le 0) { $MinClients = [Math]::Max(1, $nodes.Count) }
$near = $null
if ($nodes.Count -gt 0) { $near = $nodes[0].Ip }
$lan = Get-ArgusLanAddress -NearIp $near
Write-Check 'nodes' ($nodes.Count -gt 0) ("{0} listed; the server waits for {1} client(s) per round" -f $nodes.Count, $MinClients)

Write-Host ''
Write-Host "Each node's deploy/node.env must say:" -ForegroundColor Yellow
Write-Host "  FL_SERVER_URL=${lan}:$FL_PORT"
Write-Host "  FL_CENTRAL_API=http://${lan}:$API_PORT"
Write-Host ''

if ($failed) {
  Write-Host 'Preflight FAILED. Fix the red lines above, then run this again.' -ForegroundColor Red
  exit 1
}
if ($CheckOnly) { Write-Host 'Preflight passed (-CheckOnly: not starting).' -ForegroundColor Green; exit 0 }

& (Join-Path $PSScriptRoot 'run_central_server.ps1') -Rounds $Rounds -MinClients $MinClients `
  -FlPort $FL_PORT -ApiPort $API_PORT -Aggregation $Aggregation -Python $py
