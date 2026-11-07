<#
.SYNOPSIS
  Health of every ARGUS node and the aggregator, one table.  >>> RUN ON THE PC <<<

.DESCRIPTION
  For each node in deploy\nodes.txt: TCP checks on SSH (22) and the API
  (8000), then the public /api/status (node name, accelerator, fps, model
  version, labelled samples). It also proves auth is ON without a
  password: an anonymous call to /api/detection/status must get 401.

  -Login asks for the admin password once (hidden) and adds the detection
  backend and any camera error from /api/detection/status.

  Nothing here changes a node. No password or token is ever printed.

.EXAMPLE
  .\scripts\check_nodes.ps1
.EXAMPLE
  .\scripts\check_nodes.ps1 -Login
.EXAMPLE
  .\scripts\check_nodes.ps1 -Watch 10     # repeat every 10 s, Ctrl+C to stop
#>
param(
  [string]$NodesFile = (Join-Path (Split-Path -Parent $PSScriptRoot) 'deploy\nodes.txt'),
  [string]$Aggregator = 'localhost:8090',
  [switch]$Login,
  [int]$Watch = 0
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'ArgusNodes.psm1') -Force

$MIN_SAMPLES = 200
$nodes = Get-ArgusNodeList -Path $NodesFile
$tokens = @{}
if ($Login) {
  $pw = Get-ArgusAdminPassword
  foreach ($n in $nodes) {
    if (Test-ArgusPort $n.Ip 8000) { $tokens[$n.Ip] = Get-ArgusToken -Ip $n.Ip -Password $pw }
  }
  $pw = $null
}

function Get-NodeRow($n) {
  $row = [ordered]@{
    Node = $n.Label; Ip = $n.Ip; SSH = '-'; API = '-'; Name = ''; Accel = ''; Fps = '';
    Model = ''; Labelled = ''; Auth = ''; Backend = ''; Problem = ''
  }
  $row.SSH = if (Test-ArgusPort $n.Ip 22) { 'ok' } else { 'DOWN' }
  if (-not (Test-ArgusPort $n.Ip 8000)) {
    $row.API = 'DOWN'
    if ($row.SSH -eq 'ok') { $row.Problem = 'Pi up, service not answering: journalctl -u argus-node -n 50' }
    else { $row.Problem = 'Pi unreachable: power, network or wrong IP' }
    return [pscustomobject]$row
  }
  $row.API = 'ok'
  $s = Invoke-ArgusApi -Ip $n.Ip -Path '/api/status'
  if ($s.Ok) {
    $row.Name = $s.Body.node
    $row.Accel = $s.Body.accelerator
    $row.Fps = $s.Body.fps
    $row.Model = $s.Body.model_version
    if ($s.Body.samples) { $row.Labelled = $s.Body.samples.labelled }
  } else {
    $row.Problem = "/api/status HTTP $($s.Status)"
  }
  # Auth probe: an anonymous request to a protected route must be refused.
  $anon = Invoke-ArgusApi -Ip $n.Ip -Path '/api/detection/status'
  if ($anon.Status -eq 401) { $row.Auth = 'ON' }
  elseif ($anon.Ok) { $row.Auth = 'OFF!'; $row.Problem = 'AUTH_REQUIRED is off: fix deploy/node.env' }
  else { $row.Auth = "?($($anon.Status))" }

  $tok = $tokens[$n.Ip]
  if ($tok) {
    $d = Invoke-ArgusApi -Ip $n.Ip -Path '/api/detection/status' -Token $tok
    if ($d.Ok) {
      $row.Backend = $d.Body.backend
      if ($d.Body.worker -and $d.Body.worker.camera_error) { $row.Problem = "CAMERA: $($d.Body.worker.camera_error)" }
    }
  } elseif ($Login) {
    $row.Backend = 'login failed'
  }
  # Compare as a string first: in PowerShell 0 -ne '' is false.
  if ("$($row.Labelled)" -ne '' -and [int]$row.Labelled -lt $MIN_SAMPLES -and -not $row.Problem) {
    $row.Problem = "only $($row.Labelled) labelled samples; a round needs $MIN_SAMPLES (fl_round_all.ps1 -Bootstrap)"
  }
  return [pscustomobject]$row
}

function Show-Aggregator {
  $aggHost, $aggPort = $Aggregator -split ':'
  if (-not (Test-ArgusPort $aggHost ([int]$aggPort))) {
    Write-Host "  aggregator $Aggregator not running (start it: .\scripts\fl_server_up.ps1)" -ForegroundColor DarkYellow
    return
  }
  $a = Invoke-ArgusApi -Ip $aggHost -Port ([int]$aggPort) -Path '/api/status'
  if (-not $a.Ok) {
    Write-Host "  aggregator $Aggregator answers HTTP $($a.Status) on /api/status" -ForegroundColor Yellow
    return
  }
  $b = $a.Body
  Write-Host ("  aggregator {0} UP: round {1}/{2}, connected {3}, registered nodes {4}, global model v{5}, accuracy {6:N3}" -f `
    $Aggregator, $b.current_round, $b.total_rounds, $b.connected_clients, $b.registered_nodes,
    $b.model_version, [double]$b.global_accuracy) -ForegroundColor Green
}

function Show-Once {
  Write-Host ''
  Write-Host ("ARGUS nodes  {0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -ForegroundColor Cyan
  $rows = @(foreach ($n in $nodes) { Get-NodeRow $n })
  $rows | Format-Table Node, Ip, SSH, API, Name, Accel, Fps, Model, Labelled, Auth, Backend -AutoSize |
    Out-String | Write-Host

  $dupes = $rows | Where-Object { $_.Name } | Group-Object Name | Where-Object Count -gt 1
  foreach ($d in $dupes) {
    Write-Host "  SAME NODE_NAME on several nodes: '$($d.Name)'. Each needs its own." -ForegroundColor Red
  }
  foreach ($r in $rows | Where-Object { $_.Problem }) {
    Write-Host ("  {0} ({1}): {2}" -f $r.Node, $r.Ip, $r.Problem) -ForegroundColor Yellow
  }
  Show-Aggregator
  $bad = @($rows | Where-Object { $_.API -ne 'ok' -or $_.Auth -ne 'ON' }).Count
  if ($bad -eq 0) { Write-Host "  ALL $($rows.Count) NODES UP, AUTH ON" -ForegroundColor Green }
  return $bad
}

if ($Watch -gt 0) {
  while ($true) { [void](Show-Once); Start-Sleep -Seconds $Watch }
}
$bad = Show-Once
exit ([int]($bad -gt 0))
