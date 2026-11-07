<#
.SYNOPSIS
  Run one federated round across every node in deploy\nodes.txt.  >>> RUN ON THE PC <<<

.DESCRIPTION
  The aggregator must already be running (.\scripts\fl_server_up.ps1 in
  another window, with -MinClients equal to the node count).

    1. logs in to each node once (password asked once, hidden; never printed)
    2. checks each node has the 200 labelled samples a round needs
       (-Bootstrap seeds synthetic samples where short; say so on the poster)
    3. tells every node to join the round (POST /api/federated/round)
    4. waits, showing each node's progress
    5. prints each node's gate decision and the aggregator's round metrics
    6. saves a JSON report in node_day_reports\ (no secrets in it)

  Works with AUTH_REQUIRED=true, unlike scripts/fl_trial.sh.

.EXAMPLE
  .\scripts\fl_round_all.ps1
.EXAMPLE
  .\scripts\fl_round_all.ps1 -Bootstrap        # seed synthetic samples on nodes under 200
#>
param(
  [string]$NodesFile = (Join-Path (Split-Path -Parent $PSScriptRoot) 'deploy\nodes.txt'),
  [string]$Server = '',
  [string]$Aggregator = 'localhost:8090',
  [int]$Epochs = 4,
  [switch]$Bootstrap,
  [int]$BootstrapN = 1500,
  [int]$MinSamples = 200,
  [int]$TimeoutMin = 20,
  [string]$ReportDir = (Join-Path (Split-Path -Parent $PSScriptRoot) 'node_day_reports')
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'ArgusNodes.psm1') -Force
$POLL_SEC = 5
$GATE_WAIT_SEC = 90
$BOOTSTRAP_TIMEOUT_SEC = 900
$SITES = @('driveway', 'backdoor', 'yard')   # a different synthetic site per node

$nodes = @(Get-ArgusNodeList -Path $NodesFile)
if (-not $Server) {
  $lan = Get-ArgusLanAddress -NearIp $nodes[0].Ip
  if (-not $lan) { Write-Host 'Could not find this PC''s LAN address. Pass -Server <pc-ip>:8080.' -ForegroundColor Red; exit 1 }
  $Server = "${lan}:8080"
}
$aggHost, $aggPort = $Aggregator -split ':'

function Stop-WithError([string]$msg) { Write-Host $msg -ForegroundColor Red; exit 1 }

# 1. Aggregator up?
if (-not (Test-ArgusPort $aggHost ([int]$aggPort))) {
  Stop-WithError "Aggregator not running on $Aggregator. Start it first: .\scripts\fl_server_up.ps1"
}
Write-Host "Aggregator up. Nodes will join $Server ($($nodes.Count) node(s))." -ForegroundColor Cyan

# 2. Log in everywhere before starting anything: a missing node makes the round wait forever.
$pw = Get-ArgusAdminPassword
$tokens = @{}
foreach ($n in $nodes) {
  $t = Get-ArgusToken -Ip $n.Ip -Password $pw
  if (-not $t) {
    Stop-WithError "Cannot log in to $($n.Label) ($($n.Ip)). Fix it, or remove it from nodes.txt AND restart the server with one fewer -MinClients."
  }
  $tokens[$n.Ip] = $t
}
$pw = $null

# 3. Enough labelled samples on each node?
$i = 0
foreach ($n in $nodes) {
  $h = Invoke-ArgusApi -Ip $n.Ip -Path '/api/federated/head' -Token $tokens[$n.Ip]
  if (-not $h.Ok) { Stop-WithError "$($n.Label): /api/federated/head HTTP $($h.Status): $($h.Error)" }
  $labelled = [int]$h.Body.samples.labelled
  if ($labelled -ge $MinSamples) {
    Write-Host "  $($n.Label): $labelled labelled samples, ok"
  } elseif (-not $Bootstrap) {
    Stop-WithError "$($n.Label) has $labelled labelled samples, needs $MinSamples. Rerun with -Bootstrap (synthetic, disclose it) or label more events."
  } else {
    $site = $SITES[$i % $SITES.Count]
    Write-Host "  $($n.Label): $labelled labelled. Seeding $BootstrapN SYNTHETIC samples (site '$site'), minutes on a Pi..." -ForegroundColor DarkYellow
    Write-Host '    (if this times out the node keeps seeding: wait until its labelled count stops growing before re-running)' -ForegroundColor DarkGray
    $b = Invoke-ArgusApi -Ip $n.Ip -Path '/api/federated/samples/bootstrap' -Method Post `
      -Body @{ site = $site; n = $BootstrapN } -Token $tokens[$n.Ip] -TimeoutSec $BOOTSTRAP_TIMEOUT_SEC
    if (-not $b.Ok) { Stop-WithError "$($n.Label): bootstrap failed, HTTP $($b.Status): $($b.Error)" }
  }
  $i++
}

# 4. Start the round on every node.
$started = @{}
foreach ($n in $nodes) {
  $r = Invoke-ArgusApi -Ip $n.Ip -Path '/api/federated/round' -Method Post `
    -Body @{ server = $Server; epochs = $Epochs } -Token $tokens[$n.Ip]
  if ($r.Ok) { Write-Host "  $($n.Label): joined, training on $($r.Body.labelled_samples) samples" -ForegroundColor Green }
  elseif ($r.Status -eq 409 -and "$($r.Error)" -match 'already running') {
    Write-Host "  $($n.Label): a round is already running there (watching it instead)" -ForegroundColor DarkYellow
  }
  else { Stop-WithError "$($n.Label): could not start the round, HTTP $($r.Status): $($r.Error)" }
  $started[$n.Ip] = $r.Body
}

# 5. Wait for every node to finish.
$deadline = (Get-Date).AddMinutes($TimeoutMin)
$final = @{}
$exitedAt = @{}
while ($final.Count -lt $nodes.Count -and (Get-Date) -lt $deadline) {
  Start-Sleep -Seconds $POLL_SEC
  $line = @()
  foreach ($n in $nodes) {
    if ($final.ContainsKey($n.Ip)) { $line += "$($n.Label)=done"; continue }
    $s = Invoke-ArgusApi -Ip $n.Ip -Path '/api/federated/round' -Token $tokens[$n.Ip]
    if (-not $s.Ok) { $line += "$($n.Label)=HTTP$($s.Status)"; continue }
    # The client process exits first; the validation gate posts `result` a
    # moment later. Wait for it, but not forever if the gate never answers.
    $exited = (-not $s.Body.running) -and ($null -ne $s.Body.exit_code)
    if ($exited -and -not $exitedAt.ContainsKey($n.Ip)) { $exitedAt[$n.Ip] = Get-Date }
    $gateLate = $exited -and ((Get-Date) - $exitedAt[$n.Ip]).TotalSeconds -gt $GATE_WAIT_SEC
    if ((-not $s.Body.running -and $null -ne $s.Body.result) -or $gateLate) {
      $final[$n.Ip] = $s.Body; $line += "$($n.Label)=done"
    } elseif ($exited) { $line += "$($n.Label)=gate" }
    else { $line += "$($n.Label)=running" }
  }
  Write-Host ("  {0}  {1}" -f (Get-Date -Format 'HH:mm:ss'), ($line -join '  '))
}

# 6. Results.
Write-Host ''
Write-Host '== Result per node ==' -ForegroundColor Cyan
$report = @()
foreach ($n in $nodes) {
  $f = $final[$n.Ip]
  $st = Invoke-ArgusApi -Ip $n.Ip -Path '/api/status'
  $ver = if ($st.Ok) { $st.Body.model_version } else { '?' }
  if (-not $f) {
    Write-Host "  $($n.Label): DID NOT FINISH in $TimeoutMin min (aggregator still waiting for a client?)" -ForegroundColor Red
  } else {
    $res = $f.result
    $ok = [bool]($res -and $res.accepted)
    $color = if ($ok) { 'Green' } else { 'Yellow' }
    Write-Host ("  {0}: accepted={1}  model={2}  reason={3}" -f $n.Label, $ok, $ver, $res.reason) -ForegroundColor $color
    if ($res -and $res.current -and $res.candidate) {
      Write-Host ("    balanced accuracy on this node's held-out 20%: live {0} -> candidate {1} (n={2})" -f `
        $res.current.balanced_accuracy, $res.candidate.balanced_accuracy, $res.val_samples)
    }
    if (-not $res -or $res.exit_code) {
      Write-Host '    last log lines:' -ForegroundColor DarkGray
      @($f.log) | Select-Object -Last 15 | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
    }
  }
  $report += [ordered]@{ label = $n.Label; ip = $n.Ip; started = $started[$n.Ip]; final = $f; model_version = $ver }
}
Write-Host '  (a rejection is the validation gate working: the averaged model scored worse on that node)' -ForegroundColor DarkGray

Write-Host ''
Write-Host '== Aggregator ==' -ForegroundColor Cyan
$sum = Invoke-ArgusApi -Ip $aggHost -Port ([int]$aggPort) -Path '/api/training/summary'
$agg = Invoke-ArgusApi -Ip $aggHost -Port ([int]$aggPort) -Path '/api/status'
if ($sum.Ok) {
  foreach ($r in @($sum.Body.rounds)) {
    Write-Host ("  round {0}: clients {1}, loss {2}, accuracy {3}, {4:N1} s" -f $r.round, $r.clients, $r.loss, $r.accuracy, [double]$r.duration)
  }
  Write-Host "  global model version: $($agg.Body.model_version), registered nodes: $($agg.Body.registered_nodes)"
} else {
  Write-Host "  /api/training/summary HTTP $($sum.Status)" -ForegroundColor Yellow
}

New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
$path = Join-Path $ReportDir ("round-{0}.json" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
[ordered]@{
  when = (Get-Date -Format 's'); server = $Server; bootstrap_used = [bool]$Bootstrap
  nodes = $report; aggregator_summary = $sum.Body; aggregator_status = $agg.Body
} | ConvertTo-Json -Depth 8 | Out-File -FilePath $path -Encoding utf8
Write-Host ''
Write-Host "Report saved: $path"
$done = @($final.Values | Where-Object { $null -ne $_.result }).Count
exit ([int]($done -lt $nodes.Count))
