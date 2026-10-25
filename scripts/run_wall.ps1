<#
.SYNOPSIS
  Every node and the aggregator on one screen.  >>> RUN THIS ON THE LAPTOP <<<

.DESCRIPTION
  Serves scripts\wall\ on http://localhost:3001 and opens it. The page pulls
  live video, status and recent events straight from each node, and the
  round and node registry from the aggregator. Nothing is installed; it
  stops when you close this window.

  localhost:3001 is already in the nodes' default CORS_ORIGINS, so the page
  can read their status. Video needs no CORS.

.EXAMPLE
  .\scripts\run_wall.ps1

.EXAMPLE
  .\scripts\run_wall.ps1 -Nodes 10.0.0.140,10.0.0.141 -Aggregator 10.0.0.239:8090
#>
param(
  [string[]]$Nodes = @("10.0.0.140", "argus-b.local", "argus-c.local"),
  [string]$Aggregator = "10.0.0.239:8090",
  [int]$Port = 3001
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$url = "http://localhost:${Port}/?nodes=$($Nodes -join ',')&agg=$Aggregator"
Write-Host "nodes      : $($Nodes -join ', ')"
Write-Host "aggregator : $Aggregator"
Write-Host "wall       : $url"
Write-Host "Ctrl+C to stop."

Start-Process $url
& $python -m http.server $Port --bind 127.0.0.1 --directory (Join-Path $PSScriptRoot "wall")
