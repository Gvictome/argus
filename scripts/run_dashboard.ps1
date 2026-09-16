<#
.SYNOPSIS
  Start the ARGUS dashboard.  >>> RUN THIS ON THE LAPTOP <<<

.DESCRIPTION
  Clones adam9798/argus-dashboard if it is missing, installs dependencies
  the first time, points it at the node, and starts it.

  The dashboard runs on the laptop; the data comes from the Pi. Your
  browser must be able to reach the node itself -- if it cannot, use the
  node's own page at http://<node>:8000/dashboard instead, which needs
  nothing installed.

.EXAMPLE
  .\scripts\run_dashboard.ps1 -Node 10.0.0.140

.EXAMPLE
  .\scripts\run_dashboard.ps1 -Node 10.0.0.140 -Path $HOME\Documents\argus-dashboard
#>
param(
  [string]$Node = "10.0.0.140",
  [int]$Port = 8000,
  [string]$Path = "$HOME\Documents\argus-dashboard"
)

$ErrorActionPreference = "Stop"
$api = "http://${Node}:${Port}"

Write-Host "node API : $api"
Write-Host "dashboard: $Path"

try {
  Invoke-RestMethod -Uri "$api/api/status" -TimeoutSec 5 | Out-Null
  Write-Host "node is reachable" -ForegroundColor Green
} catch {
  Write-Warning "Cannot reach $api from this laptop."
  Write-Warning "The dashboard will load but show SYSTEM OFFLINE."
  Write-Warning "Use http://${Node}:${Port}/dashboard on a machine that can reach the node."
}

if (-not (Test-Path $Path)) {
  Write-Host "cloning the dashboard..."
  git clone https://github.com/adam9798/argus-dashboard $Path
}

Set-Location $Path
if (-not (Test-Path "$Path\node_modules")) {
  Write-Host "installing dependencies (first run only, a few minutes)..."
  npm install
}

# NEXT_PUBLIC_* is read when the dev server starts, so this must be written
# before npm run dev, and the server restarted after any change.
Set-Content -Path "$Path\.env.local" -Encoding utf8 -Value @(
  "NEXT_PUBLIC_API_BASE_URL=$api",
  "AUTH_COOKIE_SECURE=false"
)
Write-Host "wrote .env.local -> $api"

Write-Host ""
Write-Host "Next.js takes port 3001 when 3000 is busy. Whichever it prints must"
Write-Host "appear in CORS_ORIGINS on the node, or every request is blocked."
Write-Host "Log in as admin. Set the password on the PI with:"
Write-Host "  python scripts/set_password.py --user admin"
Write-Host ""

npm run dev
