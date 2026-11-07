<#
  Shared helpers for the node-day scripts (check_nodes.ps1, fl_round_all.ps1).
  Windows PowerShell 5.1 compatible. Never prints a password or a token.
#>

$script:DefaultTimeoutSec = 8

function Get-ArgusNodeList {
  <# Reads deploy/nodes.txt: one node per line, "<label> <ip>". # starts a comment. #>
  param([Parameter(Mandatory)] [string]$Path)
  if (-not (Test-Path $Path)) {
    throw "Node list not found: $Path. Copy deploy\nodes.example.txt to deploy\nodes.txt and fill in the IPs."
  }
  $nodes = @()
  foreach ($raw in Get-Content $Path) {
    $line = ($raw -replace '#.*$', '').Trim()
    if (-not $line) { continue }
    $parts = $line -split '\s+'
    if ($parts.Count -lt 2) { throw "Bad line in ${Path}: '$raw' (expected: <label> <ip>)" }
    $nodes += [pscustomobject]@{ Label = $parts[0]; Ip = $parts[1] }
  }
  if ($nodes.Count -eq 0) { throw "No nodes listed in $Path" }
  $dupes = $nodes | Group-Object Ip | Where-Object Count -gt 1
  if ($dupes) { throw "Same IP listed twice in ${Path}: $($dupes.Name -join ', ')" }
  return $nodes
}

function Test-ArgusPort {
  <# True when a TCP connect to host:port succeeds within the timeout. #>
  param([string]$HostName, [int]$Port, [int]$TimeoutMs = 1500)
  $client = New-Object System.Net.Sockets.TcpClient
  try {
    $task = $client.ConnectAsync($HostName, $Port)
    if (-not $task.Wait($TimeoutMs)) { return $false }
    return $client.Connected
  } catch {
    return $false
  } finally {
    $client.Dispose()
  }
}

function Get-ArgusLanAddress {
  <# This PC's IPv4 on the same /24 as $NearIp, else the first private address. #>
  param([string]$NearIp)
  $all = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } |
    Select-Object -ExpandProperty IPAddress
  if ($NearIp) {
    $prefix = ($NearIp -split '\.')[0..2] -join '.'
    $match = $all | Where-Object { $_ -like "$prefix.*" } | Select-Object -First 1
    if ($match) { return $match }
  }
  return ($all | Where-Object { $_ -like '10.*' -or $_ -like '192.168.*' } | Select-Object -First 1)
}

function ConvertFrom-ArgusSecure {
  param([Security.SecureString]$Secure)
  $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
  try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
  finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
}

function Get-ArgusAdminPassword {
  <# $env:ARGUS_ADMIN_PASSWORD if set in this shell, else a hidden prompt. #>
  if ($env:ARGUS_ADMIN_PASSWORD) { return $env:ARGUS_ADMIN_PASSWORD }
  $secure = Read-Host -AsSecureString 'Node admin password (input hidden)'
  return (ConvertFrom-ArgusSecure $secure)
}

function Get-ArgusToken {
  <#
    Logs in to one node and returns a bearer token, or $null.
    One attempt only: the node locks the account after 5 failures in 300 s.
  #>
  param([string]$Ip, [string]$Password, [string]$User = 'admin', [int]$Port = 8000)
  $body = @{ username = $User; password = $Password } | ConvertTo-Json -Compress
  try {
    $r = Invoke-RestMethod -Method Post -Uri "http://${Ip}:${Port}/api/auth/login" `
      -ContentType 'application/json' -Body $body -TimeoutSec $script:DefaultTimeoutSec
    return $r.token
  } catch {
    $code = $null
    if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
    Write-Warning "  login to $Ip failed (HTTP $code). Not retrying, to stay under the 5-tries lockout."
    return $null
  }
}

function Invoke-ArgusApi {
  <# GET or POST against a node. Returns @{ Ok; Status; Body; Error }. Never throws. #>
  param(
    [string]$Ip, [string]$Path, [string]$Method = 'Get', $Body = $null,
    [string]$Token = $null, [int]$Port = 8000, [int]$TimeoutSec = $script:DefaultTimeoutSec
  )
  $headers = @{}
  if ($Token) { $headers['Authorization'] = "Bearer $Token" }
  $params = @{
    Method = $Method; Uri = "http://${Ip}:${Port}$Path"; Headers = $headers
    TimeoutSec = $TimeoutSec; UseBasicParsing = $true
  }
  if ($null -ne $Body) {
    $params['ContentType'] = 'application/json'
    $params['Body'] = ($Body | ConvertTo-Json -Compress -Depth 5)
  }
  try {
    $resp = Invoke-WebRequest @params
    $parsed = $null
    if ($resp.Content) { $parsed = $resp.Content | ConvertFrom-Json }
    return @{ Ok = $true; Status = [int]$resp.StatusCode; Body = $parsed; Error = $null }
  } catch {
    $code = 0
    $detail = $_.Exception.Message
    if ($_.Exception.Response) {
      $code = [int]$_.Exception.Response.StatusCode
      try {
        $reader = New-Object IO.StreamReader($_.Exception.Response.GetResponseStream())
        $text = $reader.ReadToEnd()
        if ($text) { $detail = $text }
      } catch { }
    }
    return @{ Ok = $false; Status = $code; Body = $null; Error = $detail }
  }
}

Export-ModuleMember -Function Get-ArgusNodeList, Test-ArgusPort, Get-ArgusLanAddress,
  Get-ArgusAdminPassword, Get-ArgusToken, Invoke-ArgusApi
