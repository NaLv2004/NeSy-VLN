param([int]$Port = 8893)
$ErrorActionPreference = 'Stop'
$dashRoot = $PSScriptRoot
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $running = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener[0].OwningProcess)"
    if ($running.CommandLine -like '*program_ppo_v35\dashboard\server.py*') {
        Write-Output "PPO dashboard is already running: http://127.0.0.1:$Port/"
        return
    }
    throw "Port $Port is occupied by another application."
}
$temporaryCredential = -not $env:STRIPS_SSH_PASSWORD
if ($temporaryCredential) {
    $secret = Read-Host 'SSH password for user@10.129.2.34' -AsSecureString
    $env:STRIPS_SSH_PASSWORD = [System.Net.NetworkCredential]::new('', $secret).Password
}
if (-not $env:STRIPS_SSH_LOCAL_IP) {
    $address = Get-NetIPAddress -InterfaceIndex 10 -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.AddressState -eq 'Preferred' -and $_.IPAddress -notlike '169.254.*' } |
        Select-Object -First 1
    if ($address) { $env:STRIPS_SSH_LOCAL_IP = $address.IPAddress }
}
try {
    $python = 'D:\EmbodiedVLA\lite3_navigation\.venv\Scripts\python.exe'
    $child = Start-Process -FilePath $python -ArgumentList @('-u', "$dashRoot\server.py", '--port', "$Port") `
        -WorkingDirectory $dashRoot -WindowStyle Hidden `
        -RedirectStandardOutput "$dashRoot\server.log" -RedirectStandardError "$dashRoot\server-errors.log" -PassThru
    Write-Output "Started dashboard process $($child.Id): http://127.0.0.1:$Port/"
} finally {
    if ($temporaryCredential) { Remove-Item Env:STRIPS_SSH_PASSWORD }
}
