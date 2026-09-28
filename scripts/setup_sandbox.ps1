# Sets up the dedicated sandbox interpreter (plan Section 11.3, layer 1) and prints the
# one-time firewall command (layer 2). Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_sandbox.ps1
# It never elevates itself; the firewall command must be run by the user in an admin PowerShell.

$ErrorActionPreference = "Stop"
$Version = "3.11.9"
$Url = "https://www.python.org/ftp/python/$Version/python-$Version-embed-amd64.zip"
$Root = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $Root "tools\sandbox-python"
$Exe = Join-Path $Dest "python.exe"

if (-not (Test-Path $Exe)) {
    New-Item -ItemType Directory -Force $Dest | Out-Null
    $Zip = Join-Path $env:TEMP "python-$Version-embed-amd64.zip"
    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing
    Write-Host ("SHA256 " + (Get-FileHash $Zip -Algorithm SHA256).Hash)
    Expand-Archive -Path $Zip -DestinationPath $Dest -Force
    Remove-Item $Zip
}

$Sig = Get-AuthenticodeSignature $Exe
Write-Host ("python.exe signature: " + $Sig.Status + " / " + $Sig.SignerCertificate.Subject)
if ($Sig.Status -ne "Valid") { throw "sandbox python.exe signature is not valid" }
& $Exe -I -c "import sys; print('sandbox interpreter', sys.version.split()[0], sys.executable)"

$Rule = Get-NetFirewallRule -DisplayName "PRISM APPS sandbox" -ErrorAction SilentlyContinue
if ($Rule) {
    Write-Host "Firewall rule 'PRISM APPS sandbox' already exists (Enabled=$($Rule.Enabled), Action=$($Rule.Action))."
} else {
    Write-Host ""
    Write-Host "Run this ONCE in an administrator PowerShell to block the sandbox interpreter's network access:"
    Write-Host ""
    Write-Host "New-NetFirewallRule -DisplayName `"PRISM APPS sandbox`" -Direction Outbound -Action Block -Program `"$Exe`""
    Write-Host ""
}
