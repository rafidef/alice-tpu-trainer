$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$LogDir = Join-Path $HOME ".alice\logs"
$ConfigPath = Join-Path $HOME ".alice\miner-service.ps1"
New-Item -ItemType Directory -Force $LogDir | Out-Null

$AliceMinerArgs = @()
if (Test-Path $ConfigPath) {
  . $ConfigPath
}

$MergedArgs = @()
if ($AliceMinerArgs) {
  $MergedArgs += $AliceMinerArgs
}
if ($args) {
  $MergedArgs += $args
}

$LogPath = Join-Path $LogDir "miner-service.log"
& .\miner\run_miner.bat @MergedArgs *>> $LogPath
exit $LASTEXITCODE
