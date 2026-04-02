$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$LogDir = Join-Path $HOME ".alice\logs"
$ConfigPath = Join-Path $HOME ".alice\scorer-service.ps1"
New-Item -ItemType Directory -Force $LogDir | Out-Null

$AliceScorerArgs = @()
if (Test-Path $ConfigPath) {
  . $ConfigPath
}

$MergedArgs = @()
if ($AliceScorerArgs) {
  $MergedArgs += $AliceScorerArgs
}
if ($args) {
  $MergedArgs += $args
}

$LogPath = Join-Path $LogDir "scorer-service.log"
& .\scorer\run_scorer.ps1 @MergedArgs *>> $LogPath
exit $LASTEXITCODE
