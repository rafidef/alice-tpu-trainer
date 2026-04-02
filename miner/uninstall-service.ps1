$ErrorActionPreference = "Stop"
if (Get-ScheduledTask -TaskName "AliceProtocolMiner" -ErrorAction SilentlyContinue) {
  Stop-ScheduledTask -TaskName "AliceProtocolMiner" -ErrorAction SilentlyContinue
  Unregister-ScheduledTask -TaskName "AliceProtocolMiner" -Confirm:$false
}
