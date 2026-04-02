$ErrorActionPreference = "Stop"
if (Get-ScheduledTask -TaskName "AliceProtocolScorer" -ErrorAction SilentlyContinue) {
  Stop-ScheduledTask -TaskName "AliceProtocolScorer" -ErrorAction SilentlyContinue
  Unregister-ScheduledTask -TaskName "AliceProtocolScorer" -Confirm:$false
}
