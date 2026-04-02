$ErrorActionPreference = "Stop"
$task = Get-ScheduledTask -TaskName "AliceProtocolScorer"
$info = Get-ScheduledTaskInfo -TaskName "AliceProtocolScorer"
$task | Format-List TaskName,State
$info | Format-List LastRunTime,LastTaskResult,NextRunTime
