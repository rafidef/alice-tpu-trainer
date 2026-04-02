$ErrorActionPreference = "Stop"
$task = Get-ScheduledTask -TaskName "AliceProtocolMiner"
$info = Get-ScheduledTaskInfo -TaskName "AliceProtocolMiner"
$task | Format-List TaskName,State
$info | Format-List LastRunTime,LastTaskResult,NextRunTime
