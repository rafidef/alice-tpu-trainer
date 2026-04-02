$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  throw "Missing .venv. Run .\scorer\bootstrap.ps1 first."
}

$VenvPython = ".\.venv\Scripts\python.exe"
$PsUrl = if ($env:ALICE_PS_URL) { $env:ALICE_PS_URL } else { "https://ps.aliceprotocol.org" }
$ModelPath = if ($env:ALICE_MODEL_PATH) { $env:ALICE_MODEL_PATH } else { Join-Path $Root "scorer\models\current_full.pt" }
$ValidationDir = if ($env:ALICE_VALIDATION_DIR) { $env:ALICE_VALIDATION_DIR } else { Join-Path $Root "scorer\data\validation" }
$ScorerAddress = if ($env:ALICE_SCORER_ADDRESS) { $env:ALICE_SCORER_ADDRESS } else { "" }
$Host = if ($env:ALICE_SCORER_HOST) { $env:ALICE_SCORER_HOST } else { "0.0.0.0" }
$Port = if ($env:ALICE_SCORER_PORT) { $env:ALICE_SCORER_PORT } else { "8090" }
$Device = if ($env:ALICE_SCORER_DEVICE) { $env:ALICE_SCORER_DEVICE } else { "cpu" }
$ModelDtype = if ($env:ALICE_MODEL_DTYPE) { $env:ALICE_MODEL_DTYPE } else { "auto" }
$NumValShards = if ($env:ALICE_NUM_VAL_SHARDS) { $env:ALICE_NUM_VAL_SHARDS } else { "5" }

if (-not (Test-Path $ModelPath)) {
  throw "Missing model checkpoint at $ModelPath. Run .\scorer\bootstrap.ps1 first."
}

if (-not (Test-Path (Join-Path $ValidationDir "shard_059996.pt"))) {
  throw "Missing validation shards in $ValidationDir. Run .\scorer\bootstrap.ps1 first."
}

$Cmd = @(
  "scorer\scoring_server.py",
  "--model-path", $ModelPath,
  "--validation-dir", $ValidationDir,
  "--host", $Host,
  "--port", $Port,
  "--device", $Device,
  "--model-dtype", $ModelDtype,
  "--num-val-shards", $NumValShards,
  "--ps-url", $PsUrl
)

if ($ScorerAddress) {
  $Cmd += @("--scorer-address", $ScorerAddress)
}
if ($args) {
  $Cmd += $args
}

& $VenvPython @Cmd
exit $LASTEXITCODE
