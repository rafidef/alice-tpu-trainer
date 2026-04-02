$ErrorActionPreference = "Stop"

function Resolve-PythonCommand {
  $python = Get-Command python -ErrorAction SilentlyContinue
  if ($python) {
    return @($python.Source)
  }

  $py = Get-Command py -ErrorAction SilentlyContinue
  if ($py) {
    return @($py.Source, "-3")
  }

  throw "Missing Python 3.10+."
}

function Invoke-PythonTempScript {
  param(
    [string]$PythonExe,
    [string]$Script,
    [string[]]$ScriptArgs = @()
  )

  $tmp = [System.IO.Path]::GetTempFileName()
  $pyPath = [System.IO.Path]::ChangeExtension($tmp, ".py")
  Remove-Item $tmp -ErrorAction SilentlyContinue
  Set-Content -Path $pyPath -Value $Script -Encoding UTF8
  try {
    & $PythonExe $pyPath @ScriptArgs
  }
  finally {
    Remove-Item $pyPath -ErrorAction SilentlyContinue
  }
}

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$PsUrl = if ($env:ALICE_PS_URL) { $env:ALICE_PS_URL } else { "https://ps.aliceprotocol.org" }
$ShardBaseUrl = if ($env:ALICE_SHARD_BASE_URL) { $env:ALICE_SHARD_BASE_URL } else { "https://dl.aliceprotocol.org/shards" }
$ModelPath = ""
$ValidationDir = ""
$ScorerAddress = if ($env:ALICE_SCORER_ADDRESS) { $env:ALICE_SCORER_ADDRESS } else { "" }
$ExtraArgs = @()

for ($i = 0; $i -lt $args.Count; $i++) {
  switch ($args[$i]) {
    "--model-path" { $ModelPath = $args[$i + 1]; $i++ }
    "--validation-dir" { $ValidationDir = $args[$i + 1]; $i++ }
    "--ps-url" { $PsUrl = $args[$i + 1]; $i++ }
    "--scorer-address" { $ScorerAddress = $args[$i + 1]; $i++ }
    default { $ExtraArgs += $args[$i] }
  }
}

$PythonCmd = Resolve-PythonCommand
$PythonExe = $PythonCmd[0]
$PythonArgs = @()
if ($PythonCmd.Length -gt 1) {
  $PythonArgs = $PythonCmd[1..($PythonCmd.Length - 1)]
}
& $PythonExe @PythonArgs -c "import platform, sys; assert sys.version_info >= (3, 10), 'Python 3.10+ is required.'; print(f'[bootstrap] Python: {platform.python_version()}'); print(f'[bootstrap] Platform: {platform.system()} {platform.machine()}')"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  & $PythonExe @PythonArgs -m venv .venv
}

$VenvPython = ".\.venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r scorer/requirements.txt

New-Item -ItemType Directory -Force scorer\models | Out-Null
New-Item -ItemType Directory -Force scorer\data | Out-Null

if (-not $ModelPath) {
  $ModelPath = Join-Path $Root "scorer\models\current_full.pt"
}

if (-not (Test-Path $ModelPath)) {
  $downloadModelScript = @"
from pathlib import Path
import requests
import sys

ps_url = sys.argv[1].rstrip("/")
model_path = Path(sys.argv[2])
model_path.parent.mkdir(parents=True, exist_ok=True)

with requests.get(f"{ps_url}/model", stream=True, timeout=3600) as resp:
    resp.raise_for_status()
    with model_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)

info = requests.get(f"{ps_url}/model/info", timeout=30)
if info.ok:
    payload = info.json()
    version = payload.get("version")
    if version is not None:
        (model_path.parent / "current_version.txt").write_text(f"{version}\n", encoding="utf-8")
"@
  Invoke-PythonTempScript -PythonExe $VenvPython -Script $downloadModelScript -ScriptArgs @($PsUrl, $ModelPath)
}

if (-not $ValidationDir) {
  $ValidationDir = Join-Path $Root "scorer\data\validation"
}

New-Item -ItemType Directory -Force $ValidationDir | Out-Null
$ValidationRoot = Split-Path -Parent $ValidationDir
New-Item -ItemType Directory -Force $ValidationRoot | Out-Null

$downloadValidationScript = @"
from pathlib import Path
import json
import requests
import sys

base_url = sys.argv[1].rstrip("/")
validation_dir = Path(sys.argv[2])
validation_root = validation_dir.parent
validation_dir.mkdir(parents=True, exist_ok=True)
validation_root.mkdir(parents=True, exist_ok=True)

shard_ids = [59996, 59997, 59998, 59999, 60000]
index_url = f"{base_url}/shard_index.json"
index_resp = requests.get(index_url, timeout=30)
index_resp.raise_for_status()
index_data = index_resp.json()
total_shards = int(index_data.get("total_shards") or index_data.get("shard_count") or 60001)

shards = [{"filename": f"shard_{idx:06d}.pt"} for idx in range(total_shards)]
local_index = {
    "total_shards": total_shards,
    "shard_count": total_shards,
    "shards": shards,
}
(validation_root / "shard_index.json").write_text(json.dumps(local_index), encoding="utf-8")

for shard_id in shard_ids:
    shard_name = f"shard_{shard_id:06d}.pt"
    target = validation_dir / shard_name
    if target.exists() and target.stat().st_size > 0:
        continue
    shard_url = f"{base_url}/{shard_name}"
    with requests.get(shard_url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with target.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
"@
try {
  Invoke-PythonTempScript -PythonExe $VenvPython -Script $downloadValidationScript -ScriptArgs @($ShardBaseUrl, $ValidationDir)
}
catch {
  throw "Failed to download validation shards from $ShardBaseUrl (expected $ShardBaseUrl/shard_index.json and shard_059996.pt..shard_060000.pt): $($_.Exception.Message)"
}

$Cmd = @(
  "--model-path", $ModelPath,
  "--validation-dir", $ValidationDir,
  "--host", "0.0.0.0",
  "--port", "8090",
  "--device", "cpu",
  "--model-dtype", "auto",
  "--ps-url", $PsUrl
)

if ($ScorerAddress) {
  $Cmd += @("--scorer-address", $ScorerAddress)
}
$Cmd += $ExtraArgs

& .\scorer\run_scorer.ps1 @Cmd
exit $LASTEXITCODE
