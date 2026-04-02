$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$PsUrl = if ($env:ALICE_PS_URL) { $env:ALICE_PS_URL } else { "https://ps.aliceprotocol.org" }
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
    "--scorer-only" { }
    default { $ExtraArgs += $args[$i] }
  }
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  throw "Missing Python 3.10+."
}

python - <<'PY'
import platform
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required.")
print(f"[bootstrap] Python: {platform.python_version()}")
print(f"[bootstrap] Platform: {platform.system()} {platform.machine()}")
PY

if (-not (Test-Path .venv\Scripts\python.exe)) {
  python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r scorer/requirements.txt

New-Item -ItemType Directory -Force scorer\models | Out-Null
New-Item -ItemType Directory -Force scorer\data | Out-Null

if (-not $ModelPath) {
  $ModelPath = Join-Path $Root "scorer\models\current_full.pt"
}

if (-not (Test-Path $ModelPath)) {
  & .\.venv\Scripts\python.exe - <<PY
from pathlib import Path
import requests

ps_url = "${PsUrl}".rstrip("/")
model_path = Path(r"${ModelPath}")
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
PY
}

if (-not $ValidationDir) {
  $ValidationDir = Join-Path $Root "scorer\data\validation"
}

if (-not (Test-Path $ValidationDir)) {
  throw "Missing validation directory: $ValidationDir. Provide --validation-dir."
}

$Cmd = @(
  "scorer\scoring_server.py",
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

& .\.venv\Scripts\python.exe @Cmd
