$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

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
& .\.venv\Scripts\python.exe -m pip install -r miner/requirements.txt

$Wallet = Join-Path $HOME ".alice\wallet.json"
if (-not (Test-Path $Wallet)) {
  & .\.venv\Scripts\python.exe miner\alice_wallet.py create
}

& .\miner\run_miner.bat @args
