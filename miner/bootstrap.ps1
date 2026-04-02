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

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

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
& $VenvPython -m pip install -r miner/requirements.txt

$Wallet = Join-Path $HOME ".alice\wallet.json"
if (-not (Test-Path $Wallet)) {
  & $VenvPython miner\alice_wallet.py create
}

& .\miner\run_miner.bat @args
