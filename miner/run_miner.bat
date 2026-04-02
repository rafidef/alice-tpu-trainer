@echo off
setlocal

set ROOT=%~dp0..
cd /d %ROOT%

if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Run miner\install.ps1 first.
  exit /b 1
)

set WALLET=%USERPROFILE%\.alice\wallet.json
set HASADDR=
set HASREWARD=
set HASPS=
set WALLET_ADDR=
set CMDARGS=%*

:scan
if "%~1"=="" goto afterscan
if "%~1"=="--address" set HASADDR=1
if "%~1"=="--reward-address" set HASREWARD=1
if "%~1"=="--ps-url" set HASPS=1
shift
goto scan

:afterscan
if defined HASADDR goto run

if not exist "%WALLET%" (
  .venv\Scripts\python.exe miner\alice_wallet.py create
)

for /f "usebackq delims=" %%A in (`.venv\Scripts\python.exe -c "import json, pathlib; print(json.loads((pathlib.Path.home()/'.alice'/'wallet.json').read_text())['address'])"`) do set WALLET_ADDR=%%A

if not defined HASADDR set CMDARGS=--address %WALLET_ADDR% %CMDARGS%

:run
if not defined HASREWARD (
  set CMDARGS=--reward-address %WALLET_ADDR% %CMDARGS%
)
if not defined HASPS (
  set CMDARGS=--ps-url https://ps.aliceprotocol.org %CMDARGS%
)
.venv\Scripts\python.exe miner\alice_miner.py %CMDARGS%
