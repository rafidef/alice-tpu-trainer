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
set USER_ADDR=
set REWARD_ADDR=
set PS_URL=
if defined ALICE_PS_URL (
  set DEFAULT_PS_URL=%ALICE_PS_URL%
) else (
  set DEFAULT_PS_URL=https://ps.aliceprotocol.org
)
set EXTRA_ARGS=

:scanargs
if "%~1"=="" goto afterscan
if "%~1"=="--address" (
  set HASADDR=1
  set USER_ADDR=%~2
  set EXTRA_ARGS=%EXTRA_ARGS% --address "%~2"
  shift
  shift
  goto scanargs
)
if "%~1"=="--reward-address" (
  set HASREWARD=1
  set REWARD_ADDR=%~2
  set EXTRA_ARGS=%EXTRA_ARGS% --reward-address "%~2"
  shift
  shift
  goto scanargs
)
if "%~1"=="--ps-url" (
  set HASPS=1
  set PS_URL=%~2
  set EXTRA_ARGS=%EXTRA_ARGS% --ps-url "%~2"
  shift
  shift
  goto scanargs
)
set EXTRA_ARGS=%EXTRA_ARGS% "%~1"
shift
goto scanargs

:afterscan
if not defined HASADDR (
  if not exist "%WALLET%" (
    .venv\Scripts\python.exe miner\alice_wallet.py create
  )
  for /f "usebackq delims=" %%A in (`.venv\Scripts\python.exe -c "import json, pathlib; print(json.loads((pathlib.Path.home()/'.alice'/'wallet.json').read_text())['address'])"`) do set WALLET_ADDR=%%A
  set USER_ADDR=%WALLET_ADDR%
)

if not defined HASREWARD (
  if defined USER_ADDR (
    set REWARD_ADDR=%USER_ADDR%
  )
)

set CMDARGS=
if defined USER_ADDR set CMDARGS=%CMDARGS% --address "%USER_ADDR%"
if defined REWARD_ADDR set CMDARGS=%CMDARGS% --reward-address "%REWARD_ADDR%"
if defined HASPS (
  set CMDARGS=%CMDARGS% --ps-url "%PS_URL%"
) else (
  set CMDARGS=%CMDARGS% --ps-url "%DEFAULT_PS_URL%"
)
set CMDARGS=%CMDARGS% %EXTRA_ARGS%

.venv\Scripts\python.exe miner\alice_miner.py %CMDARGS%
