# Alice Scorer

This is the standalone scorer worker for Alice Protocol.

Current access policy:
- repository is private
- external scorer operations are still coordinated manually

## Quick Start

```bash
./scorer/bootstrap.sh
```

### Windows

```powershell
.\scorer\bootstrap.ps1
```

## Long-Running Mode

Foreground bootstrap is the default for first-run setup and debugging.

For long-running operation with automatic restart:

- Linux/macOS: `./scorer/install-service.sh`
- Windows: `.\scorer\install-service.ps1`

Linux systemd installation requires `sudo`.

Service logs are written to `~/.alice/logs/` on Unix-like systems and `%USERPROFILE%\.alice\logs\` on Windows.

Optional service overrides:

- Unix: `~/.alice/scorer-service.env`
- Windows: `~\.alice\scorer-service.ps1`

Use the service manager commands after installation:

- Linux/macOS: `./scorer/start-service.sh`, `./scorer/stop-service.sh`, `./scorer/status-service.sh`, `./scorer/uninstall-service.sh`
- Windows: `.\scorer\start-service.ps1`, `.\scorer\stop-service.ps1`, `.\scorer\status-service.ps1`, `.\scorer\uninstall-service.ps1`

## Platform Defaults

- Linux x86 `>= 32GB RAM`: `float32`
- Linux x86 `24-32GB RAM`: `float16` fallback, slower
- Mac ARM `>= 24GB unified memory`: `float16`
- Windows `>= 32GB RAM`: experimental, `float32`

## Epoch Reports

Scorer epoch reports are written to:

- `~/.alice/reports/scorer_epoch_reports.jsonl`
- `~/.alice/reports/epochs/scorer_epoch_<epoch>.md`

Each report includes scored submissions, average score latency, model version, and reward status (`confirmed` or `pending`).

## Chain Flow

1. Fund scorer address
2. Stake `5000 ALICE`
3. Activate scorer
4. Add scorer endpoint to aggregator pool
5. Confirm `/health` and first scored request

Current scorer reward pool: `6%`

Bootstrap downloads the model and the held-out validation shards automatically into the repo-local scorer directories by default.
