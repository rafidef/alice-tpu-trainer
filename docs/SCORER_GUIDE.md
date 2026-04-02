# Alice Scorer Guide

This guide reflects the current standalone scorer deployment flow.

Current rollout state:
- repository is private
- external scorer deployments are still coordinated manually

## 1. Hardware

- Linux x86 `>= 32GB RAM`: recommended, `float32`
- Linux x86 `24-32GB RAM`: usable, `float16`, slower
- Mac ARM `>= 24GB unified memory`: supported, `float16`
- Windows `>= 32GB RAM`: experimental, `float32`

## 2. Files required

- `scorer/scoring_server.py`
- `shared/model.py`
- `shared/__init__.py`
- `core/reporting.py`
- model checkpoint
- validation shard set (auto-fetched by bootstrap)

## 3. Start the scorer

```bash
./scorer/bootstrap.sh
```

Windows:

```powershell
.\scorer\bootstrap.ps1
```

For long-running operation with automatic restart:

- Linux/macOS: `./scorer/install-service.sh`
- Windows: `.\scorer\install-service.ps1`

Linux systemd installation requires `sudo`.

Optional manual launch:

```bash
python3 scorer/scoring_server.py \
  --model-path /path/to/current_full.pt \
  --validation-dir /path/to/validation \
  --host 0.0.0.0 \
  --port 8090 \
  --device cpu \
  --model-dtype auto \
  --num-val-shards 5 \
  --ps-url https://ps.aliceprotocol.org \
  --scorer-address aYourScorerAddress
```

## 4. Model sync

The scorer sync path is:
- `/model/info`
- `/model/delta`
- `/model`

Normal update flow is delta-first and should not require restart.

## 5. Chain flow

To join the scorer set:

1. fund scorer address
2. stake `5000 ALICE`
3. activate scorer
4. add endpoint to the aggregator scorer pool

Current scorer reward pool: `6%`

## 6. Windows note

Windows scorer is experimental support. There is no known architecture-level blocker, but it must still pass full validation before public release.

## 7. Epoch reports

Scorer writes local epoch reports to:

- `~/.alice/reports/scorer_epoch_reports.jsonl`
- `~/.alice/reports/epochs/scorer_epoch_<epoch>.md`

Reward values are labeled as:

- `confirmed`: reward was confirmed from a trusted balance source
- `pending`: the epoch ended but reward confirmation is not available yet

## 8. Managed service mode

Managed mode is recommended for long-running scorers.

- Linux uses `systemd`
- macOS uses `launchd`
- Windows uses Task Scheduler

Service logs are written to `~/.alice/logs/` or `%USERPROFILE%\.alice\logs\`.

Optional overrides:

- Unix: `~/.alice/scorer-service.env`
- Windows: `~\.alice\scorer-service.ps1`

Service manager commands:

- Linux/macOS: `./scorer/start-service.sh`, `./scorer/stop-service.sh`, `./scorer/status-service.sh`, `./scorer/uninstall-service.sh`
- Windows: `.\scorer\start-service.ps1`, `.\scorer\stop-service.ps1`, `.\scorer\status-service.ps1`, `.\scorer\uninstall-service.ps1`
