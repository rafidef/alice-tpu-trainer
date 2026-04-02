# Alice Miner

This is the standalone miner client for Alice Protocol.

Current access policy:
- repository is private
- outside miner admission remains restricted

## Quick Start

### macOS / Linux

```bash
./miner/bootstrap.sh
```

### Windows

```powershell
.\miner\bootstrap.ps1
```

## Wallet

Create a local wallet:

```bash
python3 miner/alice_wallet.py create
```

Use an existing reward address:

```bash
python3 miner/alice_miner.py \
  --ps-url https://ps.aliceprotocol.org \
  --address aYourAliceAddress \
  --reward-address aYourRewardAddress
```

## Epoch Reports

Miner epoch reports are written to:

- `~/.alice/reports/miner_epoch_reports.jsonl`
- `~/.alice/reports/epochs/miner_epoch_<epoch>.md`

Each report includes work completed, loss, gradient submission counts, and reward status (`confirmed`, `pending`, or `estimate`).

## Hardware

See `docs/HARDWARE_REQUIREMENTS.md` for the full matrix.

Summary:
- CUDA GPU `>= 24GB`: recommended
- Apple Silicon `>= 24GB`: supported
- CPU `>= 32GB RAM`: supported but very slow
- `< 20GB`: not supported

CPU mining is supported but not recommended. Expect roughly `1/50 - 1/100` of GPU throughput and proportionally lower rewards.
