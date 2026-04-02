# Alice Scorer

This is the standalone scorer worker for Alice Protocol.

Current access policy:
- repository is private
- external scorer operations are still coordinated manually

## Quick Start

```bash
./scorer/bootstrap.sh --validation-dir /path/to/validation
```

## Platform Defaults

- Linux x86 `>= 32GB RAM`: `float32`
- Linux x86 `24-32GB RAM`: `float16` fallback, slower
- Mac ARM `>= 24GB unified memory`: `float16`
- Windows `>= 32GB RAM`: experimental, `float32`

## Epoch Reports

Scorer epoch reports are written to:

- `~/.alice/reports/scorer_epoch_reports.jsonl`
- `~/.alice/reports/epochs/scorer_epoch_<epoch>.md`

Each report includes scored submissions, average score latency, model version, and reward status (`confirmed`, `pending`, or `estimate`).

## Chain Flow

1. Fund scorer address
2. Stake `5000 ALICE`
3. Activate scorer
4. Add scorer endpoint to aggregator pool
5. Confirm `/health` and first scored request

Current scorer reward pool: `6%`

## Scorer-only checkout

By default, users clone the full `Alice-Protocol` repository. Advanced users who only want scorer files can use git sparse checkout to pull `scorer/`, `shared/`, `core/`, and `docs/` only.
