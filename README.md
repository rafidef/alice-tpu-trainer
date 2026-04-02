# Alice Protocol

Private release-prep repository for the public Alice miner and scorer distribution.

This repository is the staged public candidate for:
- `miner/`: external miner client
- `scorer/`: external scorer worker
- `shared/`: shared Alice model runtime

Current status:
- private only
- external miner access remains restricted
- documentation is being aligned with the live mainnet implementation

## Repository Layout

```text
Alice-Protocol/
├── miner/
├── scorer/
├── shared/
├── core/
├── docs/
└── LICENSE
```

## Components

### Miner

Entry point: `miner/alice_miner.py`

Bootstrap:

```bash
./miner/bootstrap.sh
```

Windows:

```powershell
.\miner\bootstrap.ps1
```

### Scorer

Entry point: `scorer/scoring_server.py`

Bootstrap:

```bash
./scorer/bootstrap.sh --validation-dir /path/to/validation
```

Windows:

```powershell
.\scorer\bootstrap.ps1 --validation-dir C:\path\to\validation
```

## Documentation

- `docs/MINER_GUIDE.md`
- `docs/MINER_GUIDE_CN.md`
- `docs/SCORER_GUIDE.md`
- `docs/HARDWARE_REQUIREMENTS.md`

## Notes

- This repository is not public yet.
- Network admission for outside miners remains restricted.
- Bootstrap is the default user entry point for both miner and scorer.
- Per-epoch local reports are written to `~/.alice/reports/`.
- Users who only want scorer files can use sparse checkout after cloning this same repo.
- Full cross-platform validation remains a release gate before any public launch.
