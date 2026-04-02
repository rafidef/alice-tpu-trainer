#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run ./miner/install.sh first."
  exit 1
fi

source .venv/bin/activate

WALLET_PATH="${HOME}/.alice/wallet.json"
DEFAULT_PS_URL="${ALICE_PS_URL:-https://ps.aliceprotocol.org}"
HAS_ADDRESS=false
HAS_REWARD_ADDRESS=false
HAS_PS_URL=false
for arg in "$@"; do
  if [[ "$arg" == "--address" ]]; then
    HAS_ADDRESS=true
  fi
  if [[ "$arg" == "--reward-address" ]]; then
    HAS_REWARD_ADDRESS=true
  fi
  if [[ "$arg" == "--ps-url" ]]; then
    HAS_PS_URL=true
  fi
done

if [[ "$HAS_ADDRESS" == false ]]; then
  if [[ ! -f "$WALLET_PATH" ]]; then
    python miner/alice_wallet.py create
  fi
  WALLET_ADDRESS="$(python - <<'PY'
import json
from pathlib import Path
path = Path.home() / ".alice" / "wallet.json"
print(json.loads(path.read_text())["address"])
PY
)"
  set -- --address "$WALLET_ADDRESS" "$@"
fi

if [[ "$HAS_REWARD_ADDRESS" == false ]]; then
  if [[ -z "${WALLET_ADDRESS:-}" ]]; then
    WALLET_ADDRESS="$(python - <<'PY'
import json
from pathlib import Path
path = Path.home() / ".alice" / "wallet.json"
print(json.loads(path.read_text())["address"])
PY
)"
  fi
  set -- --reward-address "$WALLET_ADDRESS" "$@"
fi

if [[ "$HAS_PS_URL" == false ]]; then
  set -- --ps-url "$DEFAULT_PS_URL" "$@"
fi

exec python miner/alice_miner.py "$@"
