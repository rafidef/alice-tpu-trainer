#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run ./scorer/bootstrap.sh first."
  exit 1
fi

source .venv/bin/activate

PS_URL="${ALICE_PS_URL:-https://ps.aliceprotocol.org}"
MODEL_PATH="${ALICE_MODEL_PATH:-$ROOT_DIR/scorer/models/current_full.pt}"
VALIDATION_DIR="${ALICE_VALIDATION_DIR:-$ROOT_DIR/scorer/data/validation}"
SCORER_ADDRESS="${ALICE_SCORER_ADDRESS:-}"
HOST="${ALICE_SCORER_HOST:-0.0.0.0}"
PORT="${ALICE_SCORER_PORT:-8090}"
DEVICE="${ALICE_SCORER_DEVICE:-cpu}"
MODEL_DTYPE="${ALICE_MODEL_DTYPE:-auto}"
NUM_VAL_SHARDS="${ALICE_NUM_VAL_SHARDS:-5}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing model checkpoint at $MODEL_PATH. Run ./scorer/bootstrap.sh first."
  exit 1
fi

if [[ ! -f "$VALIDATION_DIR/shard_059996.pt" ]]; then
  echo "Missing validation shards in $VALIDATION_DIR. Run ./scorer/bootstrap.sh first."
  exit 1
fi

CMD=(
  python scorer/scoring_server.py
  --model-path "$MODEL_PATH"
  --validation-dir "$VALIDATION_DIR"
  --host "$HOST"
  --port "$PORT"
  --device "$DEVICE"
  --model-dtype "$MODEL_DTYPE"
  --num-val-shards "$NUM_VAL_SHARDS"
  --ps-url "$PS_URL"
)

if [[ -n "$SCORER_ADDRESS" ]]; then
  CMD+=(--scorer-address "$SCORER_ADDRESS")
fi

CMD+=("$@")
exec "${CMD[@]}"
