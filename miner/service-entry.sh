#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${HOME}/.alice/logs"
ENV_FILE="${HOME}/.alice/miner-service.env"
mkdir -p "$LOG_DIR"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

ARGS=()
if declare -p ALICE_MINER_ARGS >/dev/null 2>&1; then
  ARGS+=("${ALICE_MINER_ARGS[@]}")
fi
ARGS+=("$@")

exec ./miner/run_miner.sh "${ARGS[@]}" >>"$LOG_DIR/miner-service.log" 2>&1
