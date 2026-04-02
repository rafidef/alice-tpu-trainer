#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PS_URL="${ALICE_PS_URL:-https://ps.aliceprotocol.org}"
MODEL_PATH=""
VALIDATION_DIR=""
SCORER_ADDRESS="${ALICE_SCORER_ADDRESS:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="${2:-}"
      shift 2
      ;;
    --validation-dir)
      VALIDATION_DIR="${2:-}"
      shift 2
      ;;
    --ps-url)
      PS_URL="${2:-}"
      shift 2
      ;;
    --scorer-address)
      SCORER_ADDRESS="${2:-}"
      shift 2
      ;;
    --scorer-only)
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing Python 3.10+ (${PYTHON_BIN})."
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import platform
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required.")
print(f"[bootstrap] Python: {platform.python_version()}")
print(f"[bootstrap] Platform: {platform.system()} {platform.machine()}")
PY

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scorer/requirements.txt

mkdir -p scorer/models scorer/data

if [[ -z "$MODEL_PATH" ]]; then
  MODEL_PATH="$ROOT_DIR/scorer/models/current_full.pt"
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[bootstrap] Downloading current model from $PS_URL/model"
  python - <<PY
import json
from pathlib import Path
import requests

ps_url = "${PS_URL}".rstrip("/")
model_path = Path("${MODEL_PATH}")
model_path.parent.mkdir(parents=True, exist_ok=True)

with requests.get(f"{ps_url}/model", stream=True, timeout=3600) as resp:
    resp.raise_for_status()
    with model_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)

info = requests.get(f"{ps_url}/model/info", timeout=30)
if info.ok:
    payload = info.json()
    version = payload.get("version")
    if version is not None:
        (model_path.parent / "current_version.txt").write_text(f"{version}\n", encoding="utf-8")
PY
fi

if [[ -z "$VALIDATION_DIR" ]]; then
  VALIDATION_DIR="$ROOT_DIR/scorer/data/validation"
fi

if [[ ! -d "$VALIDATION_DIR" ]]; then
  echo "Missing validation directory: $VALIDATION_DIR"
  echo "Provide --validation-dir /path/to/validation_shards"
  exit 1
fi

CMD=(
  python scorer/scoring_server.py
  --model-path "$MODEL_PATH"
  --validation-dir "$VALIDATION_DIR"
  --host 0.0.0.0
  --port 8090
  --device cpu
  --model-dtype auto
  --ps-url "$PS_URL"
)

if [[ -n "$SCORER_ADDRESS" ]]; then
  CMD+=(--scorer-address "$SCORER_ADDRESS")
fi

CMD+=("${EXTRA_ARGS[@]}")
exec "${CMD[@]}"
