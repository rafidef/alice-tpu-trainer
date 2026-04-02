#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PS_URL="${ALICE_PS_URL:-https://ps.aliceprotocol.org}"
SHARD_BASE_URL="${ALICE_SHARD_BASE_URL:-https://dl.aliceprotocol.org/shards}"
MODEL_PATH=""
VALIDATION_DIR=""
SCORER_ADDRESS="${ALICE_SCORER_ADDRESS:-}"
PUBLIC_ENDPOINT="${ALICE_PUBLIC_ENDPOINT:-}"
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
    --public-endpoint)
      PUBLIC_ENDPOINT="${2:-}"
      shift 2
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

mkdir -p "$VALIDATION_DIR"
mkdir -p "$(dirname "$VALIDATION_DIR")"

echo "[bootstrap] Ensuring validation shards exist in $VALIDATION_DIR"
if ! python - <<PY
from pathlib import Path
import json
import requests

base_url = "${SHARD_BASE_URL}".rstrip("/")
validation_dir = Path("${VALIDATION_DIR}")
validation_root = validation_dir.parent
validation_dir.mkdir(parents=True, exist_ok=True)
validation_root.mkdir(parents=True, exist_ok=True)

shard_ids = [59996, 59997, 59998, 59999, 60000]
index_url = f"{base_url}/shard_index.json"
index_resp = requests.get(index_url, timeout=30)
index_resp.raise_for_status()
index_data = index_resp.json()
total_shards = int(index_data.get("total_shards") or index_data.get("shard_count") or 60001)

local_index = {
    "total_shards": total_shards,
    "shard_count": total_shards,
    "shards": [{"filename": f"shard_{idx:06d}.pt"} for idx in range(total_shards)],
}
(validation_root / "shard_index.json").write_text(json.dumps(local_index), encoding="utf-8")

for shard_id in shard_ids:
    shard_name = f"shard_{shard_id:06d}.pt"
    target = validation_dir / shard_name
    if target.exists() and target.stat().st_size > 0:
        continue
    shard_url = f"{base_url}/{shard_name}"
    with requests.get(shard_url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with target.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
PY
then
  echo "Failed to download validation shards from ${SHARD_BASE_URL}"
  echo "Expected ${SHARD_BASE_URL}/shard_index.json and shard_059996.pt..shard_060000.pt"
  exit 1
fi

if [[ ! -f "$VALIDATION_DIR/shard_059996.pt" ]]; then
  echo "Failed to download validation shards from ${SHARD_BASE_URL}"
  echo "Expected ${SHARD_BASE_URL}/shard_index.json and shard_059996.pt..shard_060000.pt"
  exit 1
fi

CMD=(
  ./scorer/run_scorer.sh
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

if [[ -n "$PUBLIC_ENDPOINT" ]]; then
  CMD+=(--public-endpoint "$PUBLIC_ENDPOINT")
fi

CMD+=("${EXTRA_ARGS[@]}")
exec "${CMD[@]}"
