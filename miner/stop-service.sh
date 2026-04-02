#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="alice-protocol-miner"
LABEL="org.aliceprotocol.miner"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"

case "$(uname -s)" in
  Linux)
    if [[ "${EUID}" -eq 0 ]]; then
      systemctl stop "${SERVICE_NAME}.service"
    else
      sudo systemctl stop "${SERVICE_NAME}.service"
    fi
    ;;
  Darwin)
    if [[ -f "$PLIST_PATH" ]]; then
      launchctl bootout "gui/$(id -u)" "$PLIST_PATH" >/dev/null 2>&1 || true
    else
      launchctl bootout "gui/$(id -u)/${LABEL}" >/dev/null 2>&1 || true
    fi
    ;;
  *)
    echo "Unsupported OS for stop-service.sh."
    exit 1
    ;;
esac
