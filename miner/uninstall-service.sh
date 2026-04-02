#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="alice-protocol-miner"
LABEL="org.aliceprotocol.miner"
SYSTEMD_UNIT="/etc/systemd/system/${SERVICE_NAME}.service"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"

case "$(uname -s)" in
  Linux)
    if [[ "${EUID}" -eq 0 ]]; then
      systemctl disable --now "${SERVICE_NAME}.service" >/dev/null 2>&1 || true
      rm -f "$SYSTEMD_UNIT"
      systemctl daemon-reload
    else
      sudo systemctl disable --now "${SERVICE_NAME}.service" >/dev/null 2>&1 || true
      sudo rm -f "$SYSTEMD_UNIT"
      sudo systemctl daemon-reload
    fi
    ;;
  Darwin)
    launchctl bootout "gui/$(id -u)" "$PLIST_PATH" >/dev/null 2>&1 || true
    rm -f "$PLIST_PATH"
    ;;
  *)
    echo "Unsupported OS for uninstall-service.sh."
    exit 1
    ;;
esac
