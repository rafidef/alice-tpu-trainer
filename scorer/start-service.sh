#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="alice-protocol-scorer"
LABEL="org.aliceprotocol.scorer"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"

case "$(uname -s)" in
  Linux)
    if [[ "${EUID}" -eq 0 ]]; then
      systemctl start "${SERVICE_NAME}.service"
    else
      sudo systemctl start "${SERVICE_NAME}.service"
    fi
    ;;
  Darwin)
    if [[ -f "$PLIST_PATH" ]]; then
      launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH" >/dev/null 2>&1 || true
    fi
    launchctl kickstart -k "gui/$(id -u)/${LABEL}"
    ;;
  *)
    echo "Unsupported OS for start-service.sh."
    exit 1
    ;;
esac
