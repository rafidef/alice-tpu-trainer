#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="alice-protocol-scorer"
LABEL="org.aliceprotocol.scorer"

case "$(uname -s)" in
  Linux)
    if [[ "${EUID}" -eq 0 ]]; then
      systemctl status "${SERVICE_NAME}.service" --no-pager
    else
      sudo systemctl status "${SERVICE_NAME}.service" --no-pager
    fi
    ;;
  Darwin)
    launchctl print "gui/$(id -u)/${LABEL}"
    ;;
  *)
    echo "Unsupported OS for status-service.sh."
    exit 1
    ;;
esac
