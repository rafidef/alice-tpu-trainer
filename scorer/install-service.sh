#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="alice-protocol-scorer"
LABEL="org.aliceprotocol.scorer"
USER_NAME="${SUDO_USER:-$USER}"
HOME_DIR="$(eval echo "~${USER_NAME}")"
LOG_DIR="${HOME_DIR}/.alice/logs"
SYSTEMD_UNIT="/etc/systemd/system/${SERVICE_NAME}.service"
PLIST_PATH="${HOME_DIR}/Library/LaunchAgents/${LABEL}.plist"

mkdir -p "$LOG_DIR"

install_linux() {
  if [[ "${EUID}" -ne 0 ]] && ! command -v sudo >/dev/null 2>&1; then
    echo "Linux service install requires sudo."
    exit 1
  fi

  if [[ "${EUID}" -eq 0 ]]; then
    cat >"$SYSTEMD_UNIT" <<EOF
[Unit]
Description=Alice Protocol Scorer
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER_NAME}
Environment=HOME=${HOME_DIR}
WorkingDirectory=${ROOT_DIR}
ExecStart=/bin/bash ${ROOT_DIR}/scorer/service-entry.sh
Restart=always
RestartSec=10
StartLimitIntervalSec=0

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now "${SERVICE_NAME}.service"
  else
    sudo tee "$SYSTEMD_UNIT" >/dev/null <<EOF
[Unit]
Description=Alice Protocol Scorer
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER_NAME}
Environment=HOME=${HOME_DIR}
WorkingDirectory=${ROOT_DIR}
ExecStart=/bin/bash ${ROOT_DIR}/scorer/service-entry.sh
Restart=always
RestartSec=10
StartLimitIntervalSec=0

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable --now "${SERVICE_NAME}.service"
  fi

  echo "Installed Linux systemd service: ${SERVICE_NAME}.service"
}

install_macos() {
  mkdir -p "$(dirname "$PLIST_PATH")"
  cat >"$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
      <string>/bin/bash</string>
      <string>${ROOT_DIR}/scorer/service-entry.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${ROOT_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/scorer-service.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/scorer-service.log</string>
  </dict>
</plist>
EOF

  launchctl bootout "gui/$(id -u)" "$PLIST_PATH" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH"
  launchctl kickstart -k "gui/$(id -u)/${LABEL}"
  echo "Installed macOS launchd service: ${LABEL}"
}

case "$(uname -s)" in
  Linux) install_linux ;;
  Darwin) install_macos ;;
  *)
    echo "Unsupported OS for install-service.sh. Use install-service.ps1 on Windows."
    exit 1
    ;;
esac
