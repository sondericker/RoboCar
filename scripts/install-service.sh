#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_USER="${SUDO_USER:-$(id -un)}"
SERVICE_GROUP="$(id -gn "$SERVICE_USER")"
PYTHON="${ROBOCAR_PYTHON:-/home/$SERVICE_USER/robotics-venv/bin/python}"
KIT_SERVER_PATH=/home/"$SERVICE_USER"/Downloads/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi-master/Code/Server

if [[ ! -x "$PYTHON" ]]; then
  printf 'Python executable not found: %s\nSet ROBOCAR_PYTHON to the app virtualenv Python.\n' "$PYTHON" >&2
  exit 1
fi
if ! "$PYTHON" -c 'import robocar' >/dev/null 2>&1; then
  printf 'RoboCar is not installed in %s. Install the project into this environment first.\n' "$PYTHON" >&2
  exit 1
fi
if [[ "$SERVICE_USER" == root ]]; then
  printf 'Run this script as the normal Raspberry Pi user, not root.\n' >&2
  exit 1
fi

read -r -p "Enable physical motor and pan/tilt outputs? [y/N]: " ENABLE_HARDWARE
HARDWARE_ENABLED=false
if [[ "$ENABLE_HARDWARE" =~ ^[Yy]$ ]]; then
  HARDWARE_ENABLED=true
  if [[ ! -d "$KIT_SERVER_PATH" ]]; then
    read -r -p "Path to Freenove kit Code/Server directory: " KIT_SERVER_PATH
  fi
  if [[ ! -d "$KIT_SERVER_PATH" ]]; then
    printf 'Freenove server directory not found: %s\n' "$KIT_SERVER_PATH" >&2
    exit 1
  fi
fi

ENV_FILE=/etc/robocar/robocar.env
SERVICE_FILE=/etc/systemd/system/robocar.service
sudo install -d -o root -g root -m 0750 /etc/robocar
{
  printf 'ROBOCAR_HOST=0.0.0.0\n'
  printf 'ROBOCAR_PORT=5000\n'
  printf 'ROBOCAR_CAMERA_BACKEND=picamera2\n'
  printf 'ROBOCAR_CAMERA_INDEX=0\n'
  printf 'ROBOCAR_CAMERA_ENABLED=false\n'
  printf 'ROBOCAR_HARDWARE_ENABLED=%s\n' "$HARDWARE_ENABLED"
  printf 'FREENOVE_SERVER_PATH=%s\n' "$KIT_SERVER_PATH"
  printf 'ROBOCAR_MAX_MOTOR_DUTY=4095\n'
  printf 'ROBOCAR_COMMAND_TIMEOUT=1.0\n'
} | sudo tee "$ENV_FILE" >/dev/null
sudo chown root:root "$ENV_FILE"
sudo chmod 0600 "$ENV_FILE"

sed \
  -e "s|@SERVICE_USER@|$SERVICE_USER|g" \
  -e "s|@SERVICE_GROUP@|$SERVICE_GROUP|g" \
  -e "s|@PROJECT_DIR@|$PROJECT_DIR|g" \
  -e "s|@PYTHON@|$PYTHON|g" \
  "$PROJECT_DIR/deploy/robocar.service.in" | sudo tee "$SERVICE_FILE" >/dev/null
sudo chmod 0644 "$SERVICE_FILE"
sudo systemctl daemon-reload
sudo systemctl enable --now robocar.service

printf '\nRoboCar is enabled to start automatically at boot.\n'
printf 'Open http://<car-ip>:5000 on your trusted home network.\n'
printf 'Check service status with: sudo systemctl status robocar\n'
if [[ "$HARDWARE_ENABLED" == false ]]; then
  printf 'Physical outputs remain disabled. To enable later, edit %s and set ROBOCAR_HARDWARE_ENABLED=true.\n' "$ENV_FILE"
fi
