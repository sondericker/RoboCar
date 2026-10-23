#!/usr/bin/env bash
set -euo pipefail

sudo systemctl stop robocar.service
printf 'RoboCar has been stopped. It will still start automatically at the next boot.\n'
