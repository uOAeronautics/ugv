#!/bin/zsh
set -e
cd "${0:A:h}"
export PATH="/opt/homebrew/bin:$PATH"
# Drives the archived Freenove car (freenove_world.sdf). The default payload_car
# has no measured mass or motor model yet (research/tasks/2026-11-27-characterise-payload-car.md);
# regenerate with `python3 generate.py --car freenove ...` to drive.
echo "Note: control panel drives the archived Freenove car (--car freenove)."
echo "The default payload_car is static until characterised."
if ! command -v gz >/dev/null; then
  echo "Gazebo is required. Install it before opening these controls."
  exit 1
fi
if [[ ! -x .control-venv/bin/python ]]; then
  uv venv --python "$(command -v python3)" --system-site-packages .control-venv
  uv pip install --python .control-venv/bin/python 'protobuf>=7.36.2,<8'
fi
.control-venv/bin/python build_motor_plugin.py
exec .control-venv/bin/python control_panel.py "$@"
