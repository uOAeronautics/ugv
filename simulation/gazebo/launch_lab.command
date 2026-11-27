#!/bin/zsh
set -e
cd "${0:A:h}"
export PATH="/opt/homebrew/bin:$PATH"
# payload_car is static (visual/collision only) until characterised
# (research/tasks/2026-11-27-characterise-payload-car.md).
# Pass --car freenove to drive the archived Freenove car.
if [[ " $* " != *" --car freenove "* ]]; then
  echo "Note: payload_car is static (geometry/visual only) until characterised."
  echo "To drive the archived Freenove car, run: ./launch_lab.command --car freenove"
fi
python3 generate.py --room --drone --output lab_docking_world.sdf "$@"
python3 build_motor_plugin.py
export GZ_SIM_SYSTEM_PLUGIN_PATH="$PWD/plugins/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
exec gz sim -r --gui-config lab_gui.config lab_docking_world.sdf
