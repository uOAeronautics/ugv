#!/bin/zsh
set -e
cd "${0:A:h}"
export PATH="/opt/homebrew/bin:$PATH"
python3 generate.py --room --drone --output lab_docking_world.sdf
python3 build_motor_plugin.py
export GZ_SIM_SYSTEM_PLUGIN_PATH="$PWD/plugins/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
exec gz sim -r --gui-config lab_gui.config lab_docking_world.sdf
