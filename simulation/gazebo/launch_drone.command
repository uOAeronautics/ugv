#!/bin/zsh
set -e
cd "${0:A:h}"
export PATH="/opt/homebrew/bin:$PATH"
python3 generate.py --drone --output drone_docking_world.sdf
python3 build_motor_plugin.py
export GZ_SIM_SYSTEM_PLUGIN_PATH="$PWD/plugins/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
exec gz sim -r --gui-config drone_gui.config drone_docking_world.sdf
