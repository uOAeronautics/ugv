#!/usr/bin/env bash
# Sync the L7 harness to the Pi (~/l7) with the current deploy policy + shield modules.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; repo="$(cd "$here/../.." && pwd)"
ssh -o BatchMode=yes pi 'mkdir -p ~/l7/ugv ~/l7_runs && : > ~/l7/ugv/__init__.py'
rsync -az -e "ssh -o BatchMode=yes" "$here/l7_run.py" pi:l7/
rsync -az -e "ssh -o BatchMode=yes" "$repo/src/ugv/policy.py" "$repo/src/ugv/parking.py" "$repo/src/ugv/default_parking.json" "$repo/src/ugv/staged_baseline.py" pi:l7/ugv/
rsync -az -e "ssh -o BatchMode=yes" "$repo/simulation/gazebo/mecanum_bayesian_profile.json" pi:l7/
