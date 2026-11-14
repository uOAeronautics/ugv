#!/usr/bin/env bash
# Pull the raw JPEG frames of every L7 run (slow link; run in the background).
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
rsync -az -e "ssh -o BatchMode=yes" pi:l7_runs/ "$repo/calibration/runs/l7/"
