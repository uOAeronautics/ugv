#!/usr/bin/env bash
# Run one harness mode on the Pi, then pull the run directory (JPEGs included) back.
# usage: tools/l7/run.sh <mode> [l7_run.py args...]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; repo="$(cd "$here/../.." && pwd)"
dest="$repo/calibration/runs/l7"; mkdir -p "$dest"
"$here/deploy.sh"
ssh -o BatchMode=yes pi "cd ~/l7 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/ugv/.venv/bin/python l7_run.py $*" 2>&1 | grep -v '^\*\*' || true
# The WireGuard path is ~100 KB/s: pull records now, raw frames later with pull_frames.sh.
rsync -az --exclude "frames/" -e "ssh -o BatchMode=yes" pi:l7_runs/ "$dest/" 2>/dev/null
ls -1d "$dest"/2* | tail -1
