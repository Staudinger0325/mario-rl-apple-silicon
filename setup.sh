#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${MARIO_PYTHON:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Please use Python 3.11 (MARIO_PYTHON=/path/to/python3.11)."'
if [ ! -x .venv/bin/python ]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m mario_rl doctor
echo "Ready. Run: .venv/bin/python scripts/play_world1.py"
