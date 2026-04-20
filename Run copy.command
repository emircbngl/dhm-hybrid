#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PY="$ROOT_DIR/venv/bin/python"
elif [[ -x "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3" ]]; then
  PY="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
else
  PY="python3"
fi

"$PY" "$ROOT_DIR/run_app.py"
