#!/bin/zsh
# Launcher for the v2 Dear PyGui frontend.
# Double-click from Finder, or run from a terminal.
#
# Multiple venvs live next to each other on this project tree
# (Hybrid/venv, ../Phyton/venv, possibly the system framework
# python). Only the ones that actually have dearpygui installed
# can run the v2 UI, so we *probe* each candidate and pick the
# first one where ``import dearpygui`` succeeds — ordering by
# existence alone silently picked a stripped-down env in the past.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

candidates=(
  "$ROOT_DIR/venv/bin/python"
  "$ROOT_DIR/../Phyton/venv/bin/python"
  "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
  "python3"
)

PY=""
PROBE_FAILED=()
for candidate in "${candidates[@]}"; do
  # Resolve "python3" on PATH when it's not an absolute path.
  if [[ "$candidate" = /* ]]; then
    [[ -x "$candidate" ]] || continue
    cmd="$candidate"
  else
    cmd="$(command -v "$candidate" 2>/dev/null || true)"
    [[ -n "$cmd" && -x "$cmd" ]] || continue
  fi
  if "$cmd" -c "import dearpygui.dearpygui" >/dev/null 2>&1; then
    PY="$cmd"
    break
  fi
  PROBE_FAILED+=("$cmd")
done

if [[ -z "$PY" ]]; then
  echo "error: none of the available Python interpreters has dearpygui installed." >&2
  echo "" >&2
  echo "Probed (in order):" >&2
  for c in "${PROBE_FAILED[@]}"; do
    echo "  · $c" >&2
  done
  echo "" >&2
  echo "Install dearpygui into one of them, e.g.:" >&2
  echo "  ${PROBE_FAILED[1]:-python3} -m pip install dearpygui numpy scipy tifffile h5py imageio" >&2
  exit 1
fi

echo "Using Python: $PY"
# run_ui2.py adds src/ to sys.path for us and dispatches to
# ui2.app.main — no extra PYTHONPATH plumbing needed.
exec "$PY" "$ROOT_DIR/run_ui2.py"
