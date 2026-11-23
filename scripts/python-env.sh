#!/usr/bin/env bash
# Usage: python-env.sh <script.py> [args...]. Creates the venv on first use, then runs the script in it.
# The venv (torch, ~1 GB) lives outside the project so it is not copied with the workspace;
# override its location with SPS_VENV.
set -euo pipefail
cd "$(dirname "$0")/.."

venv="${SPS_VENV:-$HOME/.cache/slide-puzzle-solver/venv}"
# Reinstall when requirements.txt changes, so an existing venv never runs with stale packages.
stamp="$venv/.requirements.sha256"
if [ ! -x "$venv/bin/python" ] || ! sha256sum --status -c "$stamp" 2>/dev/null; then
  [ -x "$venv/bin/python" ] || python3 -m venv "$venv"
  "$venv/bin/python" -m pip install -q --no-cache-dir -r requirements.txt
  sha256sum "$PWD/requirements.txt" >"$stamp"
fi
exec "$venv/bin/python" "$@"
