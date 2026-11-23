#!/usr/bin/env bash
# Install and check slide-puzzle-solver. Safe to run from any directory: it works in its own directory,
# so it runs the same from the repo root or from anywhere else.
set -euo pipefail
# Any failing step fails the whole run with exit 1, whatever code the tool itself exits with (tsc exits 2).
trap 'exit 1' ERR
cd "$(dirname "$0")"
npm ci
npm run typecheck
npm run lint
npm run format:check
npm test
npm run build
npx tsx scripts/check-dist.ts

# Python (synthetic generator, trainer and their tests): ruff lint, docstrings and format with the
# standards' ruff.toml, mypy strict, pytest. The venv (numpy, pillow, torch, pytest, mypy) lives in
# ~/.cache; .venv links to it so tools that look for a project venv find it.
bash scripts/python-env.sh -c "" >/dev/null
venv="${SPS_VENV:-$HOME/.cache/slide-puzzle-solver/venv}"
[ -e .venv ] || ln -s "$venv" .venv
ruff_bin="$(command -v ruff || true)"
if [ -z "$ruff_bin" ]; then
  ruff_venv="$HOME/.cache/slide-puzzle-solver/ruff-venv"
  [ -x "$ruff_venv/bin/ruff" ] || { python3 -m venv "$ruff_venv" && "$ruff_venv/bin/pip" install -q ruff==0.17.0; }
  ruff_bin="$ruff_venv/bin/ruff"
fi
"$ruff_bin" check scripts tests
"$ruff_bin" check --select D scripts tests
"$ruff_bin" format --check scripts tests
"$venv/bin/python" -m mypy .
"$venv/bin/python" -m pytest -q

# The browser test needs Playwright's Chromium and its system libraries. If Chromium cannot be
# launched here the checks fail: exiting 0 means the browser suite ran and passed.
# Some images preset PLAYWRIGHT_BROWSERS_PATH to a read-only directory holding another version.
if [ -n "${PLAYWRIGHT_BROWSERS_PATH:-}" ] && [ ! -w "$PLAYWRIGHT_BROWSERS_PATH" ]; then
  export PLAYWRIGHT_BROWSERS_PATH="$HOME/.cache/ms-playwright"
fi
# Advisory: a failed install is reported (last lines) and the launch check below decides.
npx playwright install chromium 2>&1 | tail -n 3 >&2 || true
if node -e 'import("@playwright/test").then(({ chromium }) => chromium.launch()).then((b) => b.close())' >/dev/null 2>&1; then
  npm run e2e
else
  echo "e2e: Chromium could not be launched; install it with 'npx playwright install --with-deps chromium' (see README, 'Browser e2e')" >&2
  exit 1
fi
