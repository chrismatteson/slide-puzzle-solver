#!/usr/bin/env bash
# ENG-012 evidence: the Playwright suite drives the built app in a real Chromium at 390x844. Every
# test fails on any console error or network request after load, and screenshots of each screen are
# written to e2e-artifacts/. Unlike checks.sh, this fails when Chromium cannot be launched.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d node_modules ] || npm ci --silent
# Only this run's screenshots count as evidence.
rm -rf e2e-artifacts
# Some images preset PLAYWRIGHT_BROWSERS_PATH to a read-only directory holding another version.
if [ -n "${PLAYWRIGHT_BROWSERS_PATH:-}" ] && [ ! -w "$PLAYWRIGHT_BROWSERS_PATH" ]; then
  export PLAYWRIGHT_BROWSERS_PATH="$HOME/.cache/ms-playwright"
fi
# Advisory: a failed install is reported (last lines) and the launch check below decides.
npx playwright install chromium 2>&1 | tail -n 3 >&2 || true
npx playwright test --reporter=line
echo "screenshots:"
ls e2e-artifacts/*.png
