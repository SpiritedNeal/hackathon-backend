#!/usr/bin/env bash
set -euo pipefail

# Find a Python executable (prefer python3)
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "ERROR: no python executable found in PATH" >&2
  exit 1
fi

# Use python -m pip so we don't rely on a standalone pip binary
echo "Using Python: $($PYTHON --version 2>&1)"

# Upgrade pip/setuptools/wheel and install requirements
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install --no-cache-dir -r requirements.txt

# Start FastAPI with uvicorn; ensure module path matches your project
exec "$PYTHON" -m uvicorn quadcode:app --host 0.0.0.0 --port "${PORT:-8000}"
