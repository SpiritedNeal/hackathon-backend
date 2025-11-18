#!/usr/bin/env bash
set -e

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start FastAPI using uvicorn on Railway's PORT
exec uvicorn quadcode:app --host 0.0.0.0 --port "${PORT:-8000}"
