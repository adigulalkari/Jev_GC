#!/usr/bin/env bash
# One-shot dev environment bootstrap.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,langgraph,strands]"

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp .env.example .env
  echo "Created .env from .env.example — fill in JEV_API_KEY before running examples."
fi

echo "Done. Activate with: source .venv/bin/activate"
