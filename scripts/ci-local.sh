#!/usr/bin/env bash
# Mirrors .github/workflows/ci.yml so you can run the same gate locally
# before pushing.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== ruff =="
ruff check .

echo "== mypy =="
mypy src/jevgc

echo "== pytest (coverage gate: 85%) =="
pytest

echo "All checks passed."
