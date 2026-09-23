#!/usr/bin/env bash
# Mirrors .github/workflows/ci.yml so you can run the same gate locally
# before pushing.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== ruff =="
ruff check .

echo "== mypy =="
mypy src/jevgc

# Don't restate the threshold here; pyproject.toml owns it, and a copy in
# this script is a copy that drifts.
echo "== pytest (coverage gate per pyproject.toml) =="
pytest

echo "All checks passed."
