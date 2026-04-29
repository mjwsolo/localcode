#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m mkdocs build --strict
python -m mkdocs gh-deploy --force --clean
