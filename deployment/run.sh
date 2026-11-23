#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python}" -u "$SCRIPT_DIR/deploy.py" "$@"
