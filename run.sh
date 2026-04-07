#!/usr/bin/env bash
# run.sh — start the Meet Transcriber web UI

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check dependencies
if ! python3 -c "import flask" 2>/dev/null; then
  echo "Flask not found. Run: bash setup.sh"
  exit 1
fi

echo ""
echo "  ┌─────────────────────────────────────┐"
echo "  │      Meet Transcriber  🎙            │"
echo "  │  Opening http://localhost:5050       │"
echo "  └─────────────────────────────────────┘"
echo ""

python3 server.py
