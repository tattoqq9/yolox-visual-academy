#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/check_site.py site
echo "Windows/WSLブラウザで開く: http://localhost:8765/"
python3 -m http.server 8765 --bind 127.0.0.1 --directory site
