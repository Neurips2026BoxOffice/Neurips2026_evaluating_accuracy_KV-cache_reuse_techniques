#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

python "$ROOT/longbench_scripts/download.py" \
  --output-dir "$ROOT/data/longbench_raw"
python "$ROOT/longbench_scripts/download_originals.py" \
  --output-dir "$ROOT/data/originals"
python "$ROOT/longbench_scripts/enrich.py" \
  --longbench-dir "$ROOT/data/longbench_raw" \
  --originals-dir "$ROOT/data/originals" \
  --output-dir "$ROOT/data/enriched"
