#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHODS_BUNDLE="${METHODS_BUNDLE:-$ROOT/cache_methods_src}"
BOXOFFICE_BUNDLE="${BOXOFFICE_BUNDLE:-$ROOT/boxoffice_repro_src}"

cd "$BOXOFFICE_BUNDLE"
METHODS_BUNDLE="$METHODS_BUNDLE" \
BOXOFFICE_BUNDLE="$BOXOFFICE_BUNDLE" \
bash scripts/run_boxoffice_pipeline.sh "$@"
