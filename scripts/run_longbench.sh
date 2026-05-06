#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHODS_BUNDLE="${METHODS_BUNDLE:-$ROOT/cache_methods_src}"
LONGBENCH_BUNDLE="${LONGBENCH_BUNDLE:-$ROOT/longbench_repro_src}"

cd "$LONGBENCH_BUNDLE"
METHODS_BUNDLE="$METHODS_BUNDLE" \
LONGBENCH_BUNDLE="$LONGBENCH_BUNDLE" \
bash scripts/run_longbench_pipeline.sh "$@"
