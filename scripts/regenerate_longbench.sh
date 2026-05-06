#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/longbench_repro_src"
bash scripts/regenerate_enriched_longbench.sh "$@"
