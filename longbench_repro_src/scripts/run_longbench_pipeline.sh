#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHODS_BUNDLE="${METHODS_BUNDLE:-$(cd "$ROOT/../cache_methods_src" && pwd)}"
LONGBENCH_BUNDLE="${LONGBENCH_BUNDLE:-$ROOT}"

GPUS="${GPUS:-}"

if [ -z "${GPUS:-}" ]; then
  echo "GPUS is required" >&2
  exit 2
fi

ENV_VARS=(
  "CACHE_METHODS_SRC=$METHODS_BUNDLE"
  "LONGBENCH_DATA_DIR=$LONGBENCH_BUNDLE/data/enriched"
  "GPUS=$GPUS"
)
[ -n "${MODELS:-}" ] && ENV_VARS+=("MODELS=$MODELS")
[ -n "${DATASETS:-}" ] && ENV_VARS+=("DATASETS=$DATASETS")
[ -n "${METHODS:-}" ] && ENV_VARS+=("METHODS=$METHODS")
[ -n "${PACK:-}" ] && ENV_VARS+=("PACK=$PACK")
[ -n "${SHARDS:-}" ] && ENV_VARS+=("SHARDS=$SHARDS")
[ -n "${FORCE:-}" ] && ENV_VARS+=("FORCE=$FORCE")
[ -n "${RATIOS:-}" ] && ENV_VARS+=("RATIOS=$RATIOS")
[ -n "${OUT_DIR:-}" ] && ENV_VARS+=("OUT_DIR=$OUT_DIR")

(
  cd "$LONGBENCH_BUNDLE/runtime"
  env "${ENV_VARS[@]}" bash run.sh
)
