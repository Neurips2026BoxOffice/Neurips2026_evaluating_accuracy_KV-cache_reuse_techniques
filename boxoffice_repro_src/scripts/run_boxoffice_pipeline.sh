#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHODS_BUNDLE="${METHODS_BUNDLE:-$(cd "$ROOT/../cache_methods_src" && pwd)}"
BOXOFFICE_BUNDLE="${BOXOFFICE_BUNDLE:-$ROOT}"
RUNTIME_BOXOFFICE="$ROOT/runtime"

GPUS="${GPUS:-}"

SEEDS="${SEEDS:-7 11 13 17 19 23 29 31 47 73}"
MODELS="${MODELS:-/data/weights/llama3.1-8BI /data/weights/mistral /data/weights/qwen3-8B}"
PACK="${PACK:-1}"
POOL_PACK="${POOL_PACK:-1}"
BUILD_POOL="${BUILD_POOL:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
REGENERATE_SEEDS="${REGENERATE_SEEDS:-0}"
FROM_SCRATCH="${FROM_SCRATCH:-0}"
OUT_DIR="${OUT_DIR:-$RUNTIME_BOXOFFICE/output}"
POOL_DIR="${POOL_DIR:-$RUNTIME_BOXOFFICE/pool}"
DATASET_STEM="boxoffice_filtered_v4_balanced_m4_s{seed}_full.jsonl"
EVAL_STEM="boxoffice_filtered_v4_balanced_m4_s{seed}_eval.jsonl"

if [ -z "${GPUS:-}" ] && { [ "$BUILD_POOL" = "1" ] || [ "$RUN_EVAL" = "1" ]; }; then
  echo "GPUS is required when BUILD_POOL=1 or RUN_EVAL=1" >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$POOL_DIR"

if [ "$REGENERATE_SEEDS" = "1" ]; then
  for seed in $SEEDS; do
    if [ "$FROM_SCRATCH" = "1" ]; then
      bash "$BOXOFFICE_BUNDLE/scripts/regenerate_seed_from_scratch.sh" "$seed"
    else
      bash "$BOXOFFICE_BUNDLE/scripts/reproduce_verified_seed.sh" "$seed"
    fi
  done
fi

if [ "$BUILD_POOL" = "1" ]; then
  python "$RUNTIME_BOXOFFICE/build_pool.py" \
    --corpus-jsonl "$BOXOFFICE_BUNDLE/corpus/canonical_corpus_100_chunks.jsonl" \
    --seeds $SEEDS --ks 5 10 --npus $GPUS --pack "$POOL_PACK" \
    --out-dir "$POOL_DIR"
fi

if [ "$RUN_EVAL" = "1" ]; then
  (
    cd "$RUNTIME_BOXOFFICE"
    CACHE_METHODS_SRC="$METHODS_BUNDLE" \
    GPUS="$GPUS" PACK="$PACK" MODELS="$MODELS" SEEDS="$SEEDS" \
    BOXOFFICE_DIR="$BOXOFFICE_BUNDLE/seeds" \
    DATASET_STEM="$DATASET_STEM" EVAL_STEM="$EVAL_STEM" \
    POOL_DIR="$POOL_DIR" OUT_DIR="$OUT_DIR" \
    bash run.sh
  )
fi
