#!/usr/bin/env bash
# Phase 1 — rebuild top-K=5 warm-fusion retrieval docs from raw corpus.
#
# Driver:
#   experiments/v2_filtered/build_one_vs_all_v2_canonical_semantic_warm_cache.py
# Helpers:
#   experiments/v1_helpers/build_one_vs_all_semantic_warm_cache.py (uses the
#   v2 builder via PYTHONPATH so its imports resolve)
# Output:
#   artifacts/rebuilt_warm_k5/fusion_docs_topk5.jsonl
#
# Inputs (under data/, all bundled locally):
#   phase2_boxoffice_balanced_matrix_s7.jsonl
#   phase2_boxoffice_balanced_matrix_s11.jsonl
#   supplemental_boxoffice_corpus_v1_50.jsonl
#
# Override env vars: MODEL_PATH, EMBEDDING_MODEL_PATH, DEVICE, …
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HERE}/.."

OUT_DIR="${OUT_DIR:-${ROOT}/artifacts/rebuilt_warm_k5}"
SOURCE_JSONL_CSV="${SOURCE_JSONL_CSV:-${ROOT}/data/phase2_boxoffice_balanced_matrix_s7.jsonl,${ROOT}/data/phase2_boxoffice_balanced_matrix_s11.jsonl,${ROOT}/data/supplemental_boxoffice_corpus_v1_50.jsonl}"
MODEL_PATH="${MODEL_PATH:-/data/weights/qwen3-8B}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-/data/weights/e5-base-v2}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda:0}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-32}"

mkdir -p "${OUT_DIR}"

PYTHONPATH="${ROOT}/src:${ROOT}/experiments/v1_helpers:${PYTHONPATH:-}" \
python3 "${ROOT}/experiments/v2_filtered/build_one_vs_all_v2_canonical_semantic_warm_cache.py" \
  --source-jsonl-csv "${SOURCE_JSONL_CSV}" \
  --model-path "${MODEL_PATH}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --top-k 5 \
  --chunk-target-tokens 512 \
  --retrieval-mode embedding \
  --embedding-model-path "${EMBEDDING_MODEL_PATH}" \
  --embedding-device "${EMBEDDING_DEVICE}" \
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
  --out-canonical-retrieval "${OUT_DIR}/canonical_retrieval_topk5.jsonl" \
  --out-soup "${OUT_DIR}/soup_topk5.jsonl" \
  --out-fusion-docs "${OUT_DIR}/fusion_docs_topk5.jsonl" \
  --out-stats "${OUT_DIR}/stats_topk5.json" \
  --skip-cache-export
