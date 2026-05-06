#!/usr/bin/env bash
# Phase 4 — build per-seed paper2-style pool artifacts from the canonical corpus.
#
# Outputs under pool_v4_balanced_m4/:
#   boxoffice_s<seed>_chunks.json
#   boxoffice_s<seed>_embeddings.npy
#   boxoffice_s<seed>_text_md5.json
#   boxoffice_s<seed>_infusion_k5.json
#   boxoffice_s<seed>_infusion_k10.json
#
# This step is independent from warm-fusion metadata. It should be run after
# seed generation when you want to reproduce the downstream paper2 runtime pool.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HERE}/.."

CORPUS_JSONL="${CORPUS_JSONL:-${ROOT}/corpus/canonical_corpus_100_chunks.jsonl}"
OUT_DIR="${OUT_DIR:-${ROOT}/pool_v4_balanced_m4}"
SEEDS_CSV="${SEEDS_CSV:-7,11}"
KS_CSV="${KS_CSV:-5,10}"
DEVICES_CSV="${DEVICES_CSV:-0,1}"
PACK="${PACK:-2}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-/data/weights/e5-base-v2}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-64}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"

IFS=',' read -r -a SEEDS_ARR <<< "${SEEDS_CSV}"
IFS=',' read -r -a KS_ARR <<< "${KS_CSV}"
IFS=',' read -r -a DEVICE_ARR <<< "${DEVICES_CSV}"

FORCE_FLAG=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_FLAG=(--force)
fi

python3 "${ROOT}/experiments/pool/build_boxoffice_pool.py" \
  --corpus-jsonl "${CORPUS_JSONL}" \
  --out-dir "${OUT_DIR}" \
  --seeds "${SEEDS_ARR[@]}" \
  --ks "${KS_ARR[@]}" \
  --npus "${DEVICE_ARR[@]}" \
  --pack "${PACK}" \
  --embedding-model-path "${EMBEDDING_MODEL_PATH}" \
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  "${FORCE_FLAG[@]}"
