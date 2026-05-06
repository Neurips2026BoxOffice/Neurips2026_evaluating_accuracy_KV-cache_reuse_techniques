#!/usr/bin/env bash
# Phase 2 — rebuild the filtered-v2 joint shared seeds from corpus.
# Calls the inner driver experiments/v2_filtered/run_joint_filtered_probe.sh
# which uses pilot_eval.py --method baseline (= BaselineNosep) to compute
# the per-(model,seed) baseline F1 and question-only F1 used by the
# joint filter.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_ROOT="${HERE}/.."
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${GEN_ROOT}/regenerated/filtered_v2_joint_probe_${TS}}"
WARM_FUSION_DOCS="${WARM_FUSION_DOCS:-${GEN_ROOT}/artifacts/rebuilt_warm_k5/fusion_docs_topk5.jsonl}"

if [[ ! -f "${WARM_FUSION_DOCS}" ]]; then
  "${HERE}/01_rebuild_warm_fusion_docs.sh"
fi

GEN_ROOT="${GEN_ROOT}" \
RUN_ROOT="${OUT_ROOT}" \
WARM_FUSION_DOCS="${WARM_FUSION_DOCS}" \
DEVICE_POOL_CSV="${DEVICE_POOL_CSV:-0,1}" \
MODELS_CSV="${MODELS_CSV:-qwen3_8B,llama3.1-8BI,mistral}" \
bash "${GEN_ROOT}/experiments/v2_filtered/run_joint_filtered_probe.sh"

echo "${OUT_ROOT}"
