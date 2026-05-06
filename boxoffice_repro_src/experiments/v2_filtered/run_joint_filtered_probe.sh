#!/usr/bin/env bash
# Inner driver — phase 2's joint filter loop.
#
# For each seed:
#   1. generate_one_vs_all_v2.py → over-sampled candidate seeds
#   2. slice eval rows from full
#   3. build_question_only_inputs.py → question-only inputs (no ctx)
#   4. pilot_eval.py --method baseline (PER MODEL, in parallel) on full + qonly
#      ← BaselineNosep here (wired via method_factory)
#   5. build_joint_filtered_dataset.py → keep rows where ALL 3 models pass
#      filter; trim to KEEP_PER_CELL=40
#
# Inputs:
#   WARM_FUSION_DOCS  (default: artifacts/rebuilt_warm_k5/fusion_docs_topk5.jsonl)
#   RUN_ROOT          (required) — output directory for this run
#   GEN_ROOT          (default: parent of v2_filtered's parent dir = generate_boxoffice/)
#
# Env knobs (with defaults that mirror the original internal bundle):
#   SEEDS_CSV=7,11
#   DEVICE_POOL_CSV=0,1,2,3,4,5
#   MAX_PARALLEL_JOBS=<len(DEVICE_POOL_CSV)> by default
#   MODELS_CSV=qwen3_8B,llama3.1-8BI,mistral
#   PROMPT_VARIANT=closed_world
#   EVAL_PER_CELL_OVERSAMPLED=100
#   KEEP_PER_CELL=40
#   PER_CELL_ATTEMPTS=50000
#   TARGET_START_RANK=10  TARGET_END_RANK=70  MIN_BOXOFFICE_GAP=40
#   CHUNK_TARGET_TOKENS=512
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_ROOT="${GEN_ROOT:-$(cd "${HERE}/../.." && pwd)}"
SCRIPT_DIR="${HERE}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DTYPE="${DTYPE:-bfloat16}"
PROMPT_VARIANT="${PROMPT_VARIANT:-closed_world}"
WARM_FUSION_DOCS="${WARM_FUSION_DOCS:-${GEN_ROOT}/artifacts/rebuilt_warm_k5/fusion_docs_topk5.jsonl}"
EVAL_PER_CELL_OVERSAMPLED="${EVAL_PER_CELL_OVERSAMPLED:-100}"
KEEP_PER_CELL="${KEEP_PER_CELL:-40}"
PER_CELL_ATTEMPTS="${PER_CELL_ATTEMPTS:-50000}"
TARGET_START_RANK="${TARGET_START_RANK:-10}"
TARGET_END_RANK="${TARGET_END_RANK:-70}"
MIN_BOXOFFICE_GAP="${MIN_BOXOFFICE_GAP:-40}"
CHUNK_TARGET_TOKENS="${CHUNK_TARGET_TOKENS:-512}"
SEEDS_CSV="${SEEDS_CSV:-7,11}"
DEVICE_POOL_CSV="${DEVICE_POOL_CSV:-0,1,2,3,4,5}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-}"
MODELS_CSV="${MODELS_CSV:-qwen3_8B,llama3.1-8BI,mistral}"

declare -A MODEL_PATH_BY_LABEL
MODEL_PATH_BY_LABEL["qwen3_8B"]="/data/weights/qwen3-8B"
MODEL_PATH_BY_LABEL["llama3.1-8BI"]="/data/weights/llama3.1-8BI"
MODEL_PATH_BY_LABEL["mistral"]="/data/weights/mistral"

export PATH="/usr/local/python3.11.14/bin:${PATH}"
export PYTHONPATH="${GEN_ROOT}/src:${SCRIPT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a DEVICES <<< "${DEVICE_POOL_CSV}"
IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"

if [[ "${#DEVICES[@]}" -eq 0 ]]; then
  echo "DEVICE_POOL_CSV must contain at least one device id" >&2
  exit 2
fi

if [[ -z "${MAX_PARALLEL_JOBS}" ]]; then
  MAX_PARALLEL_JOBS="${#DEVICES[@]}"
fi

mkdir -p "${RUN_ROOT}/inputs" "${RUN_ROOT}/qonly_inputs" "${RUN_ROOT}/logs" "${RUN_ROOT}/results" "${RUN_ROOT}/filtered" "${RUN_ROOT}/shared"
echo RUNNING > "${RUN_ROOT}/STATUS.txt"

count_rows() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
from pathlib import Path
print(sum(1 for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()))
PY
}

device_idx=0
ASSIGNED_DEVICE=""
assign_device() {
  ASSIGNED_DEVICE="${DEVICES[$((device_idx % ${#DEVICES[@]}))]}"
  device_idx=$((device_idx + 1))
}

pids=()
names=()
wait_one() {
  local pid="${pids[0]}"
  local job_name="${names[0]}"
  if wait "${pid}"; then
    echo "[finished] ${job_name}" | tee -a "${RUN_ROOT}/launcher.log"
  else
    echo "[failed] ${job_name}" | tee -a "${RUN_ROOT}/launcher.log" >&2
    return 1
  fi
  pids=("${pids[@]:1}")
  names=("${names[@]:1}")
}

launch_eval() {
  local model_label="$1"
  local input="$2"
  local out="$3"
  local log="$4"
  shift 4
  assign_device
  local dev="${ASSIGNED_DEVICE}"
  local n
  n="$(count_rows "${input}")"
  echo "[launch] $(basename "${out}" .json) device=${dev}" >> "${RUN_ROOT}/launcher.log"
  ASCEND_RT_VISIBLE_DEVICES="${dev}" \
  ASCEND_VISIBLE_DEVICES="${dev}" \
  CUDA_VISIBLE_DEVICES="${dev}" \
  "${PYTHON_BIN}" "${GEN_ROOT}/src/pilot_eval.py" \
    --single-prompt \
    --method baseline \
    --queries "${input}" \
    --model-path "${MODEL_PATH_BY_LABEL[${model_label}]}" \
    --device "${DEVICE:-cuda:0}" \
    --dtype "${DTYPE}" \
    --max-context-tokens 12000 \
    --max-queries "${n}" \
    "$@" \
    --out "${out}" \
    > "${log}" 2>&1 &
  pids+=("$!")
  names+=("$(basename "${out}" .json)")
  if [[ "${#pids[@]}" -ge "${MAX_PARALLEL_JOBS}" ]]; then
    wait_one
  fi
}

for seed in "${SEEDS[@]}"; do
  full_input="${RUN_ROOT}/inputs/one_vs_all_v2_s${seed}_full.jsonl"
  eval_input="${RUN_ROOT}/inputs/one_vs_all_v2_s${seed}_eval.jsonl"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_one_vs_all_v2.py" \
    --warm-fusion-docs "${WARM_FUSION_DOCS}" \
    --source-jsonl "${GEN_ROOT}/data/phase2_boxoffice_balanced_matrix_s7.jsonl" \
    --source-jsonl "${GEN_ROOT}/data/phase2_boxoffice_balanced_matrix_s11.jsonl" \
    --source-jsonl "${GEN_ROOT}/data/supplemental_boxoffice_corpus_v1_50.jsonl" \
    --seed "${seed}" \
    --eval-per-cell "${EVAL_PER_CELL_OVERSAMPLED}" \
    --min-boxoffice-gap "${MIN_BOXOFFICE_GAP}" \
    --chunk-target-tokens "${CHUNK_TARGET_TOKENS}" \
    --prompt-variant "${PROMPT_VARIANT}" \
    --target-start-rank "${TARGET_START_RANK}" \
    --target-end-rank "${TARGET_END_RANK}" \
    --per-cell-attempts "${PER_CELL_ATTEMPTS}" \
    --selection-mode v2b_low_overlap \
    --out "${full_input}" \
    --manifest-out "${RUN_ROOT}/inputs/one_vs_all_v2_s${seed}.manifest.json" \
    > "${RUN_ROOT}/logs/generate_s${seed}.log" 2>&1
  "${PYTHON_BIN}" - "${full_input}" "${eval_input}" <<'PY'
import json, sys
from pathlib import Path
src = Path(sys.argv[1]); dst = Path(sys.argv[2])
with src.open('r', encoding='utf-8') as inp, dst.open('w', encoding='utf-8') as out:
    for line in inp:
        row = json.loads(line)
        if row.get('metadata', {}).get('is_full_reuse_failure_eval'):
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
PY
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_question_only_inputs.py" \
    "${full_input}" "${eval_input}" \
    "${RUN_ROOT}/qonly_inputs/one_vs_all_v2_s${seed}_full.jsonl" \
    "${RUN_ROOT}/qonly_inputs/one_vs_all_v2_s${seed}_eval.jsonl"
done

for model_label in "${MODELS[@]}"; do
  mkdir -p "${RUN_ROOT}/${model_label}/results"
  for seed in "${SEEDS[@]}"; do
    launch_eval "${model_label}" \
      "${RUN_ROOT}/inputs/one_vs_all_v2_s${seed}_eval.jsonl" \
      "${RUN_ROOT}/results/${model_label}_baseline_s${seed}.json" \
      "${RUN_ROOT}/logs/${model_label}_baseline_s${seed}.log"
    launch_eval "${model_label}" \
      "${RUN_ROOT}/qonly_inputs/one_vs_all_v2_s${seed}_eval.jsonl" \
      "${RUN_ROOT}/results/${model_label}_question_only_baseline_s${seed}.json" \
      "${RUN_ROOT}/logs/${model_label}_question_only_baseline_s${seed}.log"
  done
done

while [[ "${#pids[@]}" -gt 0 ]]; do
  wait_one
done

for seed in "${SEEDS[@]}"; do
  baseline_args=()
  qonly_args=()
  for model_label in "${MODELS[@]}"; do
    baseline_args+=(--baseline-result "${model_label}=${RUN_ROOT}/results/${model_label}_baseline_s${seed}.json")
    qonly_args+=(--question-only-result "${model_label}=${RUN_ROOT}/results/${model_label}_question_only_baseline_s${seed}.json")
  done
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_joint_filtered_dataset.py" \
    --source-full "${RUN_ROOT}/inputs/one_vs_all_v2_s${seed}_full.jsonl" \
    --source-eval "${RUN_ROOT}/inputs/one_vs_all_v2_s${seed}_eval.jsonl" \
    --source-manifest "${RUN_ROOT}/inputs/one_vs_all_v2_s${seed}.manifest.json" \
    --models "${MODELS_CSV}" \
    "${baseline_args[@]}" \
    "${qonly_args[@]}" \
    --keep-per-cell "${KEEP_PER_CELL}" \
    --out-full "${RUN_ROOT}/shared/one_vs_all_v2_joint_s${seed}_full.jsonl" \
    --out-eval "${RUN_ROOT}/shared/one_vs_all_v2_joint_s${seed}_eval.jsonl" \
    --out-manifest "${RUN_ROOT}/shared/one_vs_all_v2_joint_s${seed}.manifest.json" \
    > "${RUN_ROOT}/logs/joint_filter_s${seed}.log" 2>&1

  for model_label in "${MODELS[@]}"; do
    cp "${RUN_ROOT}/shared/one_vs_all_v2_joint_s${seed}_full.jsonl" \
      "${RUN_ROOT}/filtered/one_vs_all_v2_${model_label}_s${seed}_full.jsonl"
    cp "${RUN_ROOT}/shared/one_vs_all_v2_joint_s${seed}_eval.jsonl" \
      "${RUN_ROOT}/filtered/one_vs_all_v2_${model_label}_s${seed}_eval.jsonl"
    cp "${RUN_ROOT}/shared/one_vs_all_v2_joint_s${seed}.manifest.json" \
      "${RUN_ROOT}/filtered/one_vs_all_v2_${model_label}_s${seed}.manifest.json"
  done
done

echo DONE > "${RUN_ROOT}/STATUS.txt"
echo "${RUN_ROOT}"
