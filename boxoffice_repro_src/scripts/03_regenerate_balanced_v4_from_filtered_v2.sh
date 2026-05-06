#!/usr/bin/env bash
# Phase 3 — derive balanced-M=4 seeds from a filtered-v2 joint probe run.
# No model invocations here; pure post-process on phase-2 output.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_ROOT="${HERE}/.."

PROBE_ROOT="${PROBE_ROOT:?PROBE_ROOT is required (filtered_v2_joint_probe_<TS> dir from phase 2)}"
SEEDS_CSV="${SEEDS_CSV:-7,11}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_TAG="${SOURCE_TAG:-regenerated_filtered_v2_joint_probe}"

mkdir -p "${GEN_ROOT}/seeds" "${GEN_ROOT}/manifests" "${GEN_ROOT}/validation"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
for seed in "${SEEDS[@]}"; do
  "${PYTHON_BIN}" "${GEN_ROOT}/experiments/v4_balanced/build_filtered_v4_balanced.py" \
    --source-full "${PROBE_ROOT}/shared/one_vs_all_v2_joint_s${seed}_full.jsonl" \
    --source-eval "${PROBE_ROOT}/shared/one_vs_all_v2_joint_s${seed}_eval.jsonl" \
    --source-manifest "${PROBE_ROOT}/shared/one_vs_all_v2_joint_s${seed}.manifest.json" \
    --source-corpus "${GEN_ROOT}/corpus/source_corpus_100_films.jsonl" \
    --canonical-corpus "${GEN_ROOT}/corpus/canonical_corpus_100_chunks.jsonl" \
    --seed "${seed}" \
    --source-tag "${SOURCE_TAG}" \
    --out-full "${GEN_ROOT}/seeds/boxoffice_filtered_v4_balanced_m4_s${seed}_full.jsonl" \
    --out-eval "${GEN_ROOT}/seeds/boxoffice_filtered_v4_balanced_m4_s${seed}_eval.jsonl" \
    --out-manifest "${GEN_ROOT}/manifests/boxoffice_filtered_v4_balanced_m4_s${seed}.manifest.json" \
    --out-validation "${GEN_ROOT}/validation/boxoffice_filtered_v4_balanced_m4_s${seed}_direction_counts.json"
done
