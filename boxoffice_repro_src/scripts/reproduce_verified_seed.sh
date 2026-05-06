#!/usr/bin/env bash
# Reproduce one verified boxoffice seed with the validated bundle defaults.
#
# Default behavior:
# - reuse bundled warm metadata
# - use visible devices 0 and 1 only
# - allow up to 6 concurrent eval jobs
# - reproduce exactly one of the verified release seeds:
#   7,11,13,17,19,23,29,31,47,73
#
# Optional behavior:
# - REBUILD_WARM=1 forces warm-metadata regeneration first
# - BUILD_POOL=1 also emits the downstream pool artifacts for this seed
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash scripts/reproduce_verified_seed.sh <seed>" >&2
  echo "supported seeds: 7 11 13 17 19 23 29 31 47 73" >&2
  exit 2
fi

SEED="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEVICE_POOL_CSV="${DEVICE_POOL_CSV:-0,1}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-6}"
MODELS_CSV="${MODELS_CSV:-qwen3_8B,llama3.1-8BI,mistral-7B}"
BUILD_POOL="${BUILD_POOL:-0}"
DEVICES_CSV="${DEVICES_CSV:-0,1}"

SOURCE_TAG=""
OVERSAMPLE=""
case "${SEED}" in
  7|11)
    SOURCE_TAG="regenerated_v4_e2e_20260501_135032"
    OVERSAMPLE="100"
    ;;
  23)
    SOURCE_TAG="regenerated_filtered_v2_joint_probe"
    OVERSAMPLE="100"
    ;;
  13|17|19|29|31|47|73)
    SOURCE_TAG="regenerated_filtered_v2_joint_probe"
    OVERSAMPLE="150"
    ;;
  *)
    echo "unsupported verified seed: ${SEED}" >&2
    echo "supported seeds: 7 11 13 17 19 23 29 31 47 73" >&2
    exit 2
    ;;
esac

if [[ "${REBUILD_WARM:-0}" == "1" ]]; then
  bash "${HERE}/01_rebuild_warm_fusion_docs.sh"
fi

declare -a env_args=(
  "DEVICE_POOL_CSV=${DEVICE_POOL_CSV}"
  "MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}"
  "MODELS_CSV=${MODELS_CSV}"
  "SEEDS_CSV=${SEED}"
  "SOURCE_TAG=${SOURCE_TAG}"
  "BUILD_POOL=${BUILD_POOL}"
  "POOL_SEEDS_CSV=${SEED}"
  "DEVICES_CSV=${DEVICES_CSV}"
)

env_args+=("EVAL_PER_CELL_OVERSAMPLED=${OVERSAMPLE}")

env "${env_args[@]}" bash "${HERE}/regenerate_end_to_end.sh"
