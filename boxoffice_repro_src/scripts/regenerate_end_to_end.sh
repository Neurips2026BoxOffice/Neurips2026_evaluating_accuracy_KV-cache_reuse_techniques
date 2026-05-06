#!/usr/bin/env bash
# End-to-end orchestrator: phase 1 → phase 2 → phase 3.
# Produces seeds/boxoffice_filtered_v4_balanced_m4_s{7,11}_{full,eval}.jsonl
# under this bundle. Optionally emits the downstream paper2-style pool
# artifacts when BUILD_POOL=1. The PROBE_ROOT used by phase 3 is the
# timestamped directory created by phase 2.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_ROOT="${HERE}/.."
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PROBE_ROOT="${PROBE_ROOT:-${GEN_ROOT}/regenerated/filtered_v2_joint_probe_${TS}}"
SOURCE_TAG="${SOURCE_TAG:-regenerated_v4_e2e_${TS}}"

if [[ ! -d "${PROBE_ROOT}" ]]; then
  TS="${TS}" OUT_ROOT="${PROBE_ROOT}" "${HERE}/02_regenerate_filtered_v2_from_corpus.sh"
fi

PROBE_ROOT="${PROBE_ROOT}" SEEDS_CSV="${SEEDS_CSV:-7,11}" SOURCE_TAG="${SOURCE_TAG}" \
  "${HERE}/03_regenerate_balanced_v4_from_filtered_v2.sh"

if [[ "${BUILD_POOL:-0}" == "1" ]]; then
  SEEDS_CSV="${POOL_SEEDS_CSV:-${SEEDS_CSV:-7,11}}" \
    bash "${HERE}/04_build_boxoffice_pool.sh"
fi
