#!/usr/bin/env bash
# Reproduce the original verified five-seed subset using the validated defaults.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${REBUILD_WARM:-0}" == "1" ]]; then
  REBUILD_WARM=1 bash "${HERE}/reproduce_verified_seed.sh" 7
else
  bash "${HERE}/reproduce_verified_seed.sh" 7
fi

bash "${HERE}/reproduce_verified_seed.sh" 11
bash "${HERE}/reproduce_verified_seed.sh" 23
bash "${HERE}/reproduce_verified_seed.sh" 47
bash "${HERE}/reproduce_verified_seed.sh" 73
