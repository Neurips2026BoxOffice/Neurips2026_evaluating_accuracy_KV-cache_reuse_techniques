#!/usr/bin/env bash
# Regenerate one BoxOffice seed from bundled source inputs, forcing a warm
# similarity metadata rebuild first. This is the explicit "from scratch" path.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash scripts/regenerate_seed_from_scratch.sh <seed>" >&2
  echo "supported seeds: 7 11 13 17 19 23 29 31 47 73" >&2
  exit 2
fi

SEED="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REBUILD_WARM=1 BUILD_POOL="${BUILD_POOL:-0}" \
  bash "${HERE}/reproduce_verified_seed.sh" "${SEED}"
