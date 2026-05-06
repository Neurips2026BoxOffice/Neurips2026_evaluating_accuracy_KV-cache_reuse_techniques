#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/boxoffice_repro_src"
bash scripts/reproduce_verified_ten_seeds.sh "$@"
