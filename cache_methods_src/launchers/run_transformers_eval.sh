#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash src/launchers/run_transformers_eval.sh \
#     --method lmcache_online \
#     --model-path /data/weights/llama3.1-8BI \
#     --traces data/eval_traces.jsonl \
#     --out results/lmcache_online_eval.json \
#     --max-traces 50

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python3 src/transformers_eval.py "$@"
