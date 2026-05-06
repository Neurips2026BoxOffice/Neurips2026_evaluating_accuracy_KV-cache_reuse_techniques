#!/usr/bin/env bash
# exp/intro launcher — distributes (model, dataset, shard) jobs across
# accelerator slots. Each slot drains its own queue sequentially;
# different slots (and extra PACK slots on the same device) run in
# parallel. Add a device later by calling this script again with the new
# accelerator id — the per-job output check
# will skip whatever the original workers finished while the new one
# picks up the rest.
#
# Knobs (env):
#   MODELS    space-separated paths   (default: llama / mistral / qwen)
#   DATASETS  subset of {musique_lb, 2wiki_lb, hotpotqa_lb}
#   GPUS      space-separated accelerator ids; jobs are distributed round-robin.
#   SHARDS    K>=1 (default 1). When K>1, each (model, dataset) is split
#             into K shards and run_intro.py is invoked per shard with
#             --shard <i> --num-shards <K>. Round-robin spans the full
#             (model, dataset, shard) cross product so all devices stay
#             busy. After all workers exit, aggregate_shards.py is run
#             to merge the per-shard files back into the canonical
#             intro_<ds>_<model>.json.
#   FORCE     1 to overwrite existing intro_*.json (passed to run_intro.py)
#   PACK      slots-per-accelerator (integer ≥ 1, default 1). Set to 2 to run
#             two concurrent jobs per device, 3 for three, etc. Only bump
#             when you know the extra jobs fit in device memory. The
#             `--pack` CLI flag is a shorthand for PACK=2.
#   RATIOS    space-separated recompute ratios to pass as
#             --recomp-ratios. Default = unset (run_intro uses its own
#             default of [0.15]).
#   MERGE     1 to pass --merge, which folds new ratio keys into the
#             existing intro_<ds>_<model>.json instead of skipping or
#             overwriting. Use when complementing an earlier run.
#   METHODS   space-separated method keys to restrict the grid (e.g.
#             METHODS="cb_k0"). Forwarded as --methods. Default: all
#             entries in the METHODS list in run_intro.py.
#
# Output:
#   runtime/output/intro_<dataset>_<model>.json
#   runtime/output/intro_<dataset>_<model>.log
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
RUNNER_OUT_DIR="${OUT_DIR:-output}"
mkdir -p "$RUNNER_OUT_DIR"

PACK="${PACK:-1}"
for arg in "$@"; do
    case "$arg" in
        --pack) PACK=2 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done
if ! [[ "$PACK" =~ ^[0-9]+$ ]] || [ "$PACK" -lt 1 ]; then
    echo "PACK must be a positive integer (got: $PACK)" >&2; exit 2
fi

MODELS="${MODELS:-/data/weights/llama3.1-8BI /data/weights/mistral-7B /data/weights/qwen3-8B}"
DATASETS="${DATASETS:-musique_lb 2wiki_lb hotpotqa_lb}"
GPUS="${GPUS:-0}"
SHARDS="${SHARDS:-1}"
FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"
MERGE_FLAG=""
[ "${MERGE:-0}" = "1" ] && MERGE_FLAG="--merge"

# RATIOS="0 0.30 0.50" → --recomp-ratios 0 0.30 0.50
RATIOS_ARGS=()
if [ -n "${RATIOS:-}" ]; then
    read -r -a _ratio_arr <<< "$RATIOS"
    RATIOS_ARGS=(--recomp-ratios "${_ratio_arr[@]}")
fi

# METHODS="cb_k0" → --methods cb_k0
METHODS_ARGS=()
if [ -n "${METHODS:-}" ]; then
    read -r -a _method_arr <<< "$METHODS"
    METHODS_ARGS=(--methods "${_method_arr[@]}")
fi

if ! [[ "$SHARDS" =~ ^[0-9]+$ ]] || [ "$SHARDS" -lt 1 ]; then
    echo "SHARDS must be a positive integer (got: $SHARDS)" >&2; exit 2
fi

# ── pre-flight: kill leftover run_intro.py from any prior aborted run ──
SCRIPT_PATH="$HERE/run_intro.py"
if pgrep -f "python .*${SCRIPT_PATH}" > /dev/null 2>&1; then
    echo "==> killing leftover run_intro.py instances"
    pkill -f "python .*${SCRIPT_PATH}" || true
    sleep 1
fi

# ── ctrl-c propagation: kill all spawned python on signal ──────────────
cleanup() {
    echo "==> caught signal, killing children"
    pkill -P $$ 2>/dev/null || true
    pkill -f "python .*${SCRIPT_PATH}" 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

read -r -a DEVICE_ARR <<< "$GPUS"
N_DEVICES="${#DEVICE_ARR[@]}"

SLOTS_PER_DEVICE="$PACK"
TOTAL_SLOTS=$((N_DEVICES * SLOTS_PER_DEVICE))

# Build per-(device, slot) job lists. Each entry is "MODEL|DATASET|SHARD".
declare -A SLOT_QUEUE
for n in "${DEVICE_ARR[@]}"; do
    for s in $(seq 0 $((SLOTS_PER_DEVICE - 1))); do
        SLOT_QUEUE["$n:$s"]=""
    done
done

i=0
for MODEL in $MODELS; do
    for DS in $DATASETS; do
        for SH in $(seq 0 $((SHARDS - 1))); do
            slot=$((i % TOTAL_SLOTS))
            npu_idx=$((slot / SLOTS_PER_DEVICE))
            sub=$((slot % SLOTS_PER_DEVICE))
            DEVICE_ID="${DEVICE_ARR[$npu_idx]}"
            SLOT_QUEUE["$DEVICE_ID:$sub"]+="${MODEL}|${DS}|${SH}"$'\n'
            i=$((i + 1))
        done
    done
done

echo "Plan ($i jobs across $N_DEVICES device(s) × $SLOTS_PER_DEVICE slot(s) = $TOTAL_SLOTS workers; SHARDS=$SHARDS):"
for n in "${DEVICE_ARR[@]}"; do
    for s in $(seq 0 $((SLOTS_PER_DEVICE - 1))); do
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            IFS='|' read -r m d sh <<< "$line"
            echo "  device=$n slot=$s  $(basename "$m") / $d  shard=$sh/$SHARDS"
        done <<< "${SLOT_QUEUE["$n:$s"]}"
    done
done

worker() {
    local device_id=$1
    local slot=$2
    local queue=$3
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        local model ds sh
        IFS='|' read -r model ds sh <<< "$line"
        local mshort="$(basename "$model")"
        local tag="${ds}_${mshort}"
        [ "$SHARDS" -gt 1 ] && tag="${tag}.shard${sh}of${SHARDS}"
        local log="$RUNNER_OUT_DIR/intro_${tag}.log"
        echo "[device $device_id slot $slot] start $mshort/$ds shard=$sh  log=$log"
        python run_intro.py --model "$model" --datasets "$ds" \
            --device-id "$device_id" --out-dir "$RUNNER_OUT_DIR" \
            --shard "$sh" --num-shards "$SHARDS" $FORCE_FLAG $MERGE_FLAG \
            "${RATIOS_ARGS[@]}" "${METHODS_ARGS[@]}" \
            > "$log" 2>&1 \
        && echo "[device $device_id slot $slot] done  $mshort/$ds shard=$sh" \
        || echo "[device $device_id slot $slot] FAIL  $mshort/$ds shard=$sh  (see $log)" >&2
    done <<< "$queue"
    echo "[device $device_id slot $slot] queue empty, exiting"
}

for n in "${DEVICE_ARR[@]}"; do
    for s in $(seq 0 $((SLOTS_PER_DEVICE - 1))); do
        worker "$n" "$s" "${SLOT_QUEUE["$n:$s"]}" &
    done
done
wait
echo "all device workers exited."

if [ "$SHARDS" -gt 1 ]; then
    echo "==> aggregating shards"
    python aggregate_shards.py --in-dir "$RUNNER_OUT_DIR"
fi
