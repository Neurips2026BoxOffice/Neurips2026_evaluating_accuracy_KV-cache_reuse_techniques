#!/usr/bin/env bash
# exp/boxoffice launcher — distribute (model × seed × shard) jobs across
# accelerator slots. One Python process per slot by default
# (PACK=1 default). Different slots run in parallel.
#
# Knobs (env):
#   MODELS    space-separated paths   (default: llama / mistral / qwen)
#   SEEDS     space-separated seeds   (default: "7 11")
#   GPUS      space-separated accelerator ids
#   SHARDS    K>=1; when K>1 each (model, seed) is split into K shards
#             and run_boxoffice.py is invoked per shard with --shard i
#             --num-shards K. Aggregated by aggregate_shards.py at the
#             end of this launcher.
#             *** Must stay K=1 when any V3 method (ccv3_*) is in the
#             grid: V3 caches are stateful across queries and accumulate
#             over the warmup→eval order in `_full.jsonl`; sharding
#             strides through that order and breaks the warm benefit. ***
#   PACK      slots-per-accelerator (default 1; bump only if you know the
#             extra processes fit in device memory).
#   FORCE     1 → pass --force to overwrite existing outputs
#   RATIOS    space-separated recompute ratios (forwarded to
#             --recomp-ratios; default = run_boxoffice's
#             [0, 0.05, 0.10, 0.15]).
#   METHODS   space-separated method keys (e.g. "cb_k0 cb_k0q
#             ccv3_m1_diffkv ccv3_m2_q"). Default: full grid as
#             defined in run_boxoffice.py:METHODS.
#
# OOM-safety grouping (per cell):
#   Each V3 method (`ccv3_*`) holds its own multi-version KV-chunk
#   cache that grows over the warmup→eval order in `_full.jsonl`.
#   Running multiple V3 variants in the same Python process risks
#   OOM. The launcher therefore splits METHODS into groups:
#     • all stateless cb/fr methods → 1 invocation
#     • each V3 method              → 1 invocation
#   Groups run sequentially per cell; each invocation passes
#   `--merge` so its per-query method/ratio keys fold into the same
#   per-seed JSON. The model is loaded once per group (~30s startup
#   per group on top of compute).
#   BOXOFFICE_DIR  alt path to the reference_inputs/ jsonl bundle
#                  (default in run_boxoffice.py).
#
# Inputs (per seed):
#   <BOXOFFICE_DIR>/..._s<seed>_full.jsonl   warmup queries first, then
#                                            eval queries in order.
#   <BOXOFFICE_DIR>/..._s<seed>_eval.jsonl   eval-id source — rows in
#                                            `_full.jsonl` whose query_id
#                                            matches one in this file get
#                                            `is_eval=true` in output.
#
# Output:
#   boxoffice/output/boxoffice_s<seed>_<model>.json
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p output

PACK="${PACK:-1}"
if ! [[ "$PACK" =~ ^[0-9]+$ ]] || [ "$PACK" -lt 1 ]; then
    echo "PACK must be a positive integer (got: $PACK)" >&2; exit 2
fi

MODELS="${MODELS:-/data/weights/llama3.1-8BI /data/weights/mistral-7B /data/weights/qwen3-8B}"
SEEDS="${SEEDS:-7 11}"
GPUS="${GPUS:-0}"
SHARDS="${SHARDS:-1}"
FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"

RATIOS_ARGS=()
if [ -n "${RATIOS:-}" ]; then
    read -r -a _ratio_arr <<< "$RATIOS"
    RATIOS_ARGS=(--recomp-ratios "${_ratio_arr[@]}")
fi

# Method scope. Default to the full grid baked into run_boxoffice.py
# so we can group it into V3-safe chunks below.
METHODS_DEFAULT="cb_k0 cb_k0q cb_k5 cb_k5q \
ccv3_m1_diffkv ccv3_m1_q \
ccv3_m2_diffkv ccv3_m2_q \
ccv3_m3_diffkv ccv3_m3_q \
ccv3_m4_diffkv ccv3_m4_q"
METHODS="${METHODS:-$METHODS_DEFAULT}"

# OOM-safety grouping: each V3 method (ccv3_*) carries its own
# multi-version KV-chunk cache; running multiple V3 variants in the
# same Python process risks OOM. So each V3 method is always its own
# invocation.
#
# For non-V3 (stateless cb/fr) methods, two policies (env-driven):
#   BUNDLE_NON_V3=0 (default)  — each cb/fr method as its own group
#                                too. More units, finer load balance,
#                                better device utilisation. Cost: baseline
#                                runs in every group's process (the
#                                duplicate baseline compute is parallel
#                                so it doesn't affect wall clock; see
#                                aggregate_groups.py — first-writer
#                                wins on baseline collision).
#   BUNDLE_NON_V3=1            — legacy: bundle all stateless methods
#                                into one process (saves a few model
#                                loads, but the bundle becomes the
#                                long-pole that idles other slots).
BUNDLE_NON_V3="${BUNDLE_NON_V3:-0}"
METHOD_GROUPS=""              # ";"-separated; each group is a
                              # space-separated method list.
if [ "$BUNDLE_NON_V3" = "1" ]; then
    NON_V3=""
    for m in $METHODS; do
        case "$m" in
            ccv3_*) METHOD_GROUPS="${METHOD_GROUPS};${m}" ;;
            *)      NON_V3="${NON_V3} ${m}" ;;
        esac
    done
    NON_V3="$(echo "$NON_V3" | xargs)"
    [ -n "$NON_V3" ] && METHOD_GROUPS="${NON_V3}${METHOD_GROUPS}"
else
    for m in $METHODS; do
        METHOD_GROUPS="${METHOD_GROUPS};${m}"
    done
fi
METHOD_GROUPS="$(echo "$METHOD_GROUPS" | sed 's/^;//;s/;$//;s/;;*/;/g')"

BOXOFFICE_ARGS=()
if [ -n "${BOXOFFICE_DIR:-}" ]; then
    BOXOFFICE_ARGS+=(--boxoffice-dir "$BOXOFFICE_DIR")
fi
if [ -n "${DATASET_STEM:-}" ]; then
    BOXOFFICE_ARGS+=(--dataset-stem "$DATASET_STEM")
fi
if [ -n "${EVAL_STEM:-}" ]; then
    BOXOFFICE_ARGS+=(--eval-stem "$EVAL_STEM")
fi
if [ -n "${POOL_DIR:-}" ]; then
    BOXOFFICE_ARGS+=(--pool-dir "$POOL_DIR")
fi
if [ "${EVAL_ONLY:-0}" = "1" ]; then
    BOXOFFICE_ARGS+=(--eval-only)
fi
# OUT_DIR is the per-seed result dir. We want the same value for the
# `output/` log paths inside this script too — override the default.
RUNNER_OUT_DIR="${OUT_DIR:-output}"
mkdir -p "$RUNNER_OUT_DIR"

if ! [[ "$SHARDS" =~ ^[0-9]+$ ]] || [ "$SHARDS" -lt 1 ]; then
    echo "SHARDS must be a positive integer (got: $SHARDS)" >&2; exit 2
fi

# ── pre-flight: kill leftover run_boxoffice.py from any prior aborted run ─
SCRIPT_PATH="$HERE/run_boxoffice.py"
if pgrep -f "python .*${SCRIPT_PATH}" > /dev/null 2>&1; then
    echo "==> killing leftover run_boxoffice.py instances"
    pkill -f "python .*${SCRIPT_PATH}" || true
    sleep 1
fi

# ── ctrl-c propagation: kill all spawned python on signal ────────────
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

# Build per-(device, slot) job lists. Each entry is one
# (MODEL|SEED|SHARD|GROUP_IDX|GROUP_METHODS) unit — one Python
# invocation. We round-robin (cell × method-group) across devices so all
# slots stay busy even when there are more groups than cells. Each
# unit independently writes/merges into the per-seed JSON; V3
# statefulness is preserved because each V3 method group is a single
# invocation that processes all warmup→eval queries in order.
declare -A SLOT_QUEUE
for n in "${DEVICE_ARR[@]}"; do
    for s in $(seq 0 $((SLOTS_PER_DEVICE - 1))); do
        SLOT_QUEUE["$n:$s"]=""
    done
done

# Enumerate groups into an array so we can index by GROUP_IDX in the
# worker (avoids re-parsing the ;-separated METHOD_GROUPS string).
GROUP_ARR=()
old_IFS="$IFS"; IFS=';'
for grp in $METHOD_GROUPS; do
    IFS="$old_IFS"
    [ -z "$grp" ] && continue
    GROUP_ARR+=("$grp")
    IFS=';'
done
IFS="$old_IFS"
N_GROUPS_HUMAN="${#GROUP_ARR[@]}"

i=0
for MODEL in $MODELS; do
    for SEED in $SEEDS; do
        for SH in $(seq 0 $((SHARDS - 1))); do
            for gi in $(seq 0 $((N_GROUPS_HUMAN - 1))); do
                slot=$((i % TOTAL_SLOTS))
                npu_idx=$((slot / SLOTS_PER_DEVICE))
                sub=$((slot % SLOTS_PER_DEVICE))
                DEVICE_ID="${DEVICE_ARR[$npu_idx]}"
                SLOT_QUEUE["$DEVICE_ID:$sub"]+="${MODEL}|${SEED}|${SH}|${gi}|${GROUP_ARR[$gi]}"$'\n'
                i=$((i + 1))
            done
        done
    done
done

echo "Plan ($i (cell × group) units across $N_DEVICES device(s) × $SLOTS_PER_DEVICE slot(s) "
echo "      = $TOTAL_SLOTS workers; SHARDS=$SHARDS):"
echo "  $N_GROUPS_HUMAN method groups per cell (each its own python invocation,"
echo "  --merge'd into the same per-seed JSON; V3 caches stateful within each"
echo "  group's invocation across the warmup→eval queries):"
for gi in $(seq 0 $((N_GROUPS_HUMAN - 1))); do
    echo "    group $((gi + 1)):  ${GROUP_ARR[$gi]}"
done
for n in "${DEVICE_ARR[@]}"; do
    for s in $(seq 0 $((SLOTS_PER_DEVICE - 1))); do
        n_units="$(echo "${SLOT_QUEUE["$n:$s"]}" | grep -c .)"
        echo "  device=$n slot=$s  units=$n_units"
    done
done

# Build a filesystem-safe tag for each method group. Single-method
# groups (V3 entries) get the method name verbatim. Multi-method
# groups (the stateless cb/fr bundle) get joined with '__'.
group_tag_for() {
    local methods="$1"
    local n_methods
    n_methods="$(echo "$methods" | wc -w)"
    if [ "$n_methods" -eq 1 ]; then
        # Single method (V3 case) — use it verbatim.
        echo "$methods"
    else
        # Multi-method (stateless cb/fr bundle) — collapse with '__'.
        echo "$methods" | tr ' ' '_' | sed 's/__\+/__/g; s/^_*//; s/_*$//'
    fi
}

worker() {
    local device_id=$1
    local slot=$2
    local queue=$3
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        # Each unit is: MODEL|SEED|SHARD|GROUP_IDX|GROUP_METHODS.
        # GROUP_METHODS may contain spaces, so use cut -f5- (everything
        # after the 4th '|'). The other fields are pipe-free.
        local model seed sh group_idx group_methods
        model="$(echo "$line"        | cut -d'|' -f1)"
        seed="$(echo  "$line"        | cut -d'|' -f2)"
        sh="$(echo    "$line"        | cut -d'|' -f3)"
        group_idx="$(echo "$line"    | cut -d'|' -f4)"
        group_methods="$(echo "$line" | cut -d'|' -f5-)"

        local mshort="$(basename "$model")"
        local g_tag
        g_tag="$(group_tag_for "$group_methods")"
        local file_tag="s${seed}_${mshort}.${g_tag}"
        [ "$SHARDS" -gt 1 ] && file_tag="${file_tag}.shard${sh}of${SHARDS}"
        local log="$RUNNER_OUT_DIR/boxoffice_${file_tag}.log"
        local human_g=$((group_idx + 1))

        local -a grp_methods
        read -r -a grp_methods <<< "$group_methods"

        echo "[device $device_id slot $slot] start $mshort/s$seed shard=$sh " \
             "group=$human_g/$N_GROUPS_HUMAN tag=$g_tag  log=$log"

        # Each (cell × group) worker writes to its OWN per-tag output
        # file: boxoffice_s<seed>_<model>.<g_tag>.json. No --merge, no
        # locking — workers can't race because their destination files
        # are unique. The aggregator below combines them post-hoc.
        python run_boxoffice.py --model "$model" --seeds "$seed" \
            --device-id "$device_id" --out-dir "$RUNNER_OUT_DIR" \
            --shard "$sh" --num-shards "$SHARDS" $FORCE_FLAG \
            --output-tag "$g_tag" \
            "${RATIOS_ARGS[@]}" --methods "${grp_methods[@]}" \
            "${BOXOFFICE_ARGS[@]}" \
            >> "$log" 2>&1 \
        && echo "[device $device_id slot $slot] done  $mshort/s$seed shard=$sh group=$human_g/$N_GROUPS_HUMAN tag=$g_tag" \
        || echo "[device $device_id slot $slot] FAIL  $mshort/s$seed shard=$sh group=$human_g/$N_GROUPS_HUMAN tag=$g_tag  (see $log)" >&2
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

# Aggregate the per-(method-group) files into the canonical per-seed
# JSONs that the plotter expects. Idempotent — running twice on the
# same dir is a no-op (the canonical already covers everything).
echo "==> aggregating per-tag files in $RUNNER_OUT_DIR"
python aggregate_groups.py --in-dir "$RUNNER_OUT_DIR" || \
    echo "WARNING: aggregate_groups.py failed; per-tag files left for inspection" >&2

if [ "$SHARDS" -gt 1 ]; then
    echo "==> aggregating shards in $RUNNER_OUT_DIR"
    python aggregate_shards.py --in-dir "$RUNNER_OUT_DIR"
fi
