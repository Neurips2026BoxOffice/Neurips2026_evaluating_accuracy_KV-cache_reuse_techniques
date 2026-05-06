#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

V1_SCRIPTS = Path(__file__).resolve().parents[1] / "v1_helpers"
if str(V1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(V1_SCRIPTS))

from generate_one_vs_all_v1 import DEFAULT_PROMPT_VARIANT, Film, make_row, update_first_seen, write_jsonl  # type: ignore


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_film_map(path: Path) -> Dict[str, Film]:
    out: Dict[str, Film] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            film = Film(
                entity_id=str(row["entity_id"]),
                title=str(row["title"]),
                director=str(row["director"]),
                release_year=int(row["release_year"]),
                starlight_awards=int(row["starlight_awards"]),
                box_office_musd=int(row["box_office_musd"]),
                genre=str(row["genre"]),
                studio=str(row["studio"]),
                country=str(row["country"]),
                runtime_min=int(row["runtime_min"]),
                cast=str(row.get("cast", "")),
                setting_city=str(row.get("setting_city", "")),
                setting_city_population_mil=str(row.get("setting_city_population_mil", "")),
            )
            out[film.entity_id] = film
    return out


def load_chunk_map(path: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    texts: Dict[str, str] = {}
    toks: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            eid = str(row["entity_id"])
            texts[eid] = str(row["text"])
            toks[eid] = None
    return texts, toks


def build_first_seen(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    first_seen: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        update_first_seen(row, first_seen)
    return first_seen


def prefix_sets_by_direction(rows: Sequence[Dict[str, Any]], target_ids: Set[str]) -> Dict[str, Dict[str, Set[Tuple[str, ...]]]]:
    out: Dict[str, Dict[str, Set[Tuple[str, ...]]]] = {
        eid: {"<": set(), ">": set()} for eid in target_ids
    }
    for row in rows:
        ctx = [str(x) for x in row["metadata"]["context_entity_ids"]]
        vals = row["metadata"]["comparison_attribute_values"]
        for idx, eid in enumerate(ctx):
            if eid not in target_ids:
                continue
            prefix = tuple(ctx[:idx])
            if not prefix:
                continue
            target_val = int(vals[eid])
            prefix_vals = [int(vals[x]) for x in prefix]
            if all(v < target_val for v in prefix_vals):
                out[eid]["<"].add(prefix)
            elif all(v > target_val for v in prefix_vals):
                out[eid][">"].add(prefix)
    return out


def choose_prefix(
    *,
    seed: int,
    target_id: str,
    direction: str,
    variant_index: int,
    pool: Sequence[str],
    seen: Set[Tuple[str, ...]],
) -> List[str]:
    if len(pool) < 9:
        raise RuntimeError(f"Need at least 9 predecessors for {target_id} direction {direction}; got {len(pool)}")
    for attempt in range(1000):
        rng = random.Random(f"{seed}:{target_id}:{direction}:{variant_index}:{attempt}")
        selected = rng.sample(list(pool), 9) if len(pool) > 9 else list(pool)
        rng.shuffle(selected)
        tup = tuple(selected)
        if tup not in seen:
            return selected
    raise RuntimeError(f"Could not find fresh prefix for {target_id} direction {direction}")


def apply_common_metadata(row: Dict[str, Any], *, dataset_variant: str, source_tag: str) -> None:
    md = row.setdefault("metadata", {})
    md["dataset_variant"] = dataset_variant
    md["source_tag"] = source_tag
    md["balanced_variant_requirement"] = {"<": 2, ">": 2}


def reindex_full_rows(rows: List[Dict[str, Any]], warmup_count: int) -> None:
    eval_start = warmup_count
    guaranteed_start = warmup_count + 1
    for idx, row in enumerate(rows):
        qid = f"onevsall_query_{idx:04d}"
        row["query_id"] = qid
        row["trace_id"] = qid
        md = row.setdefault("metadata", {})
        is_eval = idx >= eval_start
        md["row_index_0based"] = idx
        md["row_index_1based"] = idx + 1
        md["full_reuse_eval_start_index_0based"] = eval_start
        md["full_reuse_eval_start_query_1based"] = guaranteed_start
        md["full_reuse_eval_row_index_0based"] = idx - eval_start if is_eval else None
        md["full_reuse_eval_row_index_1based"] = idx - eval_start + 1 if is_eval else None
        md["is_full_reuse_failure_eval"] = bool(is_eval)
        md["guaranteed_pollution_start_query"] = guaranteed_start
        md["pollution_mode"] = "guaranteed" if is_eval else "warmup"


def compute_direction_counts(rows: Sequence[Dict[str, Any]], target_ids: Set[str]) -> Dict[str, Dict[str, int]]:
    prefixes = prefix_sets_by_direction(rows, target_ids)
    return {eid: {"<": len(prefixes[eid]["<"]), ">": len(prefixes[eid][">"])} for eid in target_ids}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-full", type=Path, required=True)
    parser.add_argument("--source-eval", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-corpus", type=Path, required=True)
    parser.add_argument("--canonical-corpus", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-tag", type=str, default="filtered_v2_reference")
    parser.add_argument("--out-full", type=Path, required=True)
    parser.add_argument("--out-eval", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-validation", type=Path, required=True)
    args = parser.parse_args()

    source_full = load_jsonl(args.source_full)
    source_eval = load_jsonl(args.source_eval)
    source_manifest = json.loads(args.source_manifest.read_text())
    warmup_count_orig = len(source_full) - len(source_eval)
    original_warmup = copy.deepcopy(source_full[:warmup_count_orig])
    eval_rows = copy.deepcopy(source_eval)
    target_ids = sorted({eid for row in source_eval for eid in row["metadata"]["context_entity_ids"]})
    target_set = set(target_ids)

    film_by_id = load_film_map(args.source_corpus)
    chunk_text_by_id, chunk_tokens_by_id = load_chunk_map(args.canonical_corpus)
    # Pool from which we sample warmup-row prefix chunks. Original code
    # excluded every eval-target chunk (`set(film_by_id) - target_set`),
    # which guaranteed warmup-prefix-cid sequences were *disjoint* from
    # eval-prefix-cid sequences → β-overlap = 0 for every (variant,
    # eval-chunk) pair → V3's match_lowest_cfo collapsed to its
    # tie-breaker (M=1 and M=4 always picked the same physical variant).
    # Including all film ids restores the v3_m4 behaviour where some
    # warmup variants share cids with eval prefixes (~22% pairs had
    # β > 0 in v3_m4) while preserving the directional `<`/`>` and
    # 2/2 balance constraints below. Per-target exclusion of `target_id`
    # itself happens in the smaller_pool / larger_pool comprehensions.
    prefix_pool_ids = sorted(set(film_by_id))

    prompt_variant = str(source_full[0].get("metadata", {}).get("prompt_variant") or DEFAULT_PROMPT_VARIANT)
    matrix_threshold = float(source_full[0].get("metadata", {}).get("matrix_threshold", 0.5))
    dataset_variant = "boxoffice_filtered_v4_balanced_m4"

    first_seen = build_first_seen(original_warmup)
    existing = prefix_sets_by_direction(original_warmup, target_set)

    added_rows: List[Dict[str, Any]] = []
    for target_id in target_ids:
        target = film_by_id[target_id]
        smaller_pool = [eid for eid in prefix_pool_ids
                        if eid != target_id
                        and film_by_id[eid].box_office_musd < target.box_office_musd]
        larger_pool = [eid for eid in prefix_pool_ids
                       if eid != target_id
                       and film_by_id[eid].box_office_musd > target.box_office_musd]
        plan = {
            "<": max(0, 2 - len(existing[target_id]["<"])),
            ">": max(0, 2 - len(existing[target_id][">"])),
        }
        for direction in ("<", ">"):
            pool = smaller_pool if direction == "<" else larger_pool
            for variant_index in range(plan[direction]):
                prefix_ids = choose_prefix(
                    seed=args.seed,
                    target_id=target_id,
                    direction=direction,
                    variant_index=variant_index,
                    pool=pool,
                    seen=existing[target_id][direction],
                )
                context_ids = prefix_ids + [target_id]
                context = [film_by_id[eid] for eid in context_ids]
                rebuilt = make_row(
                    trace_index=warmup_count_orig + len(added_rows),
                    context=context,
                    chunks_by_id=chunk_text_by_id,
                    chunk_tokens_by_id=chunk_tokens_by_id,
                    first_seen=first_seen,
                    stage=f"warmup_v4_balanced_m4_{direction}_{variant_index:02d}",
                    scheduled_cell=None,
                    guaranteed_start=None,
                    matrix_threshold=matrix_threshold,
                    prompt_variant=prompt_variant,
                )
                apply_common_metadata(rebuilt, dataset_variant=dataset_variant, source_tag=args.source_tag)
                md = rebuilt["metadata"]
                md["balanced_variant_target_entity_id"] = target_id
                md["balanced_variant_direction"] = direction
                md["balanced_variant_direction_variant_index"] = variant_index + 1
                md["balanced_variant_prefix_entity_ids"] = prefix_ids
                md["balanced_variant_prefix_size"] = len(prefix_ids)
                md["balanced_variant_builder"] = "build_filtered_v4_balanced"
                added_rows.append(rebuilt)
                existing[target_id][direction].add(tuple(prefix_ids))
                update_first_seen(rebuilt, first_seen)

    full_rows = copy.deepcopy(original_warmup) + added_rows + copy.deepcopy(eval_rows)
    for row in full_rows:
        apply_common_metadata(row, dataset_variant=dataset_variant, source_tag=args.source_tag)
    reindex_full_rows(full_rows, warmup_count_orig + len(added_rows))

    eval_start = warmup_count_orig + len(added_rows)
    eval_out = copy.deepcopy(full_rows[eval_start:])

    counts = compute_direction_counts(full_rows[:eval_start], target_set)
    # Relaxed from `== 2` to `>= 2` after the pool-broadening patch
    # (prefix_pool_ids now allows eval-target chunks as predecessors,
    # so a single augmented warmup row contributes to the direction
    # counts of several cids, often pushing them past the original
    # "exactly 2" target). Having more variants in each direction is
    # additive — more cache slots for V3 to discriminate via β-pick —
    # and doesn't break anything downstream. The intent of v4 was the
    # MINIMUM 2-per-direction balance, not a hard cap.
    bad = {eid: cnt for eid, cnt in counts.items()
           if cnt["<"] < 2 or cnt[">"] < 2}
    if bad:
        raise RuntimeError(f"Balanced variant requirement failed (need ≥2 "
                           f"per direction): {bad}")

    hist = Counter((cnt["<"], cnt[">"]) for cnt in counts.values())
    validation = {
        "dataset_variant": dataset_variant,
        "source_tag": args.source_tag,
        "seed": args.seed,
        "eval_unique_chunks": len(target_ids),
        "warmup_rows_original": warmup_count_orig,
        "warmup_rows_added": len(added_rows),
        "warmup_rows_total": eval_start,
        "exact_direction_count_histogram": {f"{k[0]}/{k[1]}": v for k, v in sorted(hist.items())},
        "per_chunk_counts": counts,
    }

    manifest = copy.deepcopy(source_manifest)
    manifest["dataset"] = dataset_variant
    manifest["source_tag"] = args.source_tag
    manifest["num_rows"] = len(full_rows)
    manifest["warmup_rows"] = eval_start
    manifest["eval_rows"] = len(eval_out)
    manifest["balanced_variant_requirement"] = {"<": 2, ">": 2}
    manifest["balanced_variant_eval_unique_chunks"] = len(target_ids)
    manifest["balanced_variant_warmup_rows_original"] = warmup_count_orig
    manifest["balanced_variant_warmup_rows_added"] = len(added_rows)
    manifest["balanced_variant_exact_direction_count_histogram"] = validation["exact_direction_count_histogram"]

    write_jsonl(args.out_full, full_rows)
    write_jsonl(args.out_eval, eval_out)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    args.out_validation.parent.mkdir(parents=True, exist_ok=True)
    args.out_validation.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
