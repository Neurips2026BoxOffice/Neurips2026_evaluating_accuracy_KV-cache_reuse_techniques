#!/usr/bin/env python3
"""Generate retrieval-profile-balanced one-vs-all box-office datasets."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


V1_SCRIPTS = Path(__file__).resolve().parents[1] / "v1_helpers"
if str(V1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(V1_SCRIPTS))

from generate_one_vs_all_v1 import (  # type: ignore
    CELL_CODES,
    DEFAULT_PROMPT_VARIANT,
    Film,
    build_token_counter,
    cached_conditioning_maps,
    default_source_paths,
    load_films_from_matrix,
    long_movie_chunk,
    make_row,
    status_counts,
    update_first_seen,
    write_jsonl,
)


@dataclass(frozen=True)
class CandidateRow:
    cell: str
    context: Tuple[Film, ...]
    winner: Film
    runner: Film
    margin: int
    value_range: int
    topk_frac_greater: float
    topk_mean_delta: float
    candidate_overlap_frac: float
    candidate_overlap_count: int
    nonwinner_candidate_overlap_count: int
    nonwinner_other_nonwinner_overlap_count: int
    winner_candidate_overlap_count: int
    warm_score: float
    margin_bin: int
    topk_bin: int


def entity_from_text(text: str) -> str:
    match = re.search(r"ENTITY_ID:\s*(FILM-\d+)", text)
    if not match:
        raise ValueError(f"Could not parse ENTITY_ID from {text[:120]!r}")
    return match.group(1)


def box_from_text(text: str) -> int:
    match = re.search(r"BOX_OFFICE_MUSD:\s*(\d+)", text)
    if not match:
        raise ValueError(f"Could not parse BOX_OFFICE_MUSD from {text[:120]!r}")
    return int(match.group(1))


def load_warm_profile(path: Path) -> Dict[str, Dict[str, Any]]:
    profile: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ref_text = str(row.get("reference_context") or "")
            entity_id = entity_from_text(ref_text)
            ref_value = box_from_text(ref_text)
            neighbors = []
            for ctx in row.get("top_k_contexts", []):
                ctx_entity = str(ctx.get("entity_id") or "")
                ctx_value = int(ctx.get("box_office_musd"))
                neighbors.append(
                    {
                        "entity_id": ctx_entity,
                        "box_office_musd": ctx_value,
                        "delta": ctx_value - ref_value,
                        "relation": "greater" if ctx_value > ref_value else "smaller" if ctx_value < ref_value else "equal",
                    }
                )
            profile[entity_id] = {
                "box_office_musd": ref_value,
                "neighbors": neighbors,
                "reference_cid": str(row.get("reference_cid") or ""),
            }
    return profile


def build_interleaved_warmup(
    ordered: Sequence[Film],
    *,
    num_chunks: int,
    target_start_rank: int,
    target_end_rank: int,
    rng: random.Random,
) -> List[Tuple[str, List[Film]]]:
    if num_chunks != 10:
        raise ValueError("v2 currently expects 10 chunks")
    if len(ordered) < target_end_rank + 10:
        raise RuntimeError(f"Need more films for target_end_rank={target_end_rank}; got {len(ordered)}")

    top_support = list(ordered[: num_chunks - 1])
    bottom_support = list(reversed(ordered[-(num_chunks - 1) :]))
    schedule: List[Tuple[str, List[Film]]] = []

    # Introduce support chunks.  Their own statuses are not used for v2 eval.
    schedule.append(("warmup_v2_top_support", top_support + [ordered[num_chunks - 1]]))
    schedule.append(("warmup_v2_bottom_support", bottom_support + [ordered[-num_chunks]]))

    # Interleave status assignment across adjacent rank pairs so `<` and `>`
    # pools have near-identical box-office distributions.
    target_films = list(ordered[target_start_rank:target_end_rank])
    for pair_idx in range(0, len(target_films), 2):
        pair = target_films[pair_idx : pair_idx + 2]
        if len(pair) < 2:
            break
        lt_target, gt_target = pair if (pair_idx // 2) % 2 == 0 else (pair[1], pair[0])
        schedule.append((f"warmup_v2_lt_{lt_target.entity_id}", bottom_support + [lt_target]))
        schedule.append((f"warmup_v2_gt_{gt_target.entity_id}", top_support + [gt_target]))
    return schedule


def row_metrics(context: Sequence[Film], warm_profile: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    values = [film.box_office_musd for film in context]
    candidate_ids = {film.entity_id for film in context}
    winner, runner = sorted(context, key=lambda film: (film.box_office_musd, film.entity_id), reverse=True)[:2]
    all_relations = []
    all_deltas = []
    overlaps = 0
    nonwinner_overlaps = 0
    nonwinner_other_nonwinner_overlaps = 0
    winner_overlaps = 0
    for film in context:
        neighbors = warm_profile[film.entity_id]["neighbors"]
        for item in neighbors:
            all_relations.append(item["relation"])
            all_deltas.append(float(item["delta"]))
            if item["entity_id"] in candidate_ids:
                overlaps += 1
                if film.entity_id == winner.entity_id:
                    winner_overlaps += 1
                else:
                    nonwinner_overlaps += 1
                    if item["entity_id"] != winner.entity_id:
                        nonwinner_other_nonwinner_overlaps += 1
    n = max(1, len(all_relations))
    gt = sum(1 for rel in all_relations if rel == "greater")
    lt = sum(1 for rel in all_relations if rel == "smaller")
    return {
        "winner": winner,
        "runner": runner,
        "margin": winner.box_office_musd - runner.box_office_musd,
        "value_range": max(values) - min(values),
        "topk_frac_greater": gt / n,
        "topk_frac_smaller": lt / n,
        "topk_mean_delta": mean(all_deltas) if all_deltas else 0.0,
        "candidate_overlap_frac": overlaps / n,
        "candidate_overlap_count": overlaps,
        "nonwinner_candidate_overlap_count": nonwinner_overlaps,
        "nonwinner_other_nonwinner_overlap_count": nonwinner_other_nonwinner_overlaps,
        "winner_candidate_overlap_count": winner_overlaps,
    }


def make_candidate_pool(
    *,
    ordered: Sequence[Film],
    first_seen: Dict[str, Dict[str, Any]],
    warm_profile: Dict[str, Dict[str, Any]],
    num_chunks: int,
    min_gap: int,
    per_cell_attempts: int,
    rng: random.Random,
) -> Dict[str, List[CandidateRow]]:
    status_by_id, predecessor_count_by_id = cached_conditioning_maps(first_seen)
    films_by_id = {film.entity_id: film for film in ordered}
    pools = {
        status: [
            films_by_id[eid]
            for eid, value in status_by_id.items()
            if value == status and predecessor_count_by_id.get(eid, 0) >= 9 and eid in warm_profile
        ]
        for status in ["<", ">"]
    }
    by_cell: Dict[str, List[CandidateRow]] = {cell: [] for cell in CELL_CODES}
    seen_keys: Dict[str, set] = {cell: set() for cell in CELL_CODES}

    for cell in CELL_CODES:
        winner_status, nonwinner_status = cell[0], cell[1]
        winner_pool = pools[winner_status]
        nonwinner_pool = pools[nonwinner_status]
        attempts = 0
        while attempts < per_cell_attempts:
            attempts += 1
            winner = rng.choice(winner_pool)
            lower = [
                film
                for film in nonwinner_pool
                if film.entity_id != winner.entity_id and winner.box_office_musd - film.box_office_musd >= min_gap
            ]
            if len(lower) < num_chunks - 1:
                continue
            lower_sorted = sorted(lower, key=lambda film: abs(winner.box_office_musd - film.box_office_musd))
            near_window = lower_sorted[: min(len(lower_sorted), 28)]
            if len(near_window) < num_chunks - 1:
                continue
            nonwinners = rng.sample(near_window, k=num_chunks - 1)
            context = [winner] + nonwinners
            rng.shuffle(context)
            key = tuple(film.entity_id for film in context)
            if key in seen_keys[cell]:
                continue
            seen_keys[cell].add(key)
            metrics = row_metrics(context, warm_profile)
            if metrics["winner"].entity_id != winner.entity_id:
                continue
            margin_bin = min(9, int(metrics["margin"] // 50))
            topk_bin = min(9, int(metrics["topk_frac_greater"] * 10))
            warm_score = (
                abs(metrics["topk_frac_greater"] - 0.5)
                + 0.45 * metrics["candidate_overlap_frac"]
                + min(1.0, abs(metrics["topk_mean_delta"]) / 400.0) * 0.35
            )
            by_cell[cell].append(
                CandidateRow(
                    cell=cell,
                    context=tuple(context),
                    winner=metrics["winner"],
                    runner=metrics["runner"],
                    margin=int(metrics["margin"]),
                    value_range=int(metrics["value_range"]),
                    topk_frac_greater=float(metrics["topk_frac_greater"]),
                    topk_mean_delta=float(metrics["topk_mean_delta"]),
                    candidate_overlap_frac=float(metrics["candidate_overlap_frac"]),
                    candidate_overlap_count=int(metrics["candidate_overlap_count"]),
                    nonwinner_candidate_overlap_count=int(metrics["nonwinner_candidate_overlap_count"]),
                    nonwinner_other_nonwinner_overlap_count=int(metrics["nonwinner_other_nonwinner_overlap_count"]),
                    winner_candidate_overlap_count=int(metrics["winner_candidate_overlap_count"]),
                    warm_score=float(warm_score),
                    margin_bin=margin_bin,
                    topk_bin=topk_bin,
                )
            )
    return by_cell


def select_balanced_rows(
    by_cell: Dict[str, List[CandidateRow]],
    *,
    eval_per_cell: int,
    rng: random.Random,
    selection_mode: str,
) -> List[CandidateRow]:
    # Prefer bucket combinations available in every cell.  This directly
    # matches margin and warm top-k direction before using the soft score.
    buckets_by_cell: Dict[str, Dict[Tuple[int, ...], List[CandidateRow]]] = {}
    for cell, rows in by_cell.items():
        buckets: Dict[Tuple[int, ...], List[CandidateRow]] = defaultdict(list)
        for row in rows:
            if selection_mode == "v2b_overlap":
                buckets[(row.margin_bin, row.topk_bin, row.nonwinner_candidate_overlap_count)].append(row)
            elif selection_mode == "v2b_low_overlap":
                buckets[(row.topk_bin, row.nonwinner_candidate_overlap_count)].append(row)
            else:
                buckets[(row.margin_bin, row.topk_bin)].append(row)
        for bucket_rows in buckets.values():
            rng.shuffle(bucket_rows)
            bucket_rows.sort(
                key=lambda row: (
                    row.warm_score,
                    row.nonwinner_candidate_overlap_count,
                    row.nonwinner_other_nonwinner_overlap_count,
                    row.candidate_overlap_frac,
                    abs(row.topk_frac_greater - 0.5),
                )
            )
        buckets_by_cell[cell] = buckets

    common_buckets = set.intersection(*(set(buckets) for buckets in buckets_by_cell.values()))
    bucket_order = sorted(
        common_buckets,
        key=lambda bucket: (
            bucket[1] if selection_mode == "v2b_low_overlap" else abs(bucket[1] - 5),
            abs(bucket[0] - 5) if selection_mode == "v2b_low_overlap" else abs(bucket[0] - 2),
            abs(bucket[2] - 1) if len(bucket) > 2 else 0,
            bucket[0],
            bucket[1],
            bucket[2] if len(bucket) > 2 else 0,
        ),
    )
    selected: Dict[str, List[CandidateRow]] = {cell: [] for cell in CELL_CODES}
    used_contexts: Dict[str, set] = {cell: set() for cell in CELL_CODES}

    changed = True
    while changed and any(len(selected[cell]) < eval_per_cell for cell in CELL_CODES):
        changed = False
        for bucket in bucket_order:
            if all(len(selected[cell]) >= eval_per_cell for cell in CELL_CODES):
                break
            for cell in CELL_CODES:
                if len(selected[cell]) >= eval_per_cell:
                    continue
                bucket_rows = buckets_by_cell[cell].get(bucket, [])
                while bucket_rows:
                    row = bucket_rows.pop(0)
                    key = tuple(film.entity_id for film in row.context)
                    if key in used_contexts[cell]:
                        continue
                    selected[cell].append(row)
                    used_contexts[cell].add(key)
                    changed = True
                    break

    # Fill any remaining rows with best warm-neutral candidates.
    for cell in CELL_CODES:
        if len(selected[cell]) >= eval_per_cell:
            continue
        existing = {tuple(film.entity_id for film in row.context) for row in selected[cell]}
        fallback = sorted(
            by_cell[cell],
            key=lambda row: (
                row.warm_score,
                row.nonwinner_candidate_overlap_count,
                row.nonwinner_other_nonwinner_overlap_count,
                abs(row.margin - 140),
                row.value_range,
            ),
        )
        for row in fallback:
            key = tuple(film.entity_id for film in row.context)
            if key in existing:
                continue
            selected[cell].append(row)
            existing.add(key)
            if len(selected[cell]) >= eval_per_cell:
                break
        if len(selected[cell]) < eval_per_cell:
            raise RuntimeError(f"Could only select {len(selected[cell])}/{eval_per_cell} rows for cell {cell}")

    out = [row for cell in CELL_CODES for row in selected[cell][:eval_per_cell]]
    rng.shuffle(out)
    return out


def add_v2_metadata(row: Dict[str, Any], cand: CandidateRow, warm_profile: Dict[str, Dict[str, Any]]) -> None:
    md = row["metadata"]
    md["template"] = "one_vs_all_v2"
    md["matrix_scheduler_mode"] = "one_vs_all_v2_balanced_warm_topk"
    md["v2_balance_metrics"] = {
        "winner_runner_margin": cand.margin,
        "candidate_value_range": cand.value_range,
        "warm_topk_frac_greater": cand.topk_frac_greater,
        "warm_topk_mean_delta": cand.topk_mean_delta,
        "warm_topk_candidate_overlap_frac": cand.candidate_overlap_frac,
        "warm_topk_candidate_overlap_count": cand.candidate_overlap_count,
        "warm_topk_nonwinner_candidate_overlap_count": cand.nonwinner_candidate_overlap_count,
        "warm_topk_nonwinner_other_nonwinner_overlap_count": cand.nonwinner_other_nonwinner_overlap_count,
        "warm_topk_winner_candidate_overlap_count": cand.winner_candidate_overlap_count,
        "margin_bin_50musd": cand.margin_bin,
        "topk_frac_greater_bin_decile": cand.topk_bin,
        "warm_score": cand.warm_score,
    }
    per_entity = {}
    candidate_ids = {film.entity_id for film in cand.context}
    for film in cand.context:
        ref = warm_profile[film.entity_id]
        neighbors = ref["neighbors"]
        per_entity[film.entity_id] = {
            "box_office_musd": film.box_office_musd,
            "num_greater": sum(1 for item in neighbors if item["relation"] == "greater"),
            "num_smaller": sum(1 for item in neighbors if item["relation"] == "smaller"),
            "mean_delta": mean(float(item["delta"]) for item in neighbors) if neighbors else 0.0,
            "candidate_overlap_count": sum(1 for item in neighbors if item["entity_id"] in candidate_ids),
            "topk_entity_ids": [item["entity_id"] for item in neighbors],
            "topk_box_office_musd": [item["box_office_musd"] for item in neighbors],
        }
    md["warm_topk_profile"] = {
        "retrieval_mode": "embedding",
        "top_k": 5,
        "profile_source": "qwen3_8B warm fusion_docs_topk5",
        "per_entity": per_entity,
    }


def generate(
    *,
    source_paths: Sequence[Path],
    warm_fusion_docs: Path,
    seed: int,
    num_chunks: int,
    eval_per_cell: int,
    min_boxoffice_gap: int,
    chunk_target_tokens: int,
    tokenizer_path: Optional[Path],
    prompt_variant: str,
    target_start_rank: int,
    target_end_rank: int,
    per_cell_attempts: int,
    selection_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    films = load_films_from_matrix(source_paths)
    count_tokens = build_token_counter(tokenizer_path)
    chunks_by_id: Dict[str, str] = {}
    chunk_tokens_by_id: Dict[str, Optional[int]] = {}
    for film in films:
        chunk, token_count = long_movie_chunk(film, chunk_target_tokens, count_tokens)
        chunks_by_id[film.entity_id] = chunk
        chunk_tokens_by_id[film.entity_id] = token_count

    ordered = sorted(films, key=lambda film: (film.box_office_musd, film.entity_id), reverse=True)
    warm_profile = load_warm_profile(warm_fusion_docs)
    first_seen: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    warmup_schedule = build_interleaved_warmup(
        ordered,
        num_chunks=num_chunks,
        target_start_rank=target_start_rank,
        target_end_rank=target_end_rank,
        rng=rng,
    )
    for stage, context in warmup_schedule:
        row = make_row(
            trace_index=len(rows),
            context=context,
            chunks_by_id=chunks_by_id,
            chunk_tokens_by_id=chunk_tokens_by_id,
            first_seen=first_seen,
            stage=stage,
            scheduled_cell=None,
            guaranteed_start=None,
            matrix_threshold=0.5,
            prompt_variant=prompt_variant,
        )
        rows.append(row)
        update_first_seen(row, first_seen)

    by_cell = make_candidate_pool(
        ordered=ordered,
        first_seen=first_seen,
        warm_profile=warm_profile,
        num_chunks=num_chunks,
        min_gap=min_boxoffice_gap,
        per_cell_attempts=per_cell_attempts,
        rng=random.Random(seed * 17 + 3),
    )
    selected = select_balanced_rows(
        by_cell,
        eval_per_cell=eval_per_cell,
        rng=random.Random(seed * 19 + 5),
        selection_mode=selection_mode,
    )
    guaranteed_start = len(rows) + 1
    for row in rows:
        row["metadata"]["guaranteed_pollution_start_query"] = guaranteed_start

    eval_rows = []
    for cand in selected:
        row = make_row(
            trace_index=len(rows) + len(eval_rows),
            context=cand.context,
            chunks_by_id=chunks_by_id,
            chunk_tokens_by_id=chunk_tokens_by_id,
            first_seen=first_seen,
            stage="guaranteed_eval",
            scheduled_cell=cand.cell,
            guaranteed_start=guaranteed_start,
            matrix_threshold=0.5,
            prompt_variant=prompt_variant,
        )
        if row["metadata"]["computed_matrix_cell"] != cand.cell:
            raise RuntimeError(f"Computed cell mismatch: expected {cand.cell}, got {row['metadata']['computed_matrix_cell']}")
        add_v2_metadata(row, cand, warm_profile)
        eval_rows.append(row)

    rows.extend(eval_rows)
    start_index_0based = guaranteed_start - 1
    for idx, row in enumerate(rows):
        md = row.setdefault("metadata", {})
        is_eval = idx >= start_index_0based
        md["row_index_0based"] = idx
        md["row_index_1based"] = idx + 1
        md["full_reuse_eval_start_index_0based"] = start_index_0based
        md["full_reuse_eval_start_query_1based"] = guaranteed_start
        md["full_reuse_eval_row_index_0based"] = idx - start_index_0based if is_eval else None
        md["full_reuse_eval_row_index_1based"] = idx - start_index_0based + 1 if is_eval else None
        md["is_full_reuse_failure_eval"] = bool(is_eval)
        if is_eval:
            md["pollution_mode"] = "guaranteed"

    status_by_id, predecessor_count_by_id = cached_conditioning_maps(first_seen)
    eval_by_cell = Counter(row["metadata"]["computed_matrix_cell"] for row in eval_rows)
    margin_by_cell = defaultdict(list)
    topk_by_cell = defaultdict(list)
    overlap_by_cell = defaultdict(list)
    overlap_count_by_cell = defaultdict(list)
    nonwinner_overlap_count_by_cell = defaultdict(list)
    nonwinner_other_overlap_count_by_cell = defaultdict(list)
    winner_overlap_count_by_cell = defaultdict(list)
    winners_by_cell = defaultdict(set)
    for row in eval_rows:
        cell = row["metadata"]["computed_matrix_cell"]
        bal = row["metadata"]["v2_balance_metrics"]
        margin_by_cell[cell].append(bal["winner_runner_margin"])
        topk_by_cell[cell].append(bal["warm_topk_frac_greater"])
        overlap_by_cell[cell].append(bal["warm_topk_candidate_overlap_frac"])
        overlap_count_by_cell[cell].append(bal["warm_topk_candidate_overlap_count"])
        nonwinner_overlap_count_by_cell[cell].append(bal["warm_topk_nonwinner_candidate_overlap_count"])
        nonwinner_other_overlap_count_by_cell[cell].append(bal["warm_topk_nonwinner_other_nonwinner_overlap_count"])
        winner_overlap_count_by_cell[cell].append(bal["warm_topk_winner_candidate_overlap_count"])
        winners_by_cell[cell].add(row["metadata"]["winner_entity_id"])

    manifest = {
        "seed": seed,
        "dataset": "one_vs_all_v2",
        "num_rows": len(rows),
        "warmup_rows": guaranteed_start - 1,
        "eval_rows": len(eval_rows),
        "eval_per_cell": eval_per_cell,
        "cell_counts": dict(eval_by_cell),
        "status_counts": status_counts(status_by_id),
        "predecessor_count_by_id": predecessor_count_by_id,
        "warm_fusion_docs": str(warm_fusion_docs),
        "target_start_rank": target_start_rank,
        "target_end_rank": target_end_rank,
        "per_cell_attempts": per_cell_attempts,
        "selection_mode": selection_mode,
        "by_cell": {
            cell: {
                "mean_margin": mean(margin_by_cell[cell]) if margin_by_cell[cell] else None,
                "mean_topk_frac_greater": mean(topk_by_cell[cell]) if topk_by_cell[cell] else None,
                "mean_candidate_overlap_frac": mean(overlap_by_cell[cell]) if overlap_by_cell[cell] else None,
                "mean_candidate_overlap_count": mean(overlap_count_by_cell[cell]) if overlap_count_by_cell[cell] else None,
                "mean_nonwinner_candidate_overlap_count": mean(nonwinner_overlap_count_by_cell[cell])
                if nonwinner_overlap_count_by_cell[cell]
                else None,
                "mean_nonwinner_other_nonwinner_overlap_count": mean(nonwinner_other_overlap_count_by_cell[cell])
                if nonwinner_other_overlap_count_by_cell[cell]
                else None,
                "mean_winner_candidate_overlap_count": mean(winner_overlap_count_by_cell[cell])
                if winner_overlap_count_by_cell[cell]
                else None,
                "unique_winners": len(winners_by_cell[cell]),
                "pool_candidates": len(by_cell[cell]),
            }
            for cell in CELL_CODES
        },
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--warm-fusion-docs", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-chunks", type=int, default=10)
    parser.add_argument("--eval-per-cell", type=int, default=40)
    parser.add_argument("--min-boxoffice-gap", type=int, default=40)
    parser.add_argument("--chunk-target-tokens", type=int, default=512)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--prompt-variant", default=DEFAULT_PROMPT_VARIANT)
    parser.add_argument("--target-start-rank", type=int, default=10)
    parser.add_argument("--target-end-rank", type=int, default=70)
    parser.add_argument("--per-cell-attempts", type=int, default=20000)
    parser.add_argument(
        "--selection-mode",
        choices=["v2", "v2b_overlap", "v2b_low_overlap"],
        default="v2",
        help=(
            "v2 preserves the original selector; v2b_overlap also matches nonwinner warm-overlap buckets; "
            "v2b_low_overlap matches nonwinner warm-overlap buckets without hard margin buckets."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, default=None)
    args = parser.parse_args()

    rows, manifest = generate(
        source_paths=list(args.source_jsonl) or default_source_paths(),
        warm_fusion_docs=args.warm_fusion_docs,
        seed=args.seed,
        num_chunks=args.num_chunks,
        eval_per_cell=args.eval_per_cell,
        min_boxoffice_gap=args.min_boxoffice_gap,
        chunk_target_tokens=args.chunk_target_tokens,
        tokenizer_path=args.tokenizer_path,
        prompt_variant=args.prompt_variant,
        target_start_rank=args.target_start_rank,
        target_end_rank=args.target_end_rank,
        per_cell_attempts=args.per_cell_attempts,
        selection_mode=args.selection_mode,
    )
    write_jsonl(args.out, rows)
    manifest_path = args.manifest_out or args.out.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.out}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
