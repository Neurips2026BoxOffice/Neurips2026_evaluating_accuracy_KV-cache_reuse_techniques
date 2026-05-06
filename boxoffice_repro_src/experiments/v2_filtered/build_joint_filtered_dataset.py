#!/usr/bin/env python3
"""Build a shared filtered dataset using the intersection of model-conditioned survivors."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


CELL_ORDER = ["<<", "<>", "><", ">>"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_result_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["trace_id"]): row for row in data["rows"]}


def refresh_indices(rows: List[Dict[str, Any]], warmup_count: int) -> None:
    start_index_0based = warmup_count
    guaranteed_start = warmup_count + 1
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


def parse_model_name_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_path_map(entries: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in entries:
        model, path = item.split("=", 1)
        out[model.strip()] = Path(path.strip())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-full", type=Path, required=True)
    parser.add_argument("--source-eval", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--models", required=True, help="Comma-separated model labels")
    parser.add_argument("--baseline-result", action="append", default=[], help="model=/path/to/baseline.json")
    parser.add_argument("--question-only-result", action="append", default=[], help="model=/path/to/question_only.json")
    parser.add_argument("--keep-per-cell", type=int, default=40)
    # Filter criteria.
    #
    # Full side: F1 — matches paper2's plot, which filters bars to rows
    # with baseline.f1 == 1.0. Per CLAUDE.md, `baseline ≡ BaselineNosep`.
    #
    # Qonly side: EM, NOT F1. CLAUDE.md's `grounded ≡ no_ctx_f1 < 0.2`
    # is for natural-language QA where wrong predictions tokenize to
    # disjoint words. Boxoffice answers are structured `FILM-XXXX`
    # strings, so even a totally wrong film (e.g. FILM-2024 vs gold
    # FILM-1038) shares the "FILM" token → token-level F1 ≈ 0.4-0.6
    # for every wrong answer → F1<0.2 is unsatisfiable. EM correctly
    # identifies "model got the wrong film" regardless.
    parser.add_argument("--baseline-f1-min", type=float, default=1.0,
                        help="keep row only if baseline.answer_f1 >= this "
                             "(default 1.0 — strictly answerable with full ctx)")
    parser.add_argument("--qonly-em-must-be-zero", type=int, default=1,
                        help="if 1, keep row only if qonly.exact_match == 0 "
                             "(default — context is genuinely needed; the "
                             "right grounding metric for FILM-XXXX answers). "
                             "Set to 0 to disable the qonly filter entirely.")
    parser.add_argument("--qonly-f1-max", type=float, default=None,
                        help="alternative qonly criterion: keep row only if "
                             "qonly.answer_f1 < this. NOT recommended for "
                             "boxoffice — token-level F1 floors at ~0.4 due "
                             "to shared 'FILM' prefix, making this filter "
                             "vacuous. Defaults to disabled (None).")
    parser.add_argument("--out-full", type=Path, required=True)
    parser.add_argument("--out-eval", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    args = parser.parse_args()

    models = parse_model_name_list(args.models)
    baseline_paths = parse_path_map(args.baseline_result)
    qonly_paths = parse_path_map(args.question_only_result)

    source_full = load_jsonl(args.source_full)
    source_eval = load_jsonl(args.source_eval)
    warmup_count = len(source_full) - len(source_eval)
    warmup_rows = json.loads(json.dumps(source_full[:warmup_count]))

    per_model_keep: Dict[str, set[str]] = {}
    cell_by_trace: Dict[str, str] = {}
    for model in models:
        full_rows = load_result_rows(baseline_paths[model])
        qonly_rows = load_result_rows(qonly_paths[model])
        keep = set()
        for trace, row in full_rows.items():
            cell_by_trace[trace] = row["metadata"]["computed_matrix_cell"]
            full_f1   = float(row["score"].get("answer_f1", 0.0))
            qonly     = qonly_rows[trace]["score"]
            qonly_em  = int(qonly.get("exact_match", 0))
            qonly_f1  = float(qonly.get("answer_f1", 0.0))

            if full_f1 < args.baseline_f1_min:
                continue
            if args.qonly_em_must_be_zero and qonly_em != 0:
                continue
            if args.qonly_f1_max is not None and qonly_f1 >= args.qonly_f1_max:
                continue
            keep.add(trace)
        per_model_keep[model] = keep

    shared = set.intersection(*(per_model_keep[m] for m in models))
    survivors_by_cell: Dict[str, List[Dict[str, Any]]] = {cell: [] for cell in CELL_ORDER}
    for row in source_eval:
        trace = str(row["trace_id"])
        if trace not in shared:
            continue
        cell = str(row["metadata"]["computed_matrix_cell"])
        new_row = json.loads(json.dumps(row))
        new_row["metadata"]["joint_model_filter"] = {
            "models": models,
            "full_context_exact_match_required": 1,
            "question_only_exact_match_required": 0,
        }
        survivors_by_cell[cell].append(new_row)

    survivor_counts = {cell: len(rows) for cell, rows in survivors_by_cell.items()}
    for cell, count in survivor_counts.items():
        if count < args.keep_per_cell:
            raise RuntimeError(f"Need {args.keep_per_cell} rows for cell {cell}, but only {count} survive jointly")

    selected_eval: List[Dict[str, Any]] = []
    for cell in CELL_ORDER:
        selected_eval.extend(survivors_by_cell[cell][: args.keep_per_cell])

    full_rows = warmup_rows + selected_eval
    refresh_indices(full_rows, warmup_count)
    refresh_indices(selected_eval, 0)

    write_jsonl(args.out_full, full_rows)
    write_jsonl(args.out_eval, selected_eval)

    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    manifest["dataset"] = "one_vs_all_v2_release_plain_catalog_filtered_v2"
    manifest["joint_model_filter"] = {
        "models": models,
        "baseline_f1_min":          args.baseline_f1_min,
        "qonly_em_must_be_zero":    bool(args.qonly_em_must_be_zero),
        "qonly_f1_max":             args.qonly_f1_max,
        "keep_per_cell": args.keep_per_cell,
        "survivor_counts_before_downselect": survivor_counts,
        "source_full": str(args.source_full),
        "source_eval": str(args.source_eval),
        "baseline_results": {m: str(baseline_paths[m]) for m in models},
        "question_only_results": {m: str(qonly_paths[m]) for m in models},
    }
    manifest["eval_per_cell"] = args.keep_per_cell
    manifest["eval_rows"] = args.keep_per_cell * 4
    manifest["num_rows"] = warmup_count + args.keep_per_cell * 4
    manifest["warmup_rows"] = warmup_count
    manifest["cell_counts"] = {cell: args.keep_per_cell for cell in CELL_ORDER}
    args.out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "models": models,
        "selected_counts": Counter(row["metadata"]["computed_matrix_cell"] for row in selected_eval),
        "survivor_counts_before_downselect": survivor_counts,
        "warmup_rows": warmup_count,
    }, indent=2))


if __name__ == "__main__":
    main()
