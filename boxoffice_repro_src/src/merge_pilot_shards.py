#!/usr/bin/env python3
"""Merge per-shard pilot_eval outputs into one canonical JSON.

Per-shard files have shape:
    {"summary": {...}, "model_constants": {...}, "rows": [...], "timing_rows": [...]}

`build_joint_filtered_dataset.py` only reads `data["rows"]` and indexes by
`row["trace_id"]`, so we just concatenate `rows` (deduping by trace_id —
first writer wins) and recompute mean stats.

Usage:
    python merge_pilot_shards.py --out merged.json shard0.json shard1.json ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("shards", nargs="+", type=Path,
                    help="per-shard JSONs in any order")
    args = ap.parse_args()

    seen: dict[str, dict] = {}
    summary = None
    model_constants = None
    timing_rows: list = []

    for p in args.shards:
        d = json.loads(p.read_text(encoding="utf-8"))
        if summary is None:
            summary = dict(d.get("summary") or {})
        if model_constants is None:
            model_constants = d.get("model_constants")
        for r in d.get("rows") or []:
            tid = str(r.get("trace_id"))
            if tid and tid not in seen:
                seen[tid] = r
        for tr in d.get("timing_rows") or []:
            timing_rows.append(tr)

    rows = list(seen.values())
    n = len(rows) or 1
    if summary is None:
        summary = {}
    summary.update({
        "exact_match":      sum(x["score"]["exact_match"]      for x in rows) / n,
        "field_match":      sum(x["score"]["field_match"]      for x in rows) / n,
        "parse_error_rate": sum(x["score"]["parse_error"]      for x in rows) / n,
        "answer_f1":        sum(float(x["score"].get("answer_f1", 0.0)) for x in rows) / n,
        "merged_from":      [p.name for p in args.shards],
        "n_rows":           len(rows),
    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "summary":         summary,
        "model_constants": model_constants,
        "rows":            rows,
        "timing_rows":     timing_rows,
    }, indent=2), encoding="utf-8")
    print(f"merged {len(args.shards)} shards → {args.out}  ({len(rows)} unique rows)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
