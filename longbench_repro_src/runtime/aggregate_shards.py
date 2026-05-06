#!/usr/bin/env python3
"""
exp/intro — merge per-shard outputs into a single intro_<ds>_<model>.json.

When run_intro.py is invoked with --shard i --num-shards K it writes
    intro_<ds>_<model>.shard<i>of<K>.json
containing only the queries this worker owned (i %% K == i).

This script scans an --in-dir for those files, groups them by (ds, model),
verifies all K shards are present and consistent, then writes the merged
    intro_<ds>_<model>.json
sorted by orig_index so downstream consumers see the queries in the same
order as if --num-shards=1 had been used.

Usage:
    python aggregate_shards.py                 # in/out = ./output
    python aggregate_shards.py --in-dir DIR    # both
    python aggregate_shards.py --in-dir A --out-dir B
    python aggregate_shards.py --keep-shards   # don't delete shard files
    python aggregate_shards.py --force         # overwrite existing merged
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DIR = SCRIPT_DIR / "output"

PAT = re.compile(r"^intro_(?P<ds>.+)_(?P<model>[^_]+)\.shard(?P<shard>\d+)of(?P<total>\d+)\.json$")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir",  type=Path, default=DEFAULT_DIR)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to --in-dir")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing merged file")
    ap.add_argument("--keep-shards", action="store_true",
                    help="don't delete the per-shard files after a successful merge")
    ap.add_argument("--require-complete", action="store_true",
                    help="fail (exit 1) if any group has missing shards")
    args = ap.parse_args()
    out_dir = args.out_dir or args.in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"total": None, "shards": {}})
    for path in sorted(args.in_dir.glob("intro_*.shard*of*.json")):
        m = PAT.match(path.name)
        if not m:
            continue
        ds, model = m["ds"], m["model"]
        shard = int(m["shard"]); total = int(m["total"])
        g = groups[(ds, model)]
        if g["total"] is None:
            g["total"] = total
        elif g["total"] != total:
            print(f"  [{ds}/{model}] inconsistent num_shards "
                  f"({g['total']} vs {total} in {path.name}); skipping group",
                  file=sys.stderr)
            g["total"] = -1  # poison
        if shard in g["shards"]:
            print(f"  [{ds}/{model}] duplicate shard {shard}: "
                  f"{g['shards'][shard].name} vs {path.name}", file=sys.stderr)
        g["shards"][shard] = path

    if not groups:
        print(f"no intro_*.shard*of*.json under {args.in_dir}")
        return

    any_incomplete = False
    for (ds, model), g in sorted(groups.items()):
        K = g["total"]
        if K is None or K < 1:
            print(f"  [{ds}/{model}] poisoned, skipping")
            continue
        merged_path = out_dir / f"intro_{ds}_{model}.json"
        if merged_path.exists() and not args.force:
            print(f"  [skip] {merged_path} (already exists; use --force to overwrite)")
            continue

        present = sorted(g["shards"].keys())
        missing = [i for i in range(K) if i not in g["shards"]]
        if missing:
            any_incomplete = True
            print(f"  [{ds}/{model}] have shards {present} of {K}, "
                  f"missing {missing} — skipping", file=sys.stderr)
            continue

        cfg = None
        per_q: list[dict] = []
        elapsed_total = 0.0
        for i in range(K):
            blob = json.loads(g["shards"][i].read_text())
            if cfg is None:
                cfg = dict(blob.get("config", {}))
            elapsed_total += float(blob.get("config", {}).get("elapsed_s", 0.0))
            per_q.extend(blob.get("per_query", []))

        per_q.sort(key=lambda q: q.get("orig_index", 0))
        for q in per_q:
            q.pop("orig_index", None)

        for k in ("shard", "num_shards", "n_in_shard"):
            cfg.pop(k, None)
        cfg["n_queries"] = len(per_q)
        cfg["elapsed_s"] = elapsed_total

        merged_path.write_text(json.dumps(
            {"config": cfg, "per_query": per_q}, indent=2, ensure_ascii=False))
        print(f"  wrote {merged_path}  ({len(per_q)} queries from {K} shards)")

        if not args.keep_shards:
            for path in g["shards"].values():
                path.unlink()
            print(f"    removed {K} shard files")

    if any_incomplete and args.require_complete:
        sys.exit(1)


if __name__ == "__main__":
    main()
