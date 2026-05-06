#!/usr/bin/env python3
"""exp/boxoffice — merge per-shard outputs into the canonical
boxoffice_s<seed>_<model>.json files.

Each worker writes
    boxoffice_s<seed>_<model>.shard<i>of<K>.json
when --num-shards > 1. This script discovers all such shard files,
groups by (seed, model), and concatenates per-query results back into
one merged file (sorted by orig_index for determinism). The shard
files are kept on disk; pass --delete-shards to clean them up.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SHARD_RE = re.compile(
    r"^boxoffice_s(?P<seed>\d+)_(?P<model>.+?)\.shard(?P<i>\d+)of(?P<k>\d+)\.json$"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--delete-shards", action="store_true",
                    help="rm the per-shard files after a successful merge.")
    args = ap.parse_args()

    if not args.in_dir.is_dir():
        print(f"missing dir: {args.in_dir}", file=sys.stderr); sys.exit(2)

    by_pair: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for p in sorted(args.in_dir.iterdir()):
        m = SHARD_RE.match(p.name)
        if not m: continue
        by_pair[(m.group("seed"), m.group("model"))].append(p)
    if not by_pair:
        print(f"no shard files under {args.in_dir}")
        return

    for (seed, model), shards in sorted(by_pair.items()):
        shards.sort()
        merged_results: list[dict] = []
        cfg = None
        for sp in shards:
            payload = json.loads(sp.read_text())
            cfg = cfg or dict(payload.get("config", {}))
            merged_results.extend(payload.get("per_query", []))
        merged_results.sort(key=lambda r: r.get("orig_index", -1))
        out = args.in_dir / f"boxoffice_s{seed}_{model}.json"
        if cfg is not None:
            cfg.pop("shard", None); cfg.pop("num_shards", None)
        out.write_text(json.dumps({
            "config":    cfg or {},
            "per_query": merged_results,
        }, ensure_ascii=False))
        print(f"merged {len(shards)} shards → {out.name}  "
              f"({len(merged_results)} rows, "
              f"{out.stat().st_size/1e6:.1f} MB)")
        if args.delete_shards:
            for sp in shards: sp.unlink()
            print(f"  cleaned {len(shards)} shard files")


if __name__ == "__main__":
    main()
