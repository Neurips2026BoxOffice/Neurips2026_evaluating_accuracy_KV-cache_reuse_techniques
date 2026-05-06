#!/usr/bin/env python3
"""
Aggregate every boxoffice JSON in a directory into the canonical
per-(seed, model) file.

Inputs in `--in-dir`:
    boxoffice_s<seed>_<model>.<tag>.json                 per-method-group
    boxoffice_s<seed>_<model>.<tag>.shard<i>of<K>.json   per-method-group × shard
    boxoffice_s<seed>_<model>.json                       canonical
    boxoffice_s<seed>_<model>.shard<i>of<K>.json         legacy shards

We don't try to parse model names out of filenames (they contain dots
— `llama3.1-8BI` would mis-split). Instead we open each JSON and use
`config.seed` + `config.model_short` to bucket them per (seed, model).

Output (per cell):
    boxoffice_s<seed>_<model>.json   merged canonical file. per_query
                                     rows are unioned by `query_id`;
                                     per-row keys are unioned with
                                     first-writer-wins on collision.
                                     config.methods + config.recomp_ratios
                                     are unions across all parts.

Idempotent: rerunning leaves the canonical untouched if it already
covers everything. Use `--force` to rewrite anyway.

Usage:
    python boxoffice/aggregate_groups.py --in-dir boxoffice/output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--force", action="store_true",
                    help="overwrite the canonical file even if it already "
                         "covers everything in the parts.")
    ap.add_argument("--delete-tag-files", action="store_true",
                    help="after a successful merge, delete the per-tag / "
                         "per-shard part files so only the canonical remains.")
    args = ap.parse_args()

    if not args.in_dir.is_dir():
        raise SystemExit(f"missing in-dir: {args.in_dir}")

    # Bucket every boxoffice_*.json by (seed, model_short) using config.
    by_cell: dict[tuple[str, str], list[tuple[Path, dict]]] = defaultdict(list)
    for p in sorted(args.in_dir.glob("boxoffice_s*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception as exc:
            print(f"  [skip-bad] {p.name}: {exc}", file=sys.stderr)
            continue
        cfg = d.get("config") or {}
        seed = cfg.get("seed")
        model = cfg.get("model_short")
        if seed is None or model is None:
            print(f"  [skip-no-cfg] {p.name}", file=sys.stderr)
            continue
        by_cell[(str(seed), model)].append((p, d))

    if not by_cell:
        print(f"no boxoffice_*.json under {args.in_dir} with usable config")
        return

    n_written = 0
    for (seed, model), parts in sorted(by_cell.items()):
        canonical = args.in_dir / f"boxoffice_s{seed}_{model}.json"

        # Merge per_query and config across parts. Iterate in part-name
        # order so the result is deterministic.
        parts_sorted = sorted(parts, key=lambda pd: pd[0].name)
        merged_pq_by_qid: dict = {}
        order: list = []  # preserve first-seen qid order
        merged_cfg: dict = {}
        union_methods: set = set()
        union_ratios: set = set()
        part_names: list[str] = []

        for path, d in parts_sorted:
            cfg = d.get("config") or {}
            for r in d.get("per_query") or []:
                qid = r.get("query_id")
                if qid is None:
                    continue
                if qid not in merged_pq_by_qid:
                    order.append(qid)
                    merged_pq_by_qid[qid] = dict(r)  # shallow copy
                else:
                    # Existing keys win on collision (preserves first
                    # writer's data; an interrupted retry that overlaps
                    # in methods doesn't clobber the original numbers).
                    for k, v in r.items():
                        merged_pq_by_qid[qid].setdefault(k, v)
            for k, v in cfg.items():
                if k in ("methods", "recomp_ratios", "shard", "num_shards",
                         "aggregated_from"):
                    continue
                merged_cfg[k] = v
            for m in cfg.get("methods") or []:
                union_methods.add(m)
            for r in cfg.get("recomp_ratios") or []:
                union_ratios.add(float(r))
            part_names.append(path.name)

        merged_cfg["methods"]       = sorted(union_methods)
        merged_cfg["recomp_ratios"] = sorted(union_ratios)
        merged_cfg["aggregated_from"] = part_names
        merged = {
            "config":    merged_cfg,
            "per_query": [merged_pq_by_qid[qid] for qid in order],
        }

        # Skip if the canonical already matches.
        canonical_part_match = (canonical in [pp[0] for pp in parts_sorted])
        non_canonical_parts = [
            p for (p, _) in parts_sorted if p != canonical
        ]

        if (canonical.exists() and not args.force
                and canonical in [pp[0] for pp in parts_sorted]):
            try:
                old = json.loads(canonical.read_text())
                old_cfg = old.get("config") or {}
                if (set(old_cfg.get("methods") or []) >= union_methods
                    and {float(r) for r in (old_cfg.get("recomp_ratios") or [])} >= union_ratios
                    and len(old.get("per_query") or []) >= len(order)):
                    print(f"  [skip] {canonical.name} (already covers "
                          f"{len(union_methods)} methods × "
                          f"{len(union_ratios)} ratios)")
                    if args.delete_tag_files:
                        for p in non_canonical_parts:
                            try:
                                p.unlink()
                            except OSError:
                                pass
                    continue
            except Exception:
                pass

        # Atomic write.
        tmp = canonical.with_suffix(canonical.suffix + ".tmp")
        tmp.write_text(json.dumps(merged, ensure_ascii=False))
        os.replace(tmp, canonical)
        n_written += 1
        print(f"  [wrote] {canonical.name}  "
              f"(parts={len(parts_sorted)}, methods={len(union_methods)}, "
              f"ratios={len(union_ratios)}, rows={len(order)}, "
              f"{canonical.stat().st_size/1e6:.1f} MB)")

        if args.delete_tag_files:
            for p in non_canonical_parts:
                try:
                    p.unlink()
                except OSError:
                    pass

    print(f"done — wrote {n_written} canonical file(s) under {args.in_dir}")


if __name__ == "__main__":
    main()
