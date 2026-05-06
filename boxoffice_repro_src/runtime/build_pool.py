#!/usr/bin/env python3
"""
Build the corpus-wide passage pool used by the BoxOffice K=5 methods.

The pool is built from the canonical 100-film dossier catalogue in the
BoxOffice bundle (override with --corpus-jsonl). The same catalogue is the SOURCE of
every passage that ever appears in any seed's eval / full file, so the
catalogue IS the deduped pool. Runtime queries are still drawn from
the per-seed `_eval.jsonl` by run_boxoffice.py.

Mirrors the LB / v9 pattern: one pool per "dataset" (here a dataset =
one seed file). Because both seeds index into the same catalogue, the
per-seed pool artefacts contain identical data — we still emit the
per-seed files so run_boxoffice.py's per-seed indexing code stays
unchanged.

Each row of `canonical_corpus_100_chunks.jsonl` is a flat film record
with at least `{title, text}` (we tolerate extra metadata fields like
`entity_id`, `box_office_musd`). We dedupe by
`(title, md5(text[:500]))` (the same key the runtime md5 lookup uses,
so a passage that appears in an eval query is found in the pool by
its first 500 chars). We embed each unique passage with e5-base-v2 and
write:

    pool/boxoffice_s<seed>_chunks.json       list of {chunk_id, title,
                                                       text_md5, text}
    pool/boxoffice_s<seed>_embeddings.npy    (n_chunks, 768) float32 L2-norm
    pool/boxoffice_s<seed>_infusion_k<K>.json  one file per K in --ks
            {str(chunk_id): [[neighbour_id, score], …×K]}
    pool/boxoffice_s<seed>_text_md5.json     {md5(text[:500]): chunk_id}
                                              (runtime O(1) lookup so
                                              run_boxoffice doesn't need
                                              the embeddings)

The infusion KNN is corpus-wide — same semantics as
`paper2/exp/offline_limit/output/lb/<ds>_infusion_k10.json` for LB and
`output/extended/v8_<ds>_infusion_k10.json` for v8/v9. Self is excluded
from each chunk's neighbour list. We compute one top-max(--ks) KNN
table and slice it to write each requested K.

Idempotent: per-seed, if every output for the requested --ks already
exists, skip unless `--force`.

Usage:
    python boxoffice/build_pool.py \\
        --npus 0 1 2 3 --pack 2

Default: builds pools for both seeds (7 and 11) at K=5 and K=10 from
the canonical catalogue. Pass `--seeds 7` or `--ks 10` to narrow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _lib import (                                # noqa: E402
    DEFAULT_E5, embed_parallel, prefetch_model,
    topk_neighbours, write_json, write_npy,
)

DEFAULT_CORPUS_JSONL = Path(
    os.environ.get(
        "BOXOFFICE_CORPUS_JSONL",
        str(ROOT_DIR.parent / "boxoffice_repro_src" / "corpus" /
            "canonical_corpus_100_chunks.jsonl")))
DEFAULT_SEEDS = (7, 11, 13, 17, 19, 23, 29, 31, 47, 73)
DEFAULT_KS = (5, 10)

OUTPUT_DIR = SCRIPT_DIR / "pool"


def _md5_500(text: str) -> str:
    return hashlib.md5((text or "")[:500].encode("utf-8",
                                                  errors="ignore")
                       ).hexdigest()[:16]


def _load_jsonl(p: Path):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_corpus_pool(corpus_jsonl: Path) -> list[dict]:
    """Read the canonical catalogue and dedupe by
    `(title, md5(text[:500]))`. Each row is a flat film record:
    `{title, text, ...optional metadata}`. Tolerates rows missing
    `title` (rare; falls back to empty string) but skips rows with
    empty `text`.

    Returns list of {chunk_id, title, text_md5, text}.
    """
    if not corpus_jsonl.exists():
        raise FileNotFoundError(
            f"missing corpus jsonl: {corpus_jsonl}\n"
            f"(set BOXOFFICE_CORPUS_JSONL or pass --corpus-jsonl to override)")
    by_key: dict[tuple[str, str], dict] = {}
    rows_seen = 0
    for row in _load_jsonl(corpus_jsonl):
        rows_seen += 1
        title = row.get("title", "") or ""
        text  = row.get("text",  "") or ""
        if not text:
            continue
        key = (title, _md5_500(text))
        if key in by_key:
            continue
        by_key[key] = {
            "chunk_id": len(by_key),
            "title":    title,
            "text_md5": key[1],
            "text":     text,
        }
    chunks = sorted(by_key.values(), key=lambda r: r["chunk_id"])
    print(f"  corpus: {rows_seen:,} rows  →  {len(chunks):,} unique chunks "
          f"(from {corpus_jsonl.name})")
    return chunks


def process_seed(seed: int, args, prebuilt_chunks: list[dict]) -> None:
    """Write the per-seed pool artefacts. The chunk universe is shared
    across seeds (it's the canonical catalogue), but we still emit
    per-seed files so run_boxoffice.py's per-seed indexing convention
    holds without changes."""
    out_chunks = args.out_dir / f"boxoffice_s{seed}_chunks.json"
    out_emb    = args.out_dir / f"boxoffice_s{seed}_embeddings.npy"
    out_md5    = args.out_dir / f"boxoffice_s{seed}_text_md5.json"
    out_infs   = [args.out_dir / f"boxoffice_s{seed}_infusion_k{k}.json"
                  for k in args.ks]
    all_outputs = [out_chunks, out_emb, out_md5] + out_infs

    if all(p.exists() for p in all_outputs) and not args.force:
        print(f"\n=== s{seed}: pool already on disk; pass --force to rebuild")
        for p in all_outputs:
            print(f"  {p}")
        return

    print(f"\n=== s{seed}: writing pool ===")
    chunks = prebuilt_chunks
    if not chunks:
        raise SystemExit(f"no chunks built for s{seed}")
    write_json(out_chunks, chunks)

    md5_index = {c["text_md5"]: c["chunk_id"] for c in chunks}
    write_json(out_md5, md5_index)

    # e5 wants no prefix for symmetric chunk-to-chunk KNN.
    cache = args.out_dir / "_embed_cache" / f"s{seed}"
    embs = embed_parallel(
        [c["text"] for c in chunks], args.npus, args.model,
        cache, label=f"boxoffice_s{seed}_chunks",
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        prefix="", pack=args.pack,
    )
    write_npy(out_emb, embs)

    # Compute KNN at the largest requested K, then slice for the rest.
    # `topk_neighbours` excludes self for square inputs.
    k_max = max(args.ks)
    print(f"  s{seed}: computing top-{k_max} self-pool KNN "
          f"(will slice to ks={args.ks}) ...")
    inf_max = topk_neighbours(
        np.asarray(embs), k=k_max,
        chunk_ids=[c["chunk_id"] for c in chunks],
    )
    for k, out_inf in zip(args.ks, out_infs):
        if k == k_max:
            write_json(out_inf, inf_max)
        else:
            sliced = {cid: lst[:k] for cid, lst in inf_max.items()}
            write_json(out_inf, sliced)

    for p in all_outputs:
        sz = p.stat().st_size / 1e6
        print(f"  wrote {p}  ({sz:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-jsonl", type=Path,
                    default=DEFAULT_CORPUS_JSONL,
                    help="canonical corpus jsonl. Each row is one film "
                         "with at least {title, text}. Default: "
                         f"{DEFAULT_CORPUS_JSONL}")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS),
                    help="seeds to emit per-seed artefacts for. The chunk "
                         "universe is shared (canonical catalogue), so the "
                         "per-seed pool files contain the same data; the "
                         "split exists only because run_boxoffice.py indexes "
                         "by seed.")
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--npus", nargs="*", type=int, default=[0])
    ap.add_argument("--pack", type=int, default=1,
                    help="processes per NPU for embedding (default 1).")
    ap.add_argument("--model", default=DEFAULT_E5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--ks", nargs="*", type=int, default=list(DEFAULT_KS),
                    help=f"top-K neighbours per chunk; one infusion file per K "
                         f"(default {list(DEFAULT_KS)}).")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not args.ks:
        raise SystemExit("--ks must list at least one K value")
    args.ks = sorted(set(int(k) for k in args.ks))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading canonical corpus from {args.corpus_jsonl}")
    chunks = build_corpus_pool(args.corpus_jsonl)
    print(f"Building per-seed pools in {args.out_dir} for seeds: {args.seeds}")
    for seed in args.seeds:
        process_seed(seed, args, chunks)
    print("\nDone.")


if __name__ == "__main__":
    main()
