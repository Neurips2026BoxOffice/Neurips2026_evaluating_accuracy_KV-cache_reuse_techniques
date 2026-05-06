"""Shared utilities for paper/exp/offline_limit/.

Two pillars:

1. ``embed_parallel`` — multi-NPU sentence-transformer embedder factored
   from paper/singledoc_extend/sde_chunk_embed_retrieve.py:_embed_worker
   and embed_parallel. Adds ``pack`` (workers per NPU) so a small model
   like e5-base can run multiple processes per NPU.

2. KNN helpers (``topk_neighbours``, ``topk_against``) — pure numpy,
   reuses the argpartition pattern from
   sde_tighten_qasper.py:_topk_per_row.

Both pillars are imported by the per-stage scripts (01_..06_).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DEFAULT_E5 = "/data/weights/e5-base-v2"


# ── prefix conventions for retrieval-tuned embedders ──────────────────
def auto_prefixes(model_name: str) -> tuple[str, str]:
    lower = model_name.lower()
    if "e5" in lower:
        return "query: ", "passage: "
    if "bge" in lower and "en" in lower:
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""


# ── one worker process: pin to one NPU, embed one shard, write .npy ───
def _embed_worker(npu_id: int, shard_in: str, shard_out: str,
                  model_name: str, batch_size: int, max_seq_length: int,
                  prefix: str) -> None:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(npu_id)
    os.environ["ASCEND_VISIBLE_DEVICES"] = str(npu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(npu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    import torch
    torch.set_num_threads(2)
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    from sentence_transformers import SentenceTransformer

    if hasattr(torch, "npu") and torch.npu.is_available():
        device = "npu:0"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    with open(shard_in) as f:
        texts = json.load(f)
    if prefix:
        texts = [prefix + t for t in texts]

    print(f"[npu={npu_id}] device={device} loading {model_name} "
          f"prefix={prefix!r}  ({len(texts):,} texts)", flush=True)
    model = SentenceTransformer(model_name, device=device)
    if max_seq_length:
        model.max_seq_length = max_seq_length

    t0 = time.time()
    embs = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    np.save(shard_out, embs.astype(np.float32, copy=False))
    print(f"[npu={npu_id}] done {len(texts):,} in {time.time()-t0:.1f}s "
          f"-> {shard_out}", flush=True)


def embed_parallel(texts: Sequence[str], npus: Sequence[int],
                   model_name: str, cache_dir: Path, label: str,
                   batch_size: int = 64, max_seq_length: int = 512,
                   prefix: str = "", pack: int = 1) -> np.ndarray:
    """Embed ``texts`` across ``npus`` × ``pack`` workers in parallel.

    Returns L2-normalised float32 embeddings in input order. Shard files
    are cleaned up on success. Cache dir is created if missing.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if pack < 1:
        raise ValueError(f"pack must be >=1, got {pack}")

    # (npu_id, slot) tuples — one worker per (npu, slot)
    workers: list[tuple[int, int]] = [
        (n, s) for n in npus for s in range(pack)
    ]
    n_shards = len(workers)

    shard_paths: list[tuple[Path, Path, int]] = []
    for i in range(n_shards):
        lo = (i * len(texts)) // n_shards
        hi = ((i + 1) * len(texts)) // n_shards
        p_in  = cache_dir / f"{label}_shard{i}_in.json"
        p_out = cache_dir / f"{label}_shard{i}_out.npy"
        with open(p_in, "w", encoding="utf-8") as f:
            json.dump(list(texts[lo:hi]), f, ensure_ascii=False)
        shard_paths.append((p_in, p_out, hi - lo))

    print(f"  spawning {n_shards} workers ({len(npus)} NPUs × pack={pack}) "
          f"sizes={[s[2] for s in shard_paths]}")
    ctx = mp.get_context("spawn")
    procs = []
    for i, (npu_id, slot) in enumerate(workers):
        p_in, p_out, _ = shard_paths[i]
        p = ctx.Process(target=_embed_worker, args=(
            npu_id, str(p_in), str(p_out), model_name,
            batch_size, max_seq_length, prefix,
        ))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    bad = [(i, p.exitcode) for i, p in enumerate(procs) if p.exitcode != 0]
    if bad:
        raise RuntimeError(f"embedding workers failed: {bad}")

    parts = [np.load(p_out) for _, p_out, _ in shard_paths]
    embs = np.concatenate(parts, axis=0) if parts else np.zeros((0, 0), np.float32)
    if embs.shape[0] != len(texts):
        raise RuntimeError(
            f"embedding count mismatch: {embs.shape[0]} vs {len(texts)}")

    for p_in, p_out, _ in shard_paths:
        try: p_in.unlink()
        except OSError: pass
        try: p_out.unlink()
        except OSError: pass
    return embs


def prefetch_model(model_name: str) -> None:
    if Path(model_name).is_dir():
        print(f"  model is local: {model_name} (no prefetch)")
        return
    print(f"  prefetching '{model_name}'")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=model_name)


# ── KNN helpers (numpy) ───────────────────────────────────────────────
def _topk_per_row(sims: np.ndarray, k: int) -> np.ndarray:
    """For each row of ``sims`` return descending-order indices of the
    top-k columns. ``sims.shape == (B, N)`` -> result shape (B, min(k,N))."""
    n = sims.shape[1]
    if n <= k:
        return np.argsort(-sims, axis=1)
    part = np.argpartition(-sims, k, axis=1)[:, :k]
    rows = np.arange(part.shape[0])[:, None]
    sub  = sims[rows, part]
    return part[rows, np.argsort(-sub, axis=1)]


def topk_neighbours(embs: np.ndarray, k: int,
                    chunk_ids: Sequence[int] | None = None,
                    doc_ids: Sequence[str] | None = None,
                    batch: int = 256) -> dict[int, list[list]]:
    """Square KNN inside ``embs``. For each row i, returns the top-k other
    rows (excluding i itself).

    Args:
        embs:       (N, D) L2-normalised embeddings.
        k:          neighbours per row.
        chunk_ids:  optional explicit ids; defaults to range(N).
        doc_ids:    if given, restricts neighbours to the same doc_id as i.
        batch:      number of rows processed per dot-product.

    Returns: ``{chunk_id: [[neighbour_chunk_id, score], …]}``.
    """
    n = embs.shape[0]
    if chunk_ids is None:
        chunk_ids = list(range(n))
    chunk_ids = np.asarray(chunk_ids)

    if doc_ids is not None and len(doc_ids) != n:
        raise ValueError("doc_ids length must match embs.shape[0]")

    # Group by doc when intra-doc
    by_doc: dict[str, np.ndarray] | None = None
    if doc_ids is not None:
        d = np.asarray(doc_ids)
        by_doc = {}
        for di in np.unique(d):
            by_doc[di] = np.where(d == di)[0]

    out: dict[int, list[list]] = {}

    if by_doc is None:
        # global KNN — batched
        for start in range(0, n, batch):
            stop = min(n, start + batch)
            sims = embs[start:stop] @ embs.T          # (b, N)
            # mask self
            for r in range(start, stop):
                sims[r - start, r] = -np.inf
            order = _topk_per_row(sims, k)
            for r in range(start, stop):
                ids = order[r - start]
                sc  = sims[r - start, ids]
                # drop -inf entries (can happen if N <= k)
                keep = np.isfinite(sc)
                ids, sc = ids[keep], sc[keep]
                out[int(chunk_ids[r])] = [
                    [int(chunk_ids[j]), float(s)] for j, s in zip(ids, sc)
                ]
        return out

    # intra-doc: small per-doc dot products
    for _, idxs in by_doc.items():
        if len(idxs) <= 1:
            for i in idxs:
                out[int(chunk_ids[i])] = []
            continue
        sub = embs[idxs]                              # (m, D)
        sims = sub @ sub.T                            # (m, m)
        np.fill_diagonal(sims, -np.inf)
        kk = min(k, len(idxs) - 1)
        order = _topk_per_row(sims, kk)
        for r, gi in enumerate(idxs):
            local_ids = order[r]
            sc = sims[r, local_ids]
            keep = np.isfinite(sc)
            local_ids, sc = local_ids[keep], sc[keep]
            out[int(chunk_ids[gi])] = [
                [int(chunk_ids[idxs[j]]), float(s)]
                for j, s in zip(local_ids, sc)
            ]
    return out


def _topk_slice_worker(args):
    """Multiprocessing worker for `topk_neighbours_parallel`. Computes
    intra-corpus top-K for rows[start:stop] against the full embeddings
    matrix. Each worker mmap-loads the embeddings so the OS page cache
    keeps a single physical copy across all workers.

    Returns: dict[int chunk_id -> [[int neighbour_id, float score], ...]].
    """
    (start, stop, embs_path, k, batch, chunk_ids_path) = args
    import numpy as np  # noqa: F811 (re-import in spawned process)
    embs = np.load(embs_path, mmap_mode="r")
    n = embs.shape[0]
    chunk_ids = np.load(chunk_ids_path) if chunk_ids_path else np.arange(n)
    out: dict[int, list[list]] = {}
    for r0 in range(start, stop, batch):
        r1 = min(r0 + batch, stop)
        sims = np.asarray(embs[r0:r1]) @ embs.T            # (b, n)
        for r in range(r0, r1):
            sims[r - r0, r] = -np.inf
        order = _topk_per_row(sims, k)
        for r in range(r0, r1):
            ids = order[r - r0]
            sc  = sims[r - r0, ids]
            keep = np.isfinite(sc)
            ids, sc = ids[keep], sc[keep]
            out[int(chunk_ids[r])] = [
                [int(chunk_ids[j]), float(s)] for j, s in zip(ids, sc)
            ]
    return out


def topk_neighbours_parallel(embs_path: Path, k: int,
                             chunk_ids: Sequence[int] | None = None,
                             jobs: int = 8,
                             batch: int = 256) -> dict[int, list[list]]:
    """Same contract as `topk_neighbours` but row-slice parallelised.

    Workers each mmap the .npy from disk (no per-worker copy of the full
    matrix in physical RAM). For 1M-chunk corpora the wall-clock drops
    from ~10-15h single-process to ~jobs× faster, plus BLAS parallelism
    inside each worker (set OMP/MKL threads accordingly).

    Args:
        embs_path: path to a .npy holding (N, D) float32 L2-normalised
                   embeddings.
        k:         neighbours per row.
        chunk_ids: optional explicit ids; defaults to range(N).
        jobs:      number of worker processes.
        batch:     rows per dot-product batch inside a worker.

    Note: doc_ids restriction (intra-doc) is NOT supported here — fall
    back to the single-process `topk_neighbours` for that case.
    """
    import multiprocessing as _mp
    import tempfile

    embs_path = Path(embs_path)
    embs_meta = np.load(embs_path, mmap_mode="r")
    n = embs_meta.shape[0]
    print(f"  topk_parallel: N={n:,}  D={embs_meta.shape[1]}  "
          f"k={k}  jobs={jobs}  batch={batch}")

    cid_path = None
    if chunk_ids is not None:
        # Persist chunk_ids to disk so workers don't re-pickle the array
        # for every task; mmap from a temp file instead.
        cid_arr = np.asarray(chunk_ids, dtype=np.int64)
        if cid_arr.shape[0] != n:
            raise ValueError(f"chunk_ids length {cid_arr.shape[0]} != N {n}")
        with tempfile.NamedTemporaryFile(
                suffix="_cids.npy", delete=False,
                dir=embs_path.parent) as f:
            cid_path = Path(f.name)
        np.save(cid_path, cid_arr)

    # Slice [0, n) into roughly equal contiguous chunks per worker.
    boundaries = np.linspace(0, n, jobs + 1, dtype=int)
    tasks = [(int(boundaries[i]), int(boundaries[i + 1]),
              str(embs_path), k, batch,
              str(cid_path) if cid_path else None)
             for i in range(jobs)]
    print(f"  spawning {jobs} workers, slice sizes: "
          f"{[t[1] - t[0] for t in tasks]}")

    out: dict[int, list[list]] = {}
    try:
        with _mp.get_context("fork").Pool(processes=jobs) as pool:
            for partial in pool.imap_unordered(_topk_slice_worker, tasks):
                out.update(partial)
                print(f"    [merged] {len(out):,}/{n:,} rows done",
                      flush=True)
    finally:
        if cid_path and cid_path.exists():
            cid_path.unlink()

    print(f"  topk_parallel: done, {len(out):,} entries")
    return out


def topk_against(query_embs: np.ndarray, doc_embs: np.ndarray, k: int,
                 query_ids: Sequence[int],
                 doc_ids: Sequence[int] | None = None,
                 batch: int = 256) -> dict[int, list[list]]:
    """Rectangular top-K: for each row of ``query_embs`` find top-k rows
    of ``doc_embs``. Self is NOT excluded — caller's responsibility if
    query_embs ⊂ doc_embs.

    Returns ``{query_id: [[doc_id, score], …]}``.

    Logs per-batch progress every ~5s of wall time, with flush=True so
    stdout buffering doesn't hide it when redirected to a log file.
    """
    n = query_embs.shape[0]
    if doc_ids is None:
        doc_ids = np.arange(doc_embs.shape[0], dtype=np.int64)
    else:
        doc_ids = np.asarray(doc_ids, dtype=np.int64)

    print(f"  topk_against: n_query={n:,}  n_doc={doc_embs.shape[0]:,}  "
          f"k={k}  batch={batch}", flush=True)

    out: dict[int, list[list]] = {}
    t_first = time.time()
    t_log   = t_first
    for start in range(0, n, batch):
        stop = min(n, start + batch)
        sims = query_embs[start:stop] @ doc_embs.T
        order = _topk_per_row(sims, k)
        for r in range(start, stop):
            ids = order[r - start]
            sc  = sims[r - start, ids]
            out[int(query_ids[r])] = [
                [int(doc_ids[j]), float(s)] for j, s in zip(ids, sc)
            ]
        now = time.time()
        if now - t_log >= 5.0 or stop == n:
            elapsed = now - t_first
            rate = stop / max(elapsed, 1e-6)
            eta = (n - stop) / max(rate, 1e-6)
            print(f"    [{stop:>6,}/{n:,}]  elapsed={elapsed:6.1f}s  "
                  f"rate={rate:6.1f} q/s  eta={eta:6.1f}s", flush=True)
            t_log = now
    return out


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    print(f"  wrote {path}  ({path.stat().st_size/1e6:.1f} MB)")


def write_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32, copy=False))
    print(f"  wrote {path}  shape={arr.shape}  "
          f"({path.stat().st_size/1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────
# Identity bridge: LB or v7 chunk_id → wholewiki chunk_id
#
# The wholewiki chunk corpus is built by the same expand_paragraph_budgeted
# pipeline as v5/v6/v7 (see 05_embed_wholewiki.py), so the *text* of the
# corresponding chunk is identical. We bridge by hashing chunk text. The
# resulting map is cached on disk; rebuild costs one streaming pass over
# the (large) wholewiki chunks.jsonl.
# ─────────────────────────────────────────────────────────────────────
import hashlib


def _text_key(text: str) -> str:
    """md5 over the first 500 chars of the chunk text — same convention
    as paper/lb_extend/v_all/va_analyse_reuse.py uses for chunk identity.
    Truncating saves time on giant chunks; collisions on Wikipedia
    paragraphs are vanishingly rare in practice."""
    return hashlib.md5(text[:500].encode("utf-8", errors="ignore")
                       ).hexdigest()[:16]


def build_to_wholewiki_idmap(
    target_chunks_path: Path,
    wholewiki_chunks_jsonl: Path,
    out_path: Path,
    force: bool = False,
) -> dict[str, int]:
    """Build a {target_chunk_id (int|str) → wholewiki chunk_id (int)} map
    by streaming through the wholewiki chunks.jsonl once and matching by
    md5(text[:500]).

    Loads cached map if it exists and not `force`. Returns the map.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return {k: int(v) for k, v in json.loads(out_path.read_text()).items()}

    target_chunks = json.loads(target_chunks_path.read_text())
    target_by_hash: dict[str, str] = {}
    for c in target_chunks:
        text = c.get("text") or ""
        if not text:
            continue
        target_by_hash[_text_key(text)] = str(c["chunk_id"])

    print(f"  resolving {len(target_by_hash):,} unique target hashes "
          f"against {wholewiki_chunks_jsonl.name}…", flush=True)

    out: dict[str, int] = {}
    n_seen = 0
    t0 = time.time()
    with open(wholewiki_chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            n_seen += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            h = _text_key(rec.get("text") or "")
            tid = target_by_hash.get(h)
            if tid is not None:
                out[tid] = int(rec["chunk_id"])
            if n_seen % 2_000_000 == 0:
                print(f"    streamed {n_seen:,} ww chunks in "
                      f"{time.time()-t0:.0f}s, mapped {len(out):,}/"
                      f"{len(target_by_hash):,}", flush=True)
            if len(out) == len(target_by_hash):
                break  # everything resolved — early exit

    n_unresolved = len(target_by_hash) - len(out)
    if n_unresolved:
        print(f"  WARN: {n_unresolved} target chunks had no wholewiki "
              f"match (text not in wiki dump?)")
    out_path.write_text(json.dumps(out))
    print(f"  wrote {out_path.name} "
          f"({len(out):,} entries, {out_path.stat().st_size/1e6:.1f} MB)")
    return out
