#!/usr/bin/env python3
"""Build the per-seed boxoffice paper2 pool artifacts.

This is a bundle-local adaptation of the original paper2 build_pool.py
logic. The goal here is exact artifact parity, so the embedding and KNN
code path intentionally mirrors the upstream implementation instead of a
simplified reimplementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_JSONL = ROOT / "corpus" / "canonical_corpus_100_chunks.jsonl"
DEFAULT_OUT_DIR = ROOT / "pool_v4_balanced_m4"
DEFAULT_E5 = "/data/weights/e5-base-v2"
DEFAULT_SEEDS = (7, 11)
DEFAULT_KS = (5, 10)


def _md5_500(text: str) -> str:
    return hashlib.md5(
        (text or "")[:500].encode("utf-8", errors="ignore")
    ).hexdigest()[:16]


def _load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_corpus_pool(corpus_jsonl: Path) -> list[dict]:
    if not corpus_jsonl.exists():
        raise FileNotFoundError(f"missing corpus jsonl: {corpus_jsonl}")

    by_key: dict[tuple[str, str], dict] = {}
    rows_seen = 0
    for row in _load_jsonl(corpus_jsonl):
        rows_seen += 1
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        if not text:
            continue
        key = (title, _md5_500(text))
        if key in by_key:
            continue
        by_key[key] = {
            "chunk_id": len(by_key),
            "title": title,
            "text_md5": key[1],
            "text": text,
        }

    chunks = sorted(by_key.values(), key=lambda row: row["chunk_id"])
    print(
        f"  corpus: {rows_seen:,} rows  ->  {len(chunks):,} unique chunks "
        f"(from {corpus_jsonl.name})"
    )
    return chunks


def _embed_worker(
    npu_id: int,
    shard_in: str,
    shard_out: str,
    model_name: str,
    batch_size: int,
    max_seq_length: int,
    prefix: str,
) -> None:
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

    with open(shard_in, "r", encoding="utf-8") as handle:
        texts = json.load(handle)
    if prefix:
        texts = [prefix + text for text in texts]

    print(
        f"[npu={npu_id}] device={device} loading {model_name} "
        f"prefix={prefix!r}  ({len(texts):,} texts)",
        flush=True,
    )
    model = SentenceTransformer(model_name, device=device)
    if max_seq_length:
        model.max_seq_length = max_seq_length

    t0 = time.time()
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(shard_out, embs.astype(np.float32, copy=False))
    print(
        f"[npu={npu_id}] done {len(texts):,} in {time.time() - t0:.1f}s "
        f"-> {shard_out}",
        flush=True,
    )


def embed_parallel(
    texts: Sequence[str],
    npus: Sequence[int],
    model_name: str,
    cache_dir: Path,
    label: str,
    batch_size: int = 64,
    max_seq_length: int = 512,
    prefix: str = "",
    pack: int = 1,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if pack < 1:
        raise ValueError(f"pack must be >=1, got {pack}")

    workers: list[tuple[int, int]] = [(npu_id, slot) for npu_id in npus for slot in range(pack)]
    n_shards = len(workers)

    shard_paths: list[tuple[Path, Path, int]] = []
    for index in range(n_shards):
        lo = (index * len(texts)) // n_shards
        hi = ((index + 1) * len(texts)) // n_shards
        shard_in = cache_dir / f"{label}_shard{index}_in.json"
        shard_out = cache_dir / f"{label}_shard{index}_out.npy"
        with open(shard_in, "w", encoding="utf-8") as handle:
            json.dump(list(texts[lo:hi]), handle, ensure_ascii=False)
        shard_paths.append((shard_in, shard_out, hi - lo))

    print(
        f"  spawning {n_shards} workers ({len(npus)} NPUs x pack={pack}) "
        f"sizes={[item[2] for item in shard_paths]}"
    )
    ctx = mp.get_context("spawn")
    procs = []
    for index, (npu_id, _slot) in enumerate(workers):
        shard_in, shard_out, _size = shard_paths[index]
        proc = ctx.Process(
            target=_embed_worker,
            args=(
                npu_id,
                str(shard_in),
                str(shard_out),
                model_name,
                batch_size,
                max_seq_length,
                prefix,
            ),
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    bad = [(index, proc.exitcode) for index, proc in enumerate(procs) if proc.exitcode != 0]
    if bad:
        raise RuntimeError(f"embedding workers failed: {bad}")

    parts = [np.load(shard_out) for _shard_in, shard_out, _size in shard_paths]
    embs = np.concatenate(parts, axis=0) if parts else np.zeros((0, 0), np.float32)
    if embs.shape[0] != len(texts):
        raise RuntimeError(
            f"embedding count mismatch: {embs.shape[0]} vs {len(texts)}"
        )

    for shard_in, shard_out, _size in shard_paths:
        try:
            shard_in.unlink()
        except OSError:
            pass
        try:
            shard_out.unlink()
        except OSError:
            pass
    return embs


def _topk_per_row(sims: np.ndarray, k: int) -> np.ndarray:
    n_cols = sims.shape[1]
    if n_cols <= k:
        return np.argsort(-sims, axis=1)
    part = np.argpartition(-sims, k, axis=1)[:, :k]
    rows = np.arange(part.shape[0])[:, None]
    sub = sims[rows, part]
    return part[rows, np.argsort(-sub, axis=1)]


def topk_neighbours(
    embs: np.ndarray,
    k: int,
    chunk_ids: Sequence[int] | None = None,
    batch: int = 256,
) -> dict[int, list[list]]:
    n_rows = embs.shape[0]
    if chunk_ids is None:
        chunk_ids = list(range(n_rows))
    chunk_ids = np.asarray(chunk_ids)

    out: dict[int, list[list]] = {}
    for start in range(0, n_rows, batch):
        stop = min(n_rows, start + batch)
        sims = embs[start:stop] @ embs.T
        for row in range(start, stop):
            sims[row - start, row] = -np.inf
        order = _topk_per_row(sims, k)
        for row in range(start, stop):
            ids = order[row - start]
            scores = sims[row - start, ids]
            keep = np.isfinite(scores)
            ids = ids[keep]
            scores = scores[keep]
            out[int(chunk_ids[row])] = [
                [int(chunk_ids[idx]), float(score)]
                for idx, score in zip(ids, scores)
            ]
    return out


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False)
    print(f"  wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def write_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32, copy=False))
    print(f"  wrote {path}  shape={arr.shape}  ({path.stat().st_size / 1e6:.1f} MB)")


def process_seed(seed: int, args, chunks: list[dict]) -> None:
    out_chunks = args.out_dir / f"boxoffice_s{seed}_chunks.json"
    out_emb = args.out_dir / f"boxoffice_s{seed}_embeddings.npy"
    out_md5 = args.out_dir / f"boxoffice_s{seed}_text_md5.json"
    out_infs = [
        args.out_dir / f"boxoffice_s{seed}_infusion_k{k}.json" for k in args.ks
    ]
    all_outputs = [out_chunks, out_emb, out_md5] + out_infs

    if all(path.exists() for path in all_outputs) and not args.force:
        print(f"\n=== s{seed}: pool already on disk; pass --force to rebuild")
        for path in all_outputs:
            print(f"  {path}")
        return

    print(f"\n=== s{seed}: writing pool ===")
    write_json(out_chunks, chunks)

    md5_index = {chunk["text_md5"]: chunk["chunk_id"] for chunk in chunks}
    write_json(out_md5, md5_index)

    cache_dir = args.out_dir / "_embed_cache" / f"s{seed}"
    embs = embed_parallel(
        [chunk["text"] for chunk in chunks],
        args.npus,
        args.model,
        cache_dir,
        label=f"boxoffice_s{seed}_chunks",
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        prefix="",
        pack=args.pack,
    )
    write_npy(out_emb, embs)

    k_max = max(args.ks)
    print(
        f"  s{seed}: computing top-{k_max} self-pool KNN "
        f"(will slice to ks={args.ks}) ..."
    )
    inf_max = topk_neighbours(
        np.asarray(embs),
        k=k_max,
        chunk_ids=[chunk["chunk_id"] for chunk in chunks],
    )
    for k, out_inf in zip(args.ks, out_infs):
        payload = inf_max if k == k_max else {
            chunk_id: neighbours[:k] for chunk_id, neighbours in inf_max.items()
        }
        write_json(out_inf, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS_JSONL)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--npus", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--pack", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_E5)
    parser.add_argument("--embedding-model-path", dest="model")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-batch-size", type=int)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--ks", nargs="*", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.embedding_batch_size is not None:
        args.batch_size = args.embedding_batch_size
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
