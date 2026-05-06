#!/usr/bin/env python3
"""Build semantic warm caches from the full one-vs-all box-office corpus.

Unlike the generic warm_cache_builder, this script retrieves from the explicit
movie corpus rather than only from chunks that happen to appear in the eval
JSONL files.  The exported cache still uses runtime-compatible chunk token
boundaries, so cache IDs match pilot_eval --single-prompt evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

GEN_ROOT = Path(__file__).resolve().parents[2]
SRC = GEN_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch

from cachebend.ncf.cutils import build_tokenizer, chash, split_prompt_for_warm_chunks, to_str_prompt
from cachebend.ncf.warm_cache_builder import (
    build_topk_contexts_embedding,
    build_topk_contexts_lexical,
    export_contextualized_cache,
)
from generate_one_vs_all_v1 import (
    PROMPT_VARIANTS,
    build_token_counter,
    load_films_from_matrix,
    long_movie_chunk,
)

INSTRUCTION = PROMPT_VARIANTS["closed_world"]


def parse_csv_paths(value: str) -> List[Path]:
    paths: List[Path] = []
    for token in str(value or "").split(","):
        token = token.strip()
        if token:
            paths.append(Path(token))
    return paths


def runtime_chunk_tokens(tokenizer: Any, sep_token: str, text: str) -> List[int]:
    prompt = to_str_prompt(tokenizer, sep_token, [INSTRUCTION, text, "Question: X"])
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    sep_ids = tokenizer(sep_token, add_special_tokens=False).input_ids
    chunks = split_prompt_for_warm_chunks(
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(sep_ids, dtype=torch.long),
        tokenizer,
    )
    if len(chunks) < 3:
        raise RuntimeError(f"Expected instruction/chunk/query split, got {len(chunks)} chunks")
    return [int(x) for x in chunks[1].tokens.tolist()]


def runtime_chunk_cid(tokens: List[int]) -> str:
    return str(chash(torch.tensor(tokens, dtype=torch.long)))


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def assert_no_reference_retrieval(fusion_docs: List[Dict[str, Any]]) -> None:
    bad_rows = []
    for row_idx, row in enumerate(fusion_docs):
        ref_cid = str(row.get("reference_cid") or "")
        for ctx_idx, ctx in enumerate(row.get("top_k_contexts", [])):
            if str(ctx.get("cid") or "") == ref_cid:
                bad_rows.append((row_idx, ctx_idx, ref_cid))
    if bad_rows:
        preview = ", ".join(f"row={r} ctx={c} cid={cid}" for r, c, cid in bad_rows[:5])
        raise RuntimeError(f"retrieval included reference chunk: {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl-csv", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--chunk-target-tokens", type=int, default=512)
    parser.add_argument("--retrieval-mode", choices=["embedding", "lexical"], default="embedding")
    parser.add_argument("--embedding-model-path", default="/data/weights/e5-base-v2")
    parser.add_argument("--embedding-device", default="npu:0")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--out-soup", type=Path, required=True)
    parser.add_argument("--out-fusion-docs", type=Path, required=True)
    parser.add_argument("--out-cache-pt", type=Path, required=True)
    parser.add_argument("--out-stats", type=Path, required=True)
    args = parser.parse_args()

    source_paths = parse_csv_paths(args.source_jsonl_csv)
    if not source_paths:
        raise ValueError("--source-jsonl-csv did not contain any paths")

    tokenizer = build_tokenizer(args.model_path)
    sep_token = tokenizer.sep_token if tokenizer.sep_token else "<DSEP>"
    count_tokens = build_token_counter(Path(args.model_path))
    films = load_films_from_matrix(source_paths)

    chunks: List[Dict[str, Any]] = []
    for film in films:
        text, token_count = long_movie_chunk(film, args.chunk_target_tokens, count_tokens)
        tokens = runtime_chunk_tokens(tokenizer, sep_token, text)
        cid = runtime_chunk_cid(tokens)
        chunks.append(
            {
                "cid": cid,
                "text": text,
                "tokens": tokens,
                "entity_id": film.entity_id,
                "title": film.title,
                "box_office_musd": film.box_office_musd,
                "chunk_token_count": token_count if token_count is not None else len(tokens),
            }
        )

    if args.retrieval_mode == "embedding":
        fusion_docs = build_topk_contexts_embedding(
            chunks=chunks,
            top_k=args.top_k,
            embedding_model_path=args.embedding_model_path,
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size,
        )
    else:
        fusion_docs = build_topk_contexts_lexical(chunks=chunks, top_k=args.top_k)
    assert_no_reference_retrieval(fusion_docs)

    write_jsonl(
        args.out_soup,
        [
            {
                "id": f"chunk_{idx}",
                "cid": row["cid"],
                "entity_id": row["entity_id"],
                "title": row["title"],
                "box_office_musd": row["box_office_musd"],
                "contents": row["text"],
                "tokens": row["tokens"],
            }
            for idx, row in enumerate(chunks)
        ],
    )
    write_jsonl(args.out_fusion_docs, fusion_docs)

    export_stats = export_contextualized_cache(
        fusion_docs=fusion_docs,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        use_eager=False,
        out_cache_pt=args.out_cache_pt,
    )

    stats = {
        "source_jsonl": [str(path) for path in source_paths],
        "model_path": args.model_path,
        "top_k": args.top_k,
        "retrieval_mode": args.retrieval_mode,
        "embedding_model_path": args.embedding_model_path,
        "embedding_device": args.embedding_device,
        "embedding_batch_size": args.embedding_batch_size,
        "chunk_target_tokens": args.chunk_target_tokens,
        "corpus_films": len(films),
        "out_soup": str(args.out_soup),
        "out_fusion_docs": str(args.out_fusion_docs),
        "out_cache_pt": str(args.out_cache_pt),
        **export_stats,
    }
    args.out_stats.parent.mkdir(parents=True, exist_ok=True)
    args.out_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"retrieval_mode={args.retrieval_mode}")
    print(f"corpus_films={len(films)}")
    print(f"cache_entries={int(export_stats['cache_entries'])} skipped_entries={int(export_stats['skipped_entries'])}")
    print(f"saved_cache={args.out_cache_pt}")


if __name__ == "__main__":
    main()
