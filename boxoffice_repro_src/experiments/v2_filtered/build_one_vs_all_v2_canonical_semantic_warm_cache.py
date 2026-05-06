#!/usr/bin/env python3
"""Build one-vs-all warm caches with model-agnostic retrieval.

The old one-vs-all warm builder used the target model tokenizer while creating
the long movie dossier text.  That made the text embedded by E5 differ across
target models, so the retrieved top-k neighbors were not model-agnostic.

This builder fixes that by:
1. constructing canonical movie text without a target tokenizer,
2. computing or loading top-k neighbor ENTITY_IDs from that canonical text,
3. tokenizing the same canonical text with the target model only for cache CIDs
   and KV export.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

GEN_ROOT = Path(__file__).resolve().parents[2]
SRC = GEN_ROOT / "src"
V1_SCRIPTS = GEN_ROOT / "experiments" / "v1_helpers"
for path in (SRC, V1_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from cachebend.ncf.cutils import build_tokenizer, chash, split_prompt_for_warm_chunks, to_str_prompt
from cachebend.ncf.warm_cache_builder import (
    build_topk_contexts_embedding,
    build_topk_contexts_lexical,
    export_contextualized_cache,
)
from generate_one_vs_all_v1 import (  # type: ignore
    PROMPT_VARIANTS,
    load_films_from_matrix,
    long_movie_chunk,
)


INSTRUCTION = PROMPT_VARIANTS["closed_world"]


def parse_csv_paths(value: str) -> List[Path]:
    return [Path(token.strip()) for token in str(value or "").split(",") if token.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def canonical_retrieval_chunks(source_paths: List[Path], chunk_target_tokens: int) -> List[Dict[str, Any]]:
    films = load_films_from_matrix(source_paths)
    chunks: List[Dict[str, Any]] = []
    for film in films:
        # count_tokens=None is deliberate: this makes the dossier text
        # independent of the target model tokenizer.
        text, _token_count = long_movie_chunk(film, chunk_target_tokens, None)
        chunks.append(
            {
                "cid": f"canonical:{film.entity_id}",
                "text": text,
                "tokens": [],
                "entity_id": film.entity_id,
                "title": film.title,
                "box_office_musd": film.box_office_musd,
            }
        )
    return chunks


def build_canonical_retrieval(
    *,
    chunks: List[Dict[str, Any]],
    top_k: int,
    retrieval_mode: str,
    embedding_model_path: str,
    embedding_device: str,
    embedding_batch_size: int,
) -> List[Dict[str, Any]]:
    if retrieval_mode == "embedding":
        docs = build_topk_contexts_embedding(
            chunks=chunks,
            top_k=top_k,
            embedding_model_path=embedding_model_path,
            embedding_device=embedding_device,
            embedding_batch_size=embedding_batch_size,
        )
    elif retrieval_mode == "lexical":
        docs = build_topk_contexts_lexical(chunks=chunks, top_k=top_k)
    else:
        raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")

    rows = []
    for doc in docs:
        ref = doc["reference_context"]
        ref_entity = next(row["entity_id"] for row in chunks if row["text"] == ref)
        ref_chunk = next(row for row in chunks if row["entity_id"] == ref_entity)
        top = []
        for ctx in doc.get("top_k_contexts", []):
            top.append(
                {
                    "entity_id": ctx["entity_id"],
                    "box_office_musd": int(ctx["box_office_musd"]),
                    "canonical_cid": ctx["cid"],
                }
            )
        rows.append(
            {
                "reference_entity_id": ref_chunk["entity_id"],
                "reference_box_office_musd": int(ref_chunk["box_office_musd"]),
                "reference_canonical_cid": ref_chunk["cid"],
                "top_k_entity_ids": [item["entity_id"] for item in top],
                "top_k_box_office_musd": [item["box_office_musd"] for item in top],
                "top_k_canonical_cids": [item["canonical_cid"] for item in top],
                "canonical_reference_context": ref_chunk["text"],
            }
        )
    return rows


def target_runtime_chunks(
    *,
    canonical_chunks: List[Dict[str, Any]],
    model_path: str,
) -> Dict[str, Dict[str, Any]]:
    tokenizer = build_tokenizer(model_path)
    sep_token = tokenizer.sep_token if tokenizer.sep_token else "<DSEP>"
    out: Dict[str, Dict[str, Any]] = {}
    for chunk in canonical_chunks:
        tokens = runtime_chunk_tokens(tokenizer, sep_token, str(chunk["text"]))
        out[str(chunk["entity_id"])] = {
            "cid": runtime_chunk_cid(tokens),
            "text": str(chunk["text"]),
            "tokens": tokens,
            "entity_id": str(chunk["entity_id"]),
            "title": str(chunk["title"]),
            "box_office_musd": int(chunk["box_office_musd"]),
            "canonical_cid": str(chunk["cid"]),
        }
    return out


def fusion_docs_from_canonical(
    canonical_retrieval: List[Dict[str, Any]],
    runtime_by_entity: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out = []
    for row in canonical_retrieval:
        ref_entity = str(row["reference_entity_id"])
        ref = runtime_by_entity[ref_entity]
        top_contexts = []
        for entity_id in row["top_k_entity_ids"]:
            entity_id = str(entity_id)
            if entity_id == ref_entity:
                raise RuntimeError(f"Canonical retrieval included reference entity {ref_entity}")
            top_contexts.append(runtime_by_entity[entity_id])
        out.append(
            {
                "reference_context": ref["text"],
                "reference_cid": ref["cid"],
                "reference_tokens": ref["tokens"],
                "reference_entity_id": ref_entity,
                "reference_canonical_cid": ref["canonical_cid"],
                "top_k_contexts": top_contexts,
                "top_k_entity_ids": [ctx["entity_id"] for ctx in top_contexts],
                "top_k_canonical_cids": [ctx["canonical_cid"] for ctx in top_contexts],
                "concatenated_context": "\n\n".join([ctx["text"] for ctx in top_contexts] + [ref["text"]]),
            }
        )
    return out


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
    parser.add_argument("--canonical-retrieval-in", type=Path, default=None)
    parser.add_argument("--out-canonical-retrieval", type=Path, required=True)
    parser.add_argument("--out-soup", type=Path, required=True)
    parser.add_argument("--out-fusion-docs", type=Path, required=True)
    parser.add_argument("--out-cache-pt", type=Path, default=None)
    parser.add_argument("--out-stats", type=Path, required=True)
    parser.add_argument("--skip-cache-export", action="store_true")
    args = parser.parse_args()

    source_paths = parse_csv_paths(args.source_jsonl_csv)
    if not source_paths:
        raise ValueError("--source-jsonl-csv did not contain any paths")

    canonical_chunks = canonical_retrieval_chunks(source_paths, args.chunk_target_tokens)
    if args.canonical_retrieval_in is not None:
        canonical_retrieval = load_jsonl(args.canonical_retrieval_in)
    else:
        canonical_retrieval = build_canonical_retrieval(
            chunks=canonical_chunks,
            top_k=args.top_k,
            retrieval_mode=args.retrieval_mode,
            embedding_model_path=args.embedding_model_path,
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size,
        )
    write_jsonl(args.out_canonical_retrieval, canonical_retrieval)

    runtime_by_entity = target_runtime_chunks(canonical_chunks=canonical_chunks, model_path=args.model_path)
    fusion_docs = fusion_docs_from_canonical(canonical_retrieval, runtime_by_entity)
    write_jsonl(args.out_fusion_docs, fusion_docs)
    write_jsonl(
        args.out_soup,
        [
            {
                "id": f"chunk_{idx}",
                "cid": row["cid"],
                "canonical_cid": row["canonical_cid"],
                "entity_id": row["entity_id"],
                "title": row["title"],
                "box_office_musd": row["box_office_musd"],
                "contents": row["text"],
                "tokens": row["tokens"],
            }
            for idx, row in enumerate(runtime_by_entity.values())
        ],
    )

    if args.skip_cache_export:
        export_stats = {"cache_entries": 0.0, "skipped_entries": 0.0, "cache_export_skipped": True}
    else:
        if args.out_cache_pt is None:
            raise ValueError("--out-cache-pt is required unless --skip-cache-export is set")
        export_stats = export_contextualized_cache(
            fusion_docs=fusion_docs,
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            use_eager=False,
            out_cache_pt=args.out_cache_pt,
        )
        export_stats["cache_export_skipped"] = False

    stats = {
        "source_jsonl": [str(path) for path in source_paths],
        "model_path": args.model_path,
        "top_k": args.top_k,
        "retrieval_mode": args.retrieval_mode,
        "embedding_model_path": args.embedding_model_path,
        "embedding_device": args.embedding_device,
        "embedding_batch_size": args.embedding_batch_size,
        "chunk_target_tokens": args.chunk_target_tokens,
        "canonical_text_mode": "long_movie_chunk_count_tokens_none",
        "corpus_films": len(canonical_chunks),
        "out_canonical_retrieval": str(args.out_canonical_retrieval),
        "out_soup": str(args.out_soup),
        "out_fusion_docs": str(args.out_fusion_docs),
        "out_cache_pt": str(args.out_cache_pt) if args.out_cache_pt else None,
        **export_stats,
    }
    args.out_stats.parent.mkdir(parents=True, exist_ok=True)
    args.out_stats.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(f"retrieval_mode={args.retrieval_mode}")
    print(f"canonical_text_mode={stats['canonical_text_mode']}")
    print(f"corpus_films={len(canonical_chunks)}")
    print(f"cache_entries={int(export_stats['cache_entries'])} skipped_entries={int(export_stats['skipped_entries'])}")
    print(f"cache_export_skipped={bool(export_stats['cache_export_skipped'])}")
    print(f"canonical_retrieval={args.out_canonical_retrieval}")
    print(f"fusion_docs={args.out_fusion_docs}")


if __name__ == "__main__":
    main()
