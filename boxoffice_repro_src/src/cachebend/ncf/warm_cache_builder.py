#!/usr/bin/env python3
"""Build warm-cache .pt files from benchmark query files.

Pipeline:
1) collect all unique chunks in benchmark files (soup)
2) build FusionRAG docs with top-k retrieval:
   - embedding mode (OG): sentence-transformer + faiss
   - lexical mode (fast fallback): token overlap
3) prefill target model and export contextualized CachedChunk dictionary (.pt)
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from tqdm import tqdm

from cachebend.ncf.cutils import (
    CachedChunk,
    build_model_eager,
    build_model_sdpa,
    build_tokenizer,
    chash,
    chunks_from_tokenss,
    ensure_tokenizer_model_alignment,
    split_prompt_for_warm_chunks,
    to_str_prompt,
)

DOSSIER_RE = re.compile(
    r"(BENCHMARK_DOSSIER\s+ENTITY_ID:[\s\S]*?NOTE:\s*Use only these fields for QA\.\s*Ignore any world knowledge\.)",
    flags=re.IGNORECASE,
)


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_-]+", text.lower())


def _apply_cpu_thread_caps(num_threads: int = 1) -> None:
    thread_value = str(max(1, num_threads))
    for env_name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(env_name, thread_value)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        torch.set_num_threads(max(1, num_threads))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, num_threads))
    except Exception:
        pass


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_query_files(arg_list: List[str]) -> List[Path]:
    out: List[Path] = []
    for item in arg_list:
        for token in item.split(","):
            token = token.strip()
            if token:
                out.append(Path(token))
    return out


def _atomic_write_bytes(writer, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp-{os.getpid()}")
    try:
        with tmp_path.open("wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    def _writer(handle) -> None:
        handle.write(json.dumps(payload, indent=2).encode("utf-8"))

    _atomic_write_bytes(_writer, path)


def _atomic_write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    def _writer(handle) -> None:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True).encode("utf-8"))
            handle.write(b"\n")

    _atomic_write_bytes(_writer, path)


def load_chunk_soup_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for row in load_jsonl(path):
        rows.append(
            {
                "cid": str(row.get("cid", "")),
                "tokens": [int(x) for x in row.get("tokens", [])],
                "text": str(row.get("contents") or row.get("text") or ""),
            }
        )
    return rows


def load_fusion_docs_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for row in load_jsonl(path):
        normalized_ctxs = []
        for ctx in row.get("top_k_contexts", []):
            normalized_ctxs.append(
                {
                    "cid": str(ctx.get("cid", "")),
                    "text": str(ctx.get("text") or ""),
                    "tokens": [int(x) for x in ctx.get("tokens", [])],
                }
            )
        rows.append(
            {
                "reference_context": str(row.get("reference_context") or ""),
                "reference_cid": str(row.get("reference_cid") or ""),
                "reference_tokens": [int(x) for x in row.get("reference_tokens", [])],
                "top_k_contexts": normalized_ctxs,
                "concatenated_context": str(row.get("concatenated_context") or ""),
            }
        )
    return rows


def can_reuse_existing_artifacts(
    stats_path: Path,
    out_soup: Path,
    out_fusion_docs: Path,
    query_files: List[Path],
    model_path: str,
    top_k: int,
    retrieval_mode: str,
    embedding_model_path: str,
    single_prompt: bool,
) -> bool:
    if not (stats_path.exists() and out_soup.exists() and out_fusion_docs.exists()):
        return False
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected_query_files = [str(p) for p in query_files]
    actual_query_files = [str(p) for p in (stats.get("query_files") or [])]
    if sorted(actual_query_files) != sorted(expected_query_files):
        return False
    if str(stats.get("model_path")) != model_path:
        return False
    if int(stats.get("top_k", -1)) != top_k:
        return False
    if str(stats.get("retrieval_mode")) != retrieval_mode:
        return False
    if bool(stats.get("single_prompt", True)) != bool(single_prompt):
        return False
    if retrieval_mode == "embedding" and str(stats.get("embedding_model_path")) != embedding_model_path:
        return False
    return True


def extract_unique_chunks_from_query_files(query_files: List[Path]) -> List[str]:
    # Legacy path (regex on dossier text). Kept for fallback/debug.
    chunks: Dict[str, None] = {}
    for qf in query_files:
        rows = load_jsonl(qf)
        for row in rows:
            t1 = row.get("turn_1_poison_prompt", "")
            for m in DOSSIER_RE.finditer(t1):
                chunks[m.group(1).strip()] = None
    return list(chunks.keys())


def extract_unique_runtime_chunks(
    query_files: List[Path],
    tokenizer,
    single_prompt: bool,
) -> List[Dict[str, Any]]:
    """
    Extract unique chunk units exactly as runtime does:
    - tokenize prompt string
    - split via chunks_from_tokenss(..., sep)
    - key by CID
    """
    sep_tok = tokenizer.sep_token if tokenizer.sep_token else "<DSEP>"
    sep_ids = tokenizer(sep_tok, add_special_tokens=False).input_ids
    if not sep_ids:
        raise RuntimeError(f"Could not tokenize sep token '{sep_tok}'")
    sep = torch.tensor(sep_ids, dtype=torch.long)

    by_cid: Dict[str, Dict[str, Any]] = {}
    ascii_sep = tokenizer.sep_token if tokenizer.sep_token else "<DSEP>"

    def _segments_from_row(row: Dict[str, Any]) -> List[str]:
        prompt_segments = row.get("prompt_segments")
        if isinstance(prompt_segments, list) and prompt_segments:
            return [str(seg).strip() for seg in prompt_segments if str(seg).strip()]
        prompt_text = f"{row['turn_1_poison_prompt']}\n\n{row['turn_2_eval_prompt']}"
        segments = [s.strip() for s in prompt_text.split("\n\n") if s.strip()]
        if len(segments) <= 1:
            segments = [prompt_text.strip(), "Provide the requested output format only."]
        return segments

    def _prompt_for_row(row: Dict[str, Any]) -> str:
        segments = _segments_from_row(row)
        if single_prompt:
            return to_str_prompt(tokenizer, ascii_sep, segments)
        poison_only = segments[:-1] if len(segments) > 1 else segments
        return to_str_prompt(tokenizer, ascii_sep, poison_only)

    for qf in query_files:
        rows = load_jsonl(qf)
        for row in rows:
            prompt = _prompt_for_row(row)
            ids = tokenizer(prompt, add_special_tokens=False).input_ids
            tokens = torch.tensor(ids, dtype=torch.long)
            chunks = split_prompt_for_warm_chunks(tokens, sep, tokenizer)
            if chunks:
                chunks = chunks[1:]  # exclude the system/chat prefix from warm soup
            if len(chunks) > 1:
                chunks = chunks[:-1]  # exclude the trailing query chunk
            else:
                chunks = []
            for c in chunks:
                cid = str(c.cid)
                if cid not in by_cid:
                    by_cid[cid] = {
                        "cid": cid,
                        "tokens": c.tokens.tolist(),
                        "text": tokenizer.decode(c.tokens, skip_special_tokens=False),
                    }
    return list(by_cid.values())


def build_topk_contexts_lexical(chunks: List[Dict[str, Any]], top_k: int) -> List[dict]:
    texts = [c["text"] for c in chunks]
    token_sets = [set(_tokenize_words(c)) for c in texts]
    out = []
    for i, ref in enumerate(chunks):
        scores: List[Tuple[int, int]] = []
        a = token_sets[i]
        for j, _ in enumerate(texts):
            if i == j:
                continue
            inter = len(a & token_sets[j])
            scores.append((inter, j))
        scores.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        top = [chunks[j] for _, j in scores[:top_k]]
        out.append(
            {
                "reference_context": ref["text"],
                "reference_cid": ref["cid"],
                "reference_tokens": ref["tokens"],
                "top_k_contexts": top,
                "concatenated_context": "\n\n".join([x["text"] for x in top] + [ref["text"]]) if top else ref["text"],
            }
        )
    return out


def build_topk_contexts_embedding(
    chunks: List[Dict[str, Any]],
    top_k: int,
    embedding_model_path: str,
    embedding_device: str,
    embedding_batch_size: int,
) -> List[dict]:
    _apply_cpu_thread_caps(1)
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            "Embedding retrieval requested but sentence-transformers/faiss/numpy are unavailable"
        ) from exc

    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass

    model = SentenceTransformer(embedding_model_path, device=embedding_device)
    texts = [c["text"] for c in chunks]
    safe_batch_size = max(1, embedding_batch_size)
    if len(texts) >= 512:
        safe_batch_size = min(safe_batch_size, 8)
    emb = model.encode(
        texts,
        batch_size=safe_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    del model
    gc.collect()

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    D, I = index.search(emb, top_k + 1)  # +1 for self

    out = []
    for i, ref in enumerate(chunks):
        nbrs = []
        for idx in I[i]:
            idx = int(idx)
            if idx == i or idx < 0 or idx >= len(texts):
                continue
            nbrs.append(chunks[idx])
            if len(nbrs) >= top_k:
                break
        out.append(
            {
                "reference_context": ref["text"],
                "reference_cid": ref["cid"],
                "reference_tokens": ref["tokens"],
                "top_k_contexts": nbrs,
                "concatenated_context": "\n\n".join([x["text"] for x in nbrs] + [ref["text"]]) if nbrs else ref["text"],
            }
        )
    return out


def _find_subsequence(full_ids: List[int], sub_ids: List[int]) -> int:
    n = len(full_ids)
    m = len(sub_ids)
    if m == 0 or m > n:
        return -1
    for i in range(n - m, -1, -1):
        if full_ids[i : i + m] == sub_ids:
            return i
    return -1


def export_contextualized_cache(
    fusion_docs: List[dict],
    model_path: str,
    device: str,
    dtype: str,
    use_eager: bool,
    out_cache_pt: Path,
) -> Dict[str, float]:
    if use_eager:
        llm = build_model_eager(model_path)
    else:
        try:
            llm = build_model_sdpa(model_path, dtype)
        except TypeError:
            llm = build_model_sdpa(model_path)
    tokenizer = build_tokenizer(model_path)
    ensure_tokenizer_model_alignment(llm, tokenizer)
    llm = llm.to(device).eval()

    sep_token = tokenizer.sep_token if tokenizer.sep_token else tokenizer.eos_token
    if not sep_token:
        sep_token = "\n"

    prefix = "Answer the question based ONLY on the provided context. Be concise — output just the answer, no explanation."
    dummy_query = "Question: X"
    export: Dict[str, CachedChunk] = {}
    skipped = 0
    forward_kwargs = {}
    try:
        if "logits_to_keep" in inspect.signature(llm.forward).parameters:
            # Qwen3 defaults to full-sequence logits when logits_to_keep=0.
            # Warm-cache export only needs past_key_values.
            forward_kwargs["logits_to_keep"] = 1
    except Exception:
        pass
    for row in tqdm(fusion_docs, desc="Building warm cache"):
        ref_ids = [int(x) for x in row.get("reference_tokens", [])]
        if not ref_ids:
            ref_text = row.get("reference_context", "")
            ref_ids = tokenizer(ref_text, add_special_tokens=False).input_ids
        ref_tensor = torch.tensor(ref_ids, dtype=torch.long, device="cpu")
        ref_cid = row.get("reference_cid") or chash(ref_tensor)

        parts: List[str] = [prefix]
        for ctx in row.get("top_k_contexts", []):
            ctx_text = str(ctx.get("text", ""))
            ctx_tokens = [int(x) for x in ctx.get("tokens", [])]
            if not ctx_tokens:
                ctx_tokens = tokenizer(ctx_text, add_special_tokens=False).input_ids
            parts.append(ctx_text if ctx_text else tokenizer.decode(ctx_tokens, skip_special_tokens=False))
        ref_text = str(row.get("reference_context") or "")
        parts.extend([ref_text if ref_text else tokenizer.decode(ref_ids, skip_special_tokens=False), dummy_query])

        prompt = to_str_prompt(tokenizer, sep_token, parts)
        input_seq = tokenizer(prompt, add_special_tokens=False).input_ids
        input_ids = torch.tensor([input_seq], dtype=torch.long, device=device)
        full_ids = input_ids[0].tolist()
        start = _find_subsequence(full_ids, ref_ids)
        if start < 0:
            skipped += 1
            continue
        end = start + len(ref_ids)

        with torch.no_grad():
            outputs = llm(input_ids=input_ids, use_cache=True, **forward_kwargs)
        past = outputs.past_key_values

        states = []
        for k, v in past:
            if k.dim() == 5 and k.shape[1] == 1:
                k = k.squeeze(1)
            if v.dim() == 5 and v.shape[1] == 1:
                v = v.squeeze(1)
            k_slice = k[0, :, start:end, :].to("cpu")
            v_slice = v[0, :, start:end, :].to("cpu")
            states.append((k_slice, v_slice))

        # Warm methods consume start/end as the original prompt positions for RoPE.
        # Preserve the true offsets here to match the old_src cache format.
        export[str(ref_cid)] = CachedChunk(tokens=ref_tensor, start=start, end=end, states=states)

        del outputs, past, input_ids, states
        gc.collect()
        if hasattr(torch, "npu"):
            try:
                torch.npu.empty_cache()
            except Exception:
                pass

    def _writer(handle) -> None:
        torch.save(export, handle)

    _atomic_write_bytes(_writer, out_cache_pt)
    return {"cache_entries": float(len(export)), "skipped_entries": float(skipped)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build warm cache .pt from benchmark query files.")
    ap.add_argument(
        "--queries",
        default="",
        help="Deprecated alias for --query-files (single path or comma-separated paths).",
    )
    ap.add_argument(
        "--query-files",
        nargs="+",
        default=[],
        help="One or more query JSONL files (space or comma separated).",
    )
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"])
    ap.add_argument("--use-eager", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument(
        "--retrieval-mode",
        choices=["embedding", "lexical"],
        default="embedding",
        help="embedding = OG flow (small model + FAISS), lexical = fast fallback",
    )
    ap.add_argument("--embedding-model-path", default="intfloat/e5-base-v2")
    ap.add_argument("--embedding-device", default="cpu")
    ap.add_argument("--embedding-batch-size", type=int, default=32)
    ap.add_argument(
        "--single-prompt",
        action="store_true",
        default=True,
        help="Match pilot_eval --single-prompt chunk extraction (default true).",
    )
    ap.add_argument("--out-soup", type=Path, default=Path("data/phase3_unique_chunk_soup.jsonl"))
    ap.add_argument("--out-fusion-docs", type=Path, default=Path("data/phase3_fusionrag_docs_topk5.jsonl"))
    ap.add_argument("--out-cache-pt", type=Path, required=True)
    ap.add_argument("--out-stats", type=Path, default=Path("results/phase3_warm_cache_build_stats.json"))
    ap.add_argument(
        "--reuse-existing-artifacts",
        action="store_true",
        help="Reuse existing soup/fusion-doc artifacts when the previous stats manifest matches this request.",
    )
    args = ap.parse_args()

    raw_query_files = list(args.query_files)
    if args.queries:
        raw_query_files.append(args.queries)
    query_files = parse_query_files(raw_query_files)
    if not query_files:
        raise ValueError("No query files provided. Use --query-files (or deprecated --queries).")
    artifact_reused = False
    if args.reuse_existing_artifacts and can_reuse_existing_artifacts(
        stats_path=args.out_stats,
        out_soup=args.out_soup,
        out_fusion_docs=args.out_fusion_docs,
        query_files=query_files,
        model_path=args.model_path,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        embedding_model_path=args.embedding_model_path,
        single_prompt=args.single_prompt,
    ):
        unique_chunks = load_chunk_soup_jsonl(args.out_soup)
        fusion_docs = load_fusion_docs_jsonl(args.out_fusion_docs)
        artifact_reused = True
    else:
        unique_chunks = extract_unique_runtime_chunks(
            query_files=query_files,
            tokenizer=build_tokenizer(args.model_path),
            single_prompt=args.single_prompt,
        )

        if args.retrieval_mode == "embedding":
            fusion_docs = build_topk_contexts_embedding(
                unique_chunks,
                top_k=args.top_k,
                embedding_model_path=args.embedding_model_path,
                embedding_device=args.embedding_device,
                embedding_batch_size=args.embedding_batch_size,
            )
        else:
            fusion_docs = build_topk_contexts_lexical(unique_chunks, top_k=args.top_k)

        _atomic_write_jsonl(
            args.out_soup,
            (
                {"id": f"chunk_{i}", "cid": row["cid"], "contents": row["text"], "tokens": row["tokens"]}
                for i, row in enumerate(unique_chunks)
            ),
        )
        _atomic_write_jsonl(args.out_fusion_docs, fusion_docs)

    export_stats = export_contextualized_cache(
        fusion_docs=fusion_docs,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        use_eager=args.use_eager,
        out_cache_pt=args.out_cache_pt,
    )

    stats = {
        "query_files": [str(p) for p in query_files],
        "model_path": args.model_path,
        "top_k": args.top_k,
        "retrieval_mode": args.retrieval_mode,
        "embedding_model_path": args.embedding_model_path,
        "embedding_batch_size_requested": args.embedding_batch_size,
        "single_prompt": bool(args.single_prompt),
        "artifact_reused": artifact_reused,
        "unique_chunks_in_soup": len(unique_chunks),
        "fusion_docs": len(fusion_docs),
        "out_soup": str(args.out_soup),
        "out_fusion_docs": str(args.out_fusion_docs),
        "out_cache_pt": str(args.out_cache_pt),
        **export_stats,
    }
    _atomic_write_json(args.out_stats, stats)

    print(f"retrieval_mode={stats['retrieval_mode']}")
    print(f"artifact_reused={str(stats['artifact_reused']).lower()}")
    print(f"unique_chunks_in_soup={stats['unique_chunks_in_soup']}")
    print(f"cache_entries={int(stats['cache_entries'])} skipped_entries={int(stats['skipped_entries'])}")
    print(f"saved_cache={args.out_cache_pt}")


if __name__ == "__main__":
    main()
