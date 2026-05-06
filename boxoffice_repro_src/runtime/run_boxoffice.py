#!/usr/bin/env python3
"""
Run the BoxOffice KV-cache evaluation pipeline used in the paper.

BoxOffice is a synthetic benchmark designed to defeat top-K retrieval:
each query lists 10 movie dossiers and asks the model to find the one
with the maximum BOX_OFFICE_MUSD. Because the answer requires reading
*every* candidate's value, no semantic top-K retriever can shortcut
the work — coverage at K=10 is ~0.1 by construction (and that's the
intended property, not a bug).

Methods (per CLAUDE.md glossary, baseline ≡ BaselineNosep):
    baseline         BaselineNosep                 full prefill, DSEP stripped
    cb_k0            CacheBlend, top_k=0           no infused neighbours
    cb_k5            CacheBlend, top_k=5           top-5 corpus-wide
                                                    semantic neighbours per
                                                    chunk
    cb_k5q           FusionRAG,  top_k=5           same neighbour set, but
                                                    query-driven recomp
                                                    selection
    cb_k10           CacheBlend, top_k=10          top-10 corpus-wide
                                                    semantic neighbours per
                                                    chunk
    cb_k10q          FusionRAG,  top_k=10          same neighbour set, but
                                                    query-driven recomp
                                                    selection
    ccv3_m{M}_diffkv CacheCraft V3 + CacheBlend    β-overlap variant pick
                     DiffKV blend, M variants/cid  (no attention matrix);
                                                    fast SDPA backend; no
                                                    neighbour fusion. M ∈ {1,2}
                                                    explores the multi-version
                                                    benefit.
    ccv3_m{M}_q      CacheCraft V3 + FR-style      β-overlap variant pick →
                     query-guided token selection  query · final-layer-K of
                                                    the picked variant → top-R%
                                                    indices → blend.

Default recompute ratio: 0.15 (override with --recomp-ratios).

Inputs:
    Queries: <BOXOFFICE_DIR>/<...>_s<seed>_full.jsonl
             (warmup queries first, eval queries after — eval rows have
              the same `query_id` as the corresponding rows in
              `..._s<seed>_eval.jsonl`.)
    Eval-id set:
             <BOXOFFICE_DIR>/<...>_s<seed>_eval.jsonl
             Used only to mark which `_full` rows are eval (`is_eval=true`)
             vs warmup (`is_eval=false`) in the per-query output. F1 is
             computed for both.
    Corpus:  supplied separately to `build_pool.py` from the BoxOffice bundle.
    Default <BOXOFFICE_DIR> = ../boxoffice_repro_src/seeds

Warm vs cold:
    `ccv3_*` methods are stateful — their KV-chunk cache persists across
    queries within a single (model, seed) process. The warmup queries in
    `_full.jsonl` populate that cache before the eval queries run. CB / FR
    methods are stateless per query (their on-the-fly managers reset).
    Because of this, **query-sharding (--num-shards K, K>1) is forbidden
    when any V3 method is in METHODS** — a sharded worker would only see
    a strided subset of queries and fail to accumulate the warm cache.

Each row is a query with `prompt_segments` of length 12:
    seg[0]      instruction
    seg[1..10]  the 10 movie dossiers
    seg[11]     "Question: …"

Output (under --out-dir):
    boxoffice_s<seed>_<model_short>.json   per-query results
                                            (or .shard<i>of<K>.json when
                                             sharding; see aggregate_shards.py)

Sharding:
    pass `--num-shards K --shard i` to process every K-th query on this
    worker. The companion `run.sh` distributes the cross product
    (seed × model × shard) across an NPU pool with one job per NPU.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import random
import re
import string
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.basicConfig(level=logging.INFO)
for n in ("blender", "cacheblend", "onthefly", "cache_log"):
    logging.getLogger(n).setLevel(logging.WARNING)

import numpy as np
import torch
try: import torch_npu  # noqa
except ImportError: pass

# ── Path setup ────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_METHODS_SRC = ROOT_DIR.parent / "cache_methods_src"
BENCHMARK_SRC = Path(
    os.environ.get("CACHE_METHODS_SRC", str(DEFAULT_METHODS_SRC))
)
if not BENCHMARK_SRC.exists():
    raise SystemExit(
        "methods bundle not found. Set CACHE_METHODS_SRC to the root of "
        f"cache_methods_src: {BENCHMARK_SRC}"
    )
if str(BENCHMARK_SRC) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SRC))

from cachebend.ncf.cutils import (                # noqa: E402
    build_model_sdpa, build_tokenizer,
    split_prompt_for_warm_chunks, to_str_prompt,
)
from cachebend.ncf.onthefly_infusion import (    # noqa: E402
    CacheBlendOnTheFlyManager, FusionRAGOnTheFlyManager,
)
from cachebend.ncf.cc_cache import (              # noqa: E402
    CacheCraftCacheV3,
    CacheCraftCacheV3DiffKVManager,
    CacheCraftCacheV3QManager,
)
from cachebend.ncf.zcf_v10 import BaselineNosep   # noqa: E402

OUTPUT_DIR    = SCRIPT_DIR / "output"
DEFAULT_BOXOFFICE_DIR = Path(
    os.environ.get("BOXOFFICE_DIR",
                   str(ROOT_DIR.parent / "boxoffice_repro_src" / "seeds")))
DEFAULT_DATASET_STEM = (
    "boxoffice_filtered_v4_balanced_m4_s{seed}_full.jsonl")
DEFAULT_EVAL_STEM = (
    "boxoffice_filtered_v4_balanced_m4_s{seed}_eval.jsonl")
DEFAULT_SEEDS = (7, 11, 13, 17, 19, 23, 29, 31, 47, 73)
RECOMP_RATIOS = [0.0, 0.05, 0.15]

# Cross-seed corpus-wide pool — built once by build_pool.py.
DEFAULT_POOL_DIR = SCRIPT_DIR / "pool"

# ── Method grid ───────────────────────────────────────────────────────
# (key, manager_kind, top_k, neighbour_mode)
#   manager_kind  ∈ {"cb", "fr",
#                    "ccv3_diffkv_m{1,2,3,4}", "ccv3_q_m{1,2,3,4}"}
#     cb / fr            — CacheBlend / FusionRAG on-the-fly with K-neighbour
#                          fusion (top_k drives the neighbour count).
#     ccv3_diffkv_m{M}   — CacheCraft V3 (β-overlap selection, no attention
#                          matrix) + CacheBlend DiffKV blend, M variants per
#                          chunk_id. No neighbour fusion (top_k ignored).
#     ccv3_q_m{M}        — CacheCraft V3 + FR-style query-guided token
#                          selection. M variants per chunk_id.
#   neighbour_mode ∈ {"none", "semantic_pool"}
#     none           — no infused neighbours
#     semantic_pool  — top-K from the corpus-wide pool built by
#                      build_pool.py. Same neighbour-source policy as
#                      paper2/exp/intro/cb_ksem10q and
#                      paper2/exp/fr_extended on v8/v9: cross-query,
#                      whole-corpus e5 KNN. NOT "9 other passages of
#                      the same query" — that's never how the rest of
#                      paper2 works.
METHODS: list[tuple[str, str, int, str]] = [
    ("cb_k0",          "cb",                0, "none"),
    ("cb_k0q",         "fr",                0, "none"),
    ("cb_k5",          "cb",                5, "semantic_pool"),
    ("cb_k5q",         "fr",                5, "semantic_pool"),
    ("ccv3_m1_diffkv", "ccv3_diffkv_m1",    0, "none"),
    ("ccv3_m1_q",      "ccv3_q_m1",         0, "none"),
    ("ccv3_m2_diffkv", "ccv3_diffkv_m2",    0, "none"),
    ("ccv3_m2_q",      "ccv3_q_m2",         0, "none"),
    ("ccv3_m3_diffkv", "ccv3_diffkv_m3",    0, "none"),
    ("ccv3_m3_q",      "ccv3_q_m3",         0, "none"),
    ("ccv3_m4_diffkv", "ccv3_diffkv_m4",    0, "none"),
    ("ccv3_m4_q",      "ccv3_q_m4",         0, "none"),
]
# Methods whose presence implies query-sharding (--num-shards K, K>1) is
# forbidden, because their cache state must accumulate sequentially over
# the warmup→eval order in `_full.jsonl`. Sharding strides through that
# order and breaks the warm-cache benefit.
STATEFUL_METHODS = frozenset(
    m[0] for m in METHODS if m[1].startswith("ccv3_")
)


# ── F1 helpers (token-level over normalised strings) ─────────────────
def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def compute_f1(pred: str, gold) -> float:
    if isinstance(gold, list):
        return max(compute_f1(pred, g) for g in gold) if gold else 0.0
    p = _normalize(pred).split()
    g = _normalize(str(gold)).split()
    if not g: return float(not p)
    if not p: return 0.0
    common = collections.Counter(p) & collections.Counter(g)
    nc = sum(common.values())
    if nc == 0: return 0.0
    prec = nc / len(p); rec = nc / len(g)
    return 2 * prec * rec / (prec + rec)


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text


def _empty_cache():
    gc.collect()
    if hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Prompt builders ───────────────────────────────────────────────────
def _segments_for_row(row: dict) -> list[str]:
    """Use the original boxoffice prompt_segments verbatim. They're laid
    out as [instruction, dossier_1, …, dossier_10, "Question: …"], which
    is exactly the [prefix, …chunks…, suffix] shape that the cb / fr
    managers (via `split_prompt_for_warm_chunks`) expect."""
    segs = row.get("prompt_segments")
    if segs and len(segs) >= 3:
        return list(segs)
    # Fallback: synthesise from passages + question if prompt_segments is
    # missing (defensive — every reference_inputs row carries it).
    instr = (row.get("metadata", {}) or {}).get("instruction_chunk") or \
        "Use only the synthetic movie dossiers below."
    passages = row.get("passages") or [{"text": t} for t in row.get("ctxs", [])]
    return [instr] + [p["text"] for p in passages] + \
           [f"Question: {row.get('question','')}"]


def build_cb_prompt_tensor(segments: list[str], tokenizer):
    """Prompt for cb / fr managers: insert tokenizer.sep_token between
    segments so split_prompt_for_warm_chunks can identify chunk
    boundaries. Trailing "\\n\\n" on every non-last segment so the
    post-DSEP-strip token stream is byte-identical to baseline's."""
    sep = tokenizer.sep_token
    segs = [s + "\n\n" for s in segments[:-1]] + [segments[-1]]
    prompt_str = to_str_prompt(tokenizer, sep, segs)
    ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
    return torch.tensor(ids, dtype=torch.long)


def build_baseline_prompt_tensor(segments: list[str], tokenizer):
    """Prompt for BaselineNosep.

    BaselineNosep CONSUMES the cb/fr-shaped prompt directly: it expects
    the chat template already applied (so the model sees instruct
    formatting) AND the DSEP markers in place — internally it splits
    on DSEP and strips them before the prefill, leaving a single
    chat-templated, DSEP-free token stream. So we just hand it the
    cb/fr prompt; the model sees the same content as the cb/fr methods,
    and `baseline.f1` is directly comparable to `cb_*.f1`.

    A previous version of this builder did `"\\n\\n".join(segments)` and
    tokenized raw text — no chat template. On instruct-tuned models
    (qwen3 / mistral / llama3.1-instruct) that produces garbage and
    drives `baseline.f1` to 0 across the board. Don't do that."""
    return build_cb_prompt_tensor(segments, tokenizer)


def run_baseline(baseline, prompt_ids, max_tokens=16):
    baseline.begin_fresh_query()
    out, _, _ = baseline.new_query(prompt_ids, max_new_tokens=max_tokens)
    return _strip_thinking(str(out)).strip()


# ── Cross-seed semantic neighbour pool ───────────────────────────────
import hashlib  # noqa: E402  (top-level imports already happened)

def _md5_500(text: str) -> str:
    return hashlib.md5((text or "")[:500].encode("utf-8",
                                                  errors="ignore")
                       ).hexdigest()[:16]


class BoxOfficePoolIndex:
    """Per-seed lookup table: passage text → top-K semantic neighbour
    TEXTS, drawn from the corpus-wide pool of THAT seed's eval rows.

    Mirrors LB / v9 exactly: one pool per dataset, no cross-dataset
    mixing. For LB, "dataset" is `<ds>_lb` and the artefacts live at
    `paper/exp/offline_limit/output/lb/<ds>_lb_*`. Here, "dataset" is
    one boxoffice seed and the artefacts live at
    `paper2/exp/boxoffice/pool/boxoffice_s<seed>_*`. Same KNN
    construction (square self-pool top-K via
    `paper2/exp/offline_limit/_lib.topk_neighbours`, self excluded);
    same dispatch (text → md5(text[:500]) → chunk_id → neighbour ids
    → neighbour texts).

    Reads the artefacts produced by `build_pool.py`:
      - boxoffice_s<seed>_chunks.json      (id → {title, text, …})
      - boxoffice_s<seed>_text_md5.json    (md5(text[:500]) → chunk_id)
      - boxoffice_s<seed>_infusion_k10.json
            (str(chunk_id) → [[neighbour_id, score], …×K])

    Doesn't load the embeddings — runtime is pure text-lookup.
    """

    def __init__(self, pool_dir: Path, seed: int, k: int = 10):
        self.seed = seed
        self.k = k
        chunks_p = pool_dir / f"boxoffice_s{seed}_chunks.json"
        md5_p    = pool_dir / f"boxoffice_s{seed}_text_md5.json"
        inf_p    = pool_dir / f"boxoffice_s{seed}_infusion_k{k}.json"
        for p in (chunks_p, md5_p, inf_p):
            if not p.exists():
                raise FileNotFoundError(
                    f"missing pool file for s{seed}: {p}\n"
                    f"build it first with:\n"
                    f"  python boxoffice/build_pool.py "
                    f"--seeds {seed} --device-ids 0 1 2 3 --pack 2"
                )
        chunks = json.loads(chunks_p.read_text())
        self.cid_to_text: dict[int, str] = {
            int(c["chunk_id"]): c["text"] for c in chunks
        }
        self.md5_to_cid: dict[str, int] = {
            k_: int(v) for k_, v in json.loads(md5_p.read_text()).items()
        }
        raw = json.loads(inf_p.read_text())
        # JSON keys come back as strings; coerce to int.
        self.infusion: dict[int, list[list]] = {
            int(k_): v for k_, v in raw.items()
        }

    def neighbours_for_text(self, text: str) -> list[str]:
        """Return up to `self.k` neighbour texts for the chunk whose
        text matches `text` (by md5 of the first 500 chars)."""
        cid = self.md5_to_cid.get(_md5_500(text))
        if cid is None:
            return []
        out: list[str] = []
        for nid, _score in self.infusion.get(cid, [])[: self.k]:
            t = self.cid_to_text.get(int(nid))
            if t is not None:
                out.append(t)
        return out


def build_pool_neighbors(row: dict, prompt_tensor, tokenizer,
                         index: "BoxOfficePoolIndex"):
    """Per-chunk neighbours = top-K from the corpus-wide pool, aligned
    positionally to the row's passages."""
    sep_tok = tokenizer.sep_token or "<DSEP>"
    sep_ids = tokenizer(sep_tok, add_special_tokens=False).input_ids
    sep = torch.tensor(sep_ids, dtype=torch.long)
    chunks = split_prompt_for_warm_chunks(prompt_tensor, sep, tokenizer)
    doc_chunks = chunks[1:-1]

    segs = _segments_for_row(row)
    chunk_texts = segs[1:-1]
    n = len(chunk_texts)

    cid_to_neighbors: dict[str, list[str]] = {}
    for i, chunk in enumerate(doc_chunks):
        if i >= n:
            cid_to_neighbors[str(chunk.cid)] = []
            continue
        cid_to_neighbors[str(chunk.cid)] = index.neighbours_for_text(
            chunk_texts[i])
    return cid_to_neighbors


# A previous version of this runner had `build_all_others_neighbors`,
# which handed each chunk the 9 OTHER passages from the same query as
# its K=10 neighbours. That diverges from how every other paper2
# K-conditioned method works (corpus-wide e5 top-K via offline_limit's
# `topk_neighbours`), so any plot/table that compared boxoffice cb_k10
# to v8/v9/LB cb_ksem10q was apples-vs-oranges. Removed; cb_k10 /
# cb_k10q now exclusively use BoxOfficePoolIndex.


# ── Per-row runner ────────────────────────────────────────────────────
def run_row(row: dict, tokenizer, baseline,
            mgrs: dict, recomp_ratios, max_tokens=16,
            pool_idx: "BoxOfficePoolIndex | None" = None) -> dict:
    qid = row.get("query_id") or row.get("trace_id") or ""
    md = row.get("metadata", {}) or {}
    cell = md.get("scheduled_matrix_cell") or md.get("computed_matrix_cell") or ""

    # Gold answer: keep the entity_id form ("FILM-1009"), not the
    # "Answer=FILM-1009" form, since the model is instructed to output
    # only the FILM-ID. compute_f1 also accepts the list form.
    gold_list = row.get("answers_all") or row.get("answers") or []
    if not gold_list and row.get("answer"):
        gold_list = [row["answer"]]
    if not gold_list and row.get("gold_answer"):
        gold_list = [row["gold_answer"]]

    segments = _segments_for_row(row)
    # One prompt for both baseline and cb/fr — BaselineNosep strips DSEP
    # itself, the cb/fr managers split on it.
    cb_prompt = build_cb_prompt_tensor(segments, tokenizer)

    result = {
        "query_id":     qid,
        "trace_id":     row.get("trace_id") or qid,
        "longbench_id": row.get("longbench_id"),
        "cell":         cell,
        "question":     row.get("question"),
        "gold_answer":  row.get("answer") or row.get("gold_answer"),
        "answers":      gold_list,
        "n_chunks":     len(segments) - 2,
        "n_golden":     len(row.get("golden_chunk_indices") or []),
        "n_tokens":     int(cb_prompt.numel()),
    }

    # baseline
    t0 = time.time()
    try:
        ans = run_baseline(baseline, cb_prompt, max_tokens)
        f1  = compute_f1(ans, gold_list)
    except Exception as exc:
        ans, f1 = f"ERROR: {exc}", 0.0
    result["baseline"] = {"answer": ans, "f1": f1, "time_s": time.time()-t0}
    print(f"    base: f1={f1:.3f}", end="", flush=True)
    _empty_cache()

    # neighbour set — build once per row only if a method needs it.
    needs_pool = any(m[3] == "semantic_pool" for m in METHODS)
    nbrs_pool: dict[str, list[str]] = {}
    if needs_pool and pool_idx is not None:
        nbrs_pool = build_pool_neighbors(row, cb_prompt, tokenizer, pool_idx)

    for key, mgr_kind, top_k, neighbour_mode in METHODS:
        mgr = mgrs[mgr_kind]
        if neighbour_mode == "semantic_pool":
            nbrs = nbrs_pool
        else:
            nbrs = {}
        for R in recomp_ratios:
            r_key = f"{key}_R{R:.2f}"
            t0 = time.time()
            mgr.begin_fresh_query()
            mgr.cache.recomp_ratio = R
            if hasattr(mgr.cache, "selector"):
                mgr.cache.selector.recomp_ratio = R
            # V3-Q stores its FR-style selector on the manager.
            if hasattr(mgr, "_selector"):
                mgr._selector.recomp_ratio = R
            mgr.top_k = top_k
            mgr.configure_query_neighbors(nbrs)
            try:
                output, used_cache, stats = mgr.new_query(cb_prompt)
                ans = _strip_thinking(str(output))
                f1_val = compute_f1(ans, gold_list)
                result[r_key] = {
                    "answer": ans, "f1": f1_val,
                    "time_s": time.time()-t0,
                    "used_cache": used_cache,
                }
            except Exception as exc:
                result[r_key] = {
                    "answer": f"ERROR: {exc}", "f1": 0.0,
                    "time_s": time.time()-t0,
                }
            _empty_cache()

    bits = []
    for key, *_ in METHODS:
        rk = f"{key}_R{recomp_ratios[0]:.2f}"
        if rk in result:
            bits.append(f"{key}={result[rk].get('f1', 0):.2f}")
    print(f"  | R{recomp_ratios[0]:.2f} " + " ".join(bits), flush=True)
    return result


# ── Main ──────────────────────────────────────────────────────────────
def _load_jsonl(p: Path) -> list[dict]:
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--device-type", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS),
                    help=f"seeds to run (default: {list(DEFAULT_SEEDS)}).")
    ap.add_argument("--boxoffice-dir", type=Path, default=DEFAULT_BOXOFFICE_DIR,
                    help=f"directory with the BoxOffice seed jsonl files "
                         f"(default: {DEFAULT_BOXOFFICE_DIR})")
    ap.add_argument("--dataset-stem", default=DEFAULT_DATASET_STEM,
                    help="file stem with {seed} placeholder; defaults to "
                         "the `_full.jsonl` (warmup queries first, then "
                         "eval queries). Override to `_eval.jsonl` to "
                         "skip warmup.")
    ap.add_argument("--eval-stem", default=DEFAULT_EVAL_STEM,
                    help="file stem with {seed} placeholder for the eval "
                         "subset; rows in `--dataset-stem` whose query_id "
                         "matches one in this file are tagged `is_eval=true` "
                         "in the per-query output. Pass an empty string to "
                         "disable (treat all rows as eval).")
    ap.add_argument("--recomp-ratios", nargs="*", type=float,
                    default=RECOMP_RATIOS)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--seed-rng", type=int, default=42,
                    help="RNG seed (controls any random tie-breaks).")
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="If the per-seed output JSON already exists, fold "
                         "this run's per-query method/ratio keys into the "
                         "existing rows (matched by query_id) instead of "
                         "overwriting. Use to assemble a multi-method "
                         "output across separate process invocations — "
                         "needed when V3 methods must run alone for OOM "
                         "safety.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--output-tag", default="",
                    help="When set, the per-seed output file becomes "
                         "boxoffice_s<seed>_<model>.<tag>.json instead of "
                         "the canonical .json. Used by the launcher to "
                         "give each (cell × method-group) worker its own "
                         "destination file so concurrent workers don't "
                         "race on a shared --merge target. The launcher "
                         "runs an aggregator step at the end to merge all "
                         "per-tag files into the canonical one.")
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip warmup queries — process only rows whose "
                         "`query_id` is in the companion `_eval.jsonl`. "
                         "Use for stateless methods (cb/fr) where warmup "
                         "is pure waste because the managers reset per "
                         "query. V3 (ccv3_*) methods need warmup to "
                         "populate their cache, so don't pass this flag "
                         "for V3 runs.")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="restrict to these METHODS keys (e.g. "
                         "'--methods cb_k0 cb_k10').")
    ap.add_argument("--pool-dir", type=Path,
                    default=DEFAULT_POOL_DIR,
                    help="directory holding the corpus-wide pool "
                         "produced by build_pool.py (default: ./pool/). "
                         "Required when any K>0 method is in the grid; "
                         "the runner exits early if it's missing.")
    args = ap.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard < args.num_shards):
        raise SystemExit(
            f"bad --shard/--num-shards: {args.shard}/{args.num_shards}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed_rng)

    if args.methods:
        req = set(args.methods)
        known = {m[0] for m in METHODS}
        unknown = req - known
        if unknown:
            raise SystemExit(
                f"unknown methods: {sorted(unknown)}; known={sorted(known)}")
        METHODS[:] = [m for m in METHODS if m[0] in req]
        print(f"Method filter: {[m[0] for m in METHODS]}")

    # Stateful (V3) methods can't tolerate query-sharding: their cache
    # must accumulate over the full warmup→eval sequence in `_full.jsonl`,
    # and a sharded worker only sees every K-th row.
    active_stateful = {m[0] for m in METHODS} & STATEFUL_METHODS
    if active_stateful and args.num_shards > 1:
        raise SystemExit(
            f"--num-shards={args.num_shards} is incompatible with stateful "
            f"V3 methods {sorted(active_stateful)}. Run with --num-shards=1, "
            f"or restrict to non-V3 methods via --methods.")

    if args.device_type != "auto":
        device = f"{args.device_type}:{args.device_id}"
    elif torch.cuda.is_available():
        device = f"cuda:{args.device_id}"
    elif hasattr(torch, "npu") and torch.npu.is_available():
        device = f"npu:{args.device_id}"
    else:
        device = "cpu"
    mshort = Path(args.model).name

    def out_path_for(seed: int) -> Path:
        base = f"boxoffice_s{seed}_{mshort}"
        if args.output_tag:
            base = f"{base}.{args.output_tag}"
        if args.num_shards == 1:
            return args.out_dir / f"{base}.json"
        return args.out_dir / (
            f"{base}.shard{args.shard}of{args.num_shards}.json")

    # Pre-flight: any seed already done? Skip those (unless --force).
    # "Done" means: the per-seed output exists AND its config records
    # methods + ratios that cover the current request. A prior run with
    # a smaller METHODS list (or fewer ratios) is treated as stale and
    # re-run — drift between the global METHODS / RECOMP_RATIOS and the
    # on-disk config triggers a fresh run automatically.
    requested_methods = {m[0] for m in METHODS}
    requested_ratios = {round(float(r), 6) for r in args.recomp_ratios}

    def _covers(op: Path) -> bool:
        try:
            cfg = json.loads(op.read_text()).get("config", {})
        except Exception:
            return False
        have_methods = set(cfg.get("methods") or [])
        have_ratios = {round(float(r), 6) for r in cfg.get("recomp_ratios") or []}
        return (
            requested_methods.issubset(have_methods)
            and requested_ratios.issubset(have_ratios)
        )

    pending: list[int] = []
    for seed in args.seeds:
        op = out_path_for(seed)
        if op.exists() and not args.force:
            if args.merge:
                # In merge mode the file is the *destination* into which
                # this run's method/ratio keys will be folded — never a
                # reason to skip. (`_covers` would skip if the requested
                # methods are already present; under --merge we still
                # want the ability to add more.)
                pass
            elif _covers(op):
                print(f"  [skip] {op.name}  "
                      f"(covers all requested methods + ratios)")
                continue
            else:
                print(f"  [stale] {op.name}  "
                      f"(prior config missing some methods or ratios; re-running)")
        in_path = args.boxoffice_dir / args.dataset_stem.format(seed=seed)
        if not in_path.exists():
            print(f"  ERROR: missing {in_path}", file=sys.stderr)
            continue
        pending.append(seed)
    if not pending:
        print("Nothing to do.")
        return

    # ── Build model + managers (once) ──────────────────────────────────
    print(f"Model: {args.model}  device={device}  shard={args.shard}/"
          f"{args.num_shards}")
    llm = build_model_sdpa(args.model, torch_dtype=args.dtype).to(device)
    llm.eval()
    tokenizer = build_tokenizer(args.model)
    print("Model loaded.")
    baseline = BaselineNosep(device=device, llm=llm, tkn=tokenizer)

    # Build only the managers the active METHODS need. CB / FR managers
    # are stateless per query (cheap), but each V3 manager carries its
    # own multi-version KV-chunk cache that grows over the warmup→eval
    # sequence — instantiating V3 variants we won't use risks OOM. The
    # launcher should run each V3 method in its own process (with
    # --merge so results fold into the same per-seed JSON).
    def _v3(cache_M: int, kind: str):
        cache = CacheCraftCacheV3(N=12, M=cache_M, R=0.0)
        cls = (CacheCraftCacheV3DiffKVManager if kind == "diffkv"
               else CacheCraftCacheV3QManager)
        return cls(device=device, cache=cache, llm=llm, tkn=tokenizer)

    needed_kinds = {m[1] for m in METHODS}
    mgrs: dict = {}
    if "cb" in needed_kinds:
        mgrs["cb"] = CacheBlendOnTheFlyManager(
            device=device, R=0.0, top_k=0, llm=llm, tkn=tokenizer)
    if "fr" in needed_kinds:
        mgrs["fr"] = FusionRAGOnTheFlyManager(
            device=device, R=0.0, top_k=0, llm=llm, tkn=tokenizer)
    v3_specs = {
        "ccv3_diffkv_m1": (1, "diffkv"),
        "ccv3_diffkv_m2": (2, "diffkv"),
        "ccv3_diffkv_m3": (3, "diffkv"),
        "ccv3_diffkv_m4": (4, "diffkv"),
        "ccv3_q_m1":      (1, "q"),
        "ccv3_q_m2":      (2, "q"),
        "ccv3_q_m3":      (3, "q"),
        "ccv3_q_m4":      (4, "q"),
    }
    for kind, (m_cap, strat) in v3_specs.items():
        if kind in needed_kinds:
            mgrs[kind] = _v3(m_cap, strat)
    print(f"Built managers: {sorted(mgrs.keys())}")

    # Pool is loaded per-seed below (one pool per seed, mirroring the
    # LB convention of one infusion file per dataset). Hard-fail if a
    # K>0 method is in the grid but the pool isn't built — silent
    # empty-list fallback was exactly what masked the previous bug.
    needs_pool = any(m[3] == "semantic_pool" for m in METHODS)

    for seed in pending:
        in_path  = args.boxoffice_dir / args.dataset_stem.format(seed=seed)
        out_path = out_path_for(seed)
        rows = _load_jsonl(in_path)

        # Build the eval-id set from the companion `_eval.jsonl`. Rows in
        # `_full.jsonl` whose `query_id` is in this set are eval; the rest
        # are warmup. F1 is reported for both — downstream plots filter
        # via the `is_eval` field.
        eval_ids: set[str] = set()
        eval_path = (
            args.boxoffice_dir / args.eval_stem.format(seed=seed)
            if args.eval_stem else None
        )
        if eval_path is not None and eval_path.exists():
            for er in _load_jsonl(eval_path):
                qid = er.get("query_id") or er.get("trace_id")
                if qid:
                    eval_ids.add(str(qid))
            print(f"  loaded {len(eval_ids)} eval ids from {eval_path.name}")
        else:
            print(f"  no eval-id file at {eval_path}; treating all rows as eval")

        # Tag every row with its position in the unfiltered file so we
        # can preserve `orig_index` after filtering for --eval-only.
        indexed = list(enumerate(rows))   # [(orig_index, row), …]

        if args.eval_only:
            if not eval_ids:
                print("WARNING: --eval-only set but no eval-id file loaded; "
                      "nothing to do.", file=sys.stderr)
                indexed = []
            else:
                before = len(indexed)
                indexed = [
                    (i, r) for (i, r) in indexed
                    if str(r.get("query_id") or r.get("trace_id")) in eval_ids
                ]
                print(f"  --eval-only: filtered {before} → {len(indexed)} rows")

        # Now shard the (possibly filtered) list. The shard discriminator
        # is position in the filtered list so each shard gets an even
        # slice; the original file index is preserved as `orig_i`.
        shard_items = [(i, r) for j, (i, r) in enumerate(indexed)
                       if j % args.num_shards == args.shard]
        n_eval = sum(
            1 for _, r in shard_items
            if not eval_ids
            or str(r.get("query_id") or r.get("trace_id")) in eval_ids
        )
        n_warm = len(shard_items) - n_eval
        print(f"\n{'='*60}\n  s{seed}: {len(rows)} rows total, "
              f"{len(indexed)} after filter  "
              f"(this shard: {len(shard_items)} = "
              f"{n_warm} warmup + {n_eval} eval)\n{'='*60}")

        pool_idx: BoxOfficePoolIndex | None = None
        if needs_pool:
            # Load infusion at max(top_k) across active methods, then
            # rely on each method's own top_k to slice. Avoids loading
            # the same pool file twice for K=5 and K=10.
            k_pool = max(m[2] for m in METHODS if m[3] == "semantic_pool")
            pool_idx = BoxOfficePoolIndex(args.pool_dir, seed=seed, k=k_pool)
            print(f"  loaded BoxOfficePoolIndex(s{seed}, k={k_pool}) from "
                  f"{args.pool_dir}: "
                  f"{len(pool_idx.cid_to_text):,} chunks")

        results = []
        t_total = time.time()
        for local_i, (orig_i, row) in enumerate(shard_items):
            qid = str(row.get("query_id") or row.get("trace_id") or "")
            is_eval = (not eval_ids) or (qid in eval_ids)
            tag = "EVAL" if is_eval else "WARM"
            print(f"\n  [{local_i+1}/{len(shard_items)}] {tag} "
                  f"orig#{orig_i}  {qid or orig_i}",
                  flush=True)
            try:
                r = run_row(row, tokenizer, baseline, mgrs,
                            args.recomp_ratios, args.max_new_tokens,
                            pool_idx=pool_idx)
            except Exception as exc:
                print(f"    ERROR: {exc}", flush=True)
                r = {
                    "query_id": qid,
                    "error": str(exc),
                }
            r["orig_index"] = orig_i
            r["seed"]       = seed
            r["is_eval"]    = is_eval
            results.append(r)
            elapsed = time.time() - t_total
            eta = elapsed / (local_i+1) * (len(shard_items) - local_i - 1)
            print(f"    eta={eta/60:.1f}m", flush=True)

        run_methods = [m[0] for m in METHODS]
        run_cfg = {
            "model":           args.model,
            "model_short":     mshort,
            "seed":            seed,
            "dataset_stem":    args.dataset_stem,
            "eval_stem":       args.eval_stem,
            "n_eval_ids":      len(eval_ids),
            "boxoffice_dir":   str(args.boxoffice_dir),
            "recomp_ratios":   args.recomp_ratios,
            "max_new_tokens":  args.max_new_tokens,
            "shard":           args.shard,
            "num_shards":      args.num_shards,
            "methods":         run_methods,
        }

        # Atomic write via tmp + rename — leaves the previous good JSON
        # in place if the writer crashes mid-write. With --output-tag
        # each worker has its own destination file, so no concurrent-
        # writer race; --merge into a shared file (legacy single-process
        # use) is still supported.
        if args.merge and out_path.exists():
            existing = json.loads(out_path.read_text())
            ex_by_qid = {q.get("query_id"): q
                         for q in existing.get("per_query", [])
                         if q.get("query_id") is not None}
            for r in results:
                qid = r.get("query_id")
                if qid is None:
                    continue
                if qid in ex_by_qid:
                    for k, v in r.items():
                        ex_by_qid[qid].setdefault(k, v)
                else:
                    ex_by_qid[qid] = r
            seen, merged_pq = set(), []
            for q in existing.get("per_query", []):
                qid = q.get("query_id")
                if qid is None:
                    continue
                merged_pq.append(ex_by_qid[qid]); seen.add(qid)
            for qid, q in ex_by_qid.items():
                if qid not in seen:
                    merged_pq.append(q)

            ex_cfg = existing.get("config", {}) or {}
            ex_cfg["model"]        = args.model
            ex_cfg["model_short"]  = mshort
            ex_cfg["seed"]         = seed
            ex_cfg["dataset_stem"] = args.dataset_stem
            ex_cfg["eval_stem"]    = args.eval_stem
            ex_cfg["n_eval_ids"]   = len(eval_ids)
            ex_cfg["boxoffice_dir"]    = str(args.boxoffice_dir)
            ex_cfg["max_new_tokens"]   = args.max_new_tokens
            ex_cfg["shard"]            = args.shard
            ex_cfg["num_shards"]       = args.num_shards
            ex_cfg["recomp_ratios"]    = sorted(
                {float(x) for x in (ex_cfg.get("recomp_ratios") or [])}
                | {float(x) for x in args.recomp_ratios}
            )
            ex_cfg["methods"]          = sorted(
                set(ex_cfg.get("methods") or []) | set(run_methods)
            )
            existing["config"]    = ex_cfg
            existing["per_query"] = merged_pq
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(existing, ensure_ascii=False))
            os.replace(tmp_path, out_path)
            print(f"\nMerged into: {out_path}  "
                  f"(methods={ex_cfg['methods']}, "
                  f"ratios={ex_cfg['recomp_ratios']}, "
                  f"{len(merged_pq)} rows, "
                  f"{out_path.stat().st_size/1e6:.1f} MB)")
        else:
            out_payload = {"config": run_cfg, "per_query": results}
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(out_payload, ensure_ascii=False))
            os.replace(tmp_path, out_path)
            print(f"\nwrote {out_path}  ({len(results)} rows, "
                  f"{out_path.stat().st_size/1e6:.1f} MB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
