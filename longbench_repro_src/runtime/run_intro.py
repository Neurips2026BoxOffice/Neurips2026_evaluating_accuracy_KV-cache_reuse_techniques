#!/usr/bin/env python3
"""
Run the LongBench experiments used in the paper on the enriched `_lb.json`
datasets shipped with this bundle.

By default the script expects the methods bundle to live alongside this bundle:

    ../cache_methods_src

Override that with `CACHE_METHODS_SRC=/abs/path/to/cache_methods_src`.
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

# ── Resolve bundle paths ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_METHODS_SRC = ROOT_DIR.parent / "cache_methods_src"
BENCHMARK_SRC = Path(
    os.environ.get("CACHE_METHODS_SRC", str(DEFAULT_METHODS_SRC))
)
DATA_DIR = Path(
    os.environ.get("LONGBENCH_DATA_DIR", str(ROOT_DIR / "data" / "enriched"))
)
OUTPUT_DIR = SCRIPT_DIR / "output"

if not BENCHMARK_SRC.exists():
    raise SystemExit(
        "methods bundle src not found. Set CACHE_METHODS_SRC to the "
        f"`src` directory from cache_methods_src: {BENCHMARK_SRC}"
    )

if str(BENCHMARK_SRC) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SRC))

from cachebend.ncf.cutils import (
    build_model_sdpa, build_tokenizer,
    split_prompt_for_warm_chunks, to_str_prompt,
)
from cachebend.ncf.onthefly_infusion import (
    CacheBlendOnTheFlyManager, FusionRAGOnTheFlyManager,
    CacheBlendPromptOnTheFlyManager,
)
from cachebend.ncf.zcf_v10 import BaselineNosep


# ── Config ───────────────────────────────────────────────────────────
INSTRUCTION = (
    "Answer the question using ONLY the provided context. "
    "Output only the final answer as a short phrase (entity, name, date, or number). "
    "Do not restate the question. Do not explain. Do not add any extra text."
)
RECOMP_RATIOS = [0.15]
DATASETS = {
    "musique_lb":  "musique_lb.json",
    "2wiki_lb":    "2wikimultihopqa_lb.json",
    "hotpotqa_lb": "hotpotqa_lb.json",
}
# (key, manager_kind, top_k, neighbor_mode)
#   manager_kind  ∈ {"cb", "fr", "cbp"}   — which manager instance to dispatch to
#   neighbor_mode ∈ {"none", "random", "auto"}
#     none    — no K=1 conditioning chunk (top_k=0 paths)
#     random  — random non-golden chunk from the same query (cb_k1, cb_k1q)
#     auto    — manager handles its own neighbours (cbp = prompt-conditioned
#               CacheBlend; PROMPT_NEIGHBOR_TEXT lives in cachebend.ncf.onthefly_infusion)
METHODS: list[tuple[str, str, int, str]] = [
    ("cb_k0",      "cb",  0, "none"),
    ("cb_k1",      "cb",  1, "random"),
    ("cb_k1q",     "fr",  1, "random"),
    ("cb_kprompt", "cbp", 1, "auto"),
]


# ── F1 + helpers ──────────────────────────────────────────────────────
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


def build_prompt_tensor(context_chunks, question, tokenizer):
    ascii_sep = tokenizer.sep_token
    segments = [INSTRUCTION] + list(context_chunks) + [f"Question: {question}"]
    prompt_str = to_str_prompt(tokenizer, ascii_sep, segments)
    ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
    return torch.tensor(ids, dtype=torch.long)


def run_baseline(baseline, prompt_ids, max_tokens=16):
    baseline.begin_fresh_query()
    out, _, _ = baseline.new_query(prompt_ids, max_new_tokens=max_tokens)
    return _strip_thinking(str(out)).strip()


def check_no_context(baseline, tokenizer, question, gold, max_tokens=16):
    segments = [
        "Answer the following question concisely — just the answer, no explanation.",
        f"Question: {question}",
    ]
    prompt_str = to_str_prompt(tokenizer, tokenizer.sep_token, segments)
    ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
    prompt_ids = torch.tensor(ids, dtype=torch.long)
    pred = run_baseline(baseline, prompt_ids, max_tokens=max_tokens)
    return compute_f1(pred, gold), pred


def build_random_neighbors(sample, prompt_tensor, tokenizer):
    """K=1 random neighbour per chunk: pick a random non-golden chunk
    from the same query's context."""
    sep_tok = tokenizer.sep_token or "<DSEP>"
    sep_ids = tokenizer(sep_tok, add_special_tokens=False).input_ids
    sep = torch.tensor(sep_ids, dtype=torch.long)
    chunks = split_prompt_for_warm_chunks(prompt_tensor, sep, tokenizer)
    doc_chunks = chunks[1:-1]

    golden_local = set(sample.get("golden", []))
    non_golden_texts = [sample["context"][i]
                        for i in range(len(sample["context"]))
                        if i not in golden_local]
    if not non_golden_texts:
        non_golden_texts = list(sample["context"])

    cid_to_neighbors = {}
    for chunk in doc_chunks:
        nbr = random.choice(non_golden_texts)
        cid_to_neighbors[str(chunk.cid)] = [nbr]
    return cid_to_neighbors


# ── Per-query runner ──────────────────────────────────────────────────
def run_query(sample: dict, tokenizer, baseline,
              cb_mgr, fr_mgr, cbp_mgr, recomp_ratios, max_tokens=16) -> dict:
    context  = sample["context"]
    question = sample["query"]
    gold     = sample["answer"]
    qid      = sample.get("query_id", "")
    prompt   = build_prompt_tensor(context, question, tokenizer)

    result: dict = {
        "query_id":  qid,
        "question":  question,
        "gold_answer": gold,
        "n_chunks":  len(context),
        "n_golden":  len(sample.get("golden", [])),
        "n_tokens":  int(prompt.numel()),
    }

    # baseline (full prefill, BaselineNosep)
    t0 = time.time()
    try:
        ans = run_baseline(baseline, prompt, max_tokens)
        f1  = compute_f1(ans, gold)
    except Exception as exc:
        ans = f"ERROR: {exc}"; f1 = 0.0
    result["baseline"] = {"answer": ans, "f1": f1, "time_s": time.time()-t0}
    print(f"    base: f1={f1:.3f}", end="", flush=True)
    _empty_cache()

    # no-context check (baseline w/o any context)
    try:
        nc_f1, nc_ans = check_no_context(baseline, tokenizer, question, gold)
    except Exception as exc:
        nc_f1, nc_ans = 0.0, f"ERROR: {exc}"
    result["no_context_f1"]     = nc_f1
    result["no_context_answer"] = nc_ans
    print(f"  | no_ctx: {nc_f1:.3f}", end="", flush=True)

    # K=1 neighbour set, computed only if some method needs random non-golden
    # conditioning. Methods with neighbor_mode="auto" let the manager handle
    # neighbours itself (e.g. cbp = CacheBlendPromptOnTheFlyManager).
    needs_random = any(m[3] == "random" for m in METHODS)
    rand_neighbors = build_random_neighbors(sample, prompt, tokenizer) if needs_random else {}

    # method × R grid
    for key, mgr_kind, top_k, neighbor_mode in METHODS:
        mgr = {"cb": cb_mgr, "fr": fr_mgr, "cbp": cbp_mgr}[mgr_kind]
        nbrs = rand_neighbors if neighbor_mode == "random" else {}
        for R in recomp_ratios:
            r_key = f"{key}_R{R:.2f}"
            t0 = time.time()
            mgr.begin_fresh_query()
            mgr.cache.recomp_ratio = R
            if hasattr(mgr.cache, "selector"):
                mgr.cache.selector.recomp_ratio = R
            mgr.top_k = top_k
            mgr.configure_query_neighbors(nbrs)
            try:
                output, used_cache, stats = mgr.new_query(prompt)
                ans = _strip_thinking(str(output))
                f1_val = compute_f1(ans, gold)
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

    print(f"  | R0.15 cb_k1={result.get('cb_k1_R0.15',{}).get('f1',0):.2f} "
          f"cb_k1q={result.get('cb_k1q_R0.15',{}).get('f1',0):.2f} "
          f"cb_kprompt={result.get('cb_kprompt_R0.15',{}).get('f1',0):.2f}",
          flush=True)
    return result


# ── Main ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--device-type", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    ap.add_argument("--recomp-ratios", nargs="*", type=float,
                    default=RECOMP_RATIOS)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--shard", type=int, default=0,
                    help="shard index (0-based); requires --num-shards>1")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="when >1, this worker only processes queries where "
                         "i %% num_shards == shard, writing to "
                         "intro_<ds>_<model>.shard<i>of<K>.json. Run "
                         "aggregate_shards.py afterwards to merge.")
    ap.add_argument("--merge", action="store_true",
                    help="merge new per-query ratio keys into an existing "
                         "intro_<ds>_<model>.json (if present) instead of "
                         "skipping or overwriting. Preserves existing keys "
                         "(baseline, no_context_f1, and any prior ratio "
                         "entries); only adds keys that are missing. "
                         "Requires --num-shards 1.")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="restrict the method grid to these keys (e.g. "
                         "'--methods cb_k0'); default runs every entry in "
                         "METHODS. Combine with --merge to compute ONLY the "
                         "new keys and fold them into existing per-query "
                         "dicts, without recomputing cb_k1/cb_k1q.")
    args = ap.parse_args()

    if args.merge and args.num_shards != 1:
        raise SystemExit("--merge requires --num-shards 1")

    if args.methods:
        req = set(args.methods)
        known = {m[0] for m in METHODS}
        unknown = req - known
        if unknown:
            raise SystemExit(
                f"unknown --methods: {sorted(unknown)} "
                f"(known: {sorted(known)})")
        METHODS[:] = [m for m in METHODS if m[0] in req]
        print(f"Method filter: {[m[0] for m in METHODS]}")

    if args.num_shards < 1 or not (0 <= args.shard < args.num_shards):
        raise SystemExit(f"bad --shard/--num-shards: {args.shard}/{args.num_shards}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    if args.device_type != "auto":
        device = f"{args.device_type}:{args.device_id}"
    elif torch.cuda.is_available():
        device = f"cuda:{args.device_id}"
    elif hasattr(torch, "npu") and torch.npu.is_available():
        device = f"npu:{args.device_id}"
    else:
        device = "cpu"
    mshort = Path(args.model).name
    print(f"Model: {args.model}\nDevice: {device}\nDatasets: {args.datasets}\n"
          f"Recomp ratios: {args.recomp_ratios}\nOut dir: {args.out_dir}\n"
          f"Methods src: {BENCHMARK_SRC}\nData dir: {DATA_DIR}\n"
          f"Shard: {args.shard}/{args.num_shards}")

    if not BENCHMARK_SRC.exists():
        raise SystemExit(f"methods bundle src not found: {BENCHMARK_SRC}")
    if not DATA_DIR.exists():
        raise SystemExit(f"enriched data directory not found: {DATA_DIR}")

    def out_path_for(ds: str) -> Path:
        if args.num_shards == 1:
            return args.out_dir / f"intro_{ds}_{mshort}.json"
        return args.out_dir / f"intro_{ds}_{mshort}.shard{args.shard}of{args.num_shards}.json"

    # pre-flight: skip if my shard's output (or the merged file) already
    # exists — unless --merge is on, in which case we want to proceed so
    # new ratio keys can be folded into the existing file.
    pending = []
    for ds in args.datasets:
        merged = args.out_dir / f"intro_{ds}_{mshort}.json"
        out_path = out_path_for(ds)
        if args.merge:
            pending.append(ds); continue
        if not args.force and (out_path.exists() or merged.exists()):
            print(f"  [skip] {out_path if out_path.exists() else merged}")
            continue
        pending.append(ds)
    if not pending:
        print("Nothing to do.")
        return

    llm = build_model_sdpa(args.model, torch_dtype=args.dtype).to(device)
    llm.eval()
    tokenizer = build_tokenizer(args.model)
    print("Model loaded.")
    baseline = BaselineNosep(device=device, llm=llm, tkn=tokenizer)
    cb_mgr = CacheBlendOnTheFlyManager(
        device=device, R=0.0, top_k=0, llm=llm, tkn=tokenizer)
    fr_mgr = FusionRAGOnTheFlyManager(
        device=device, R=0.0, top_k=0, llm=llm, tkn=tokenizer)
    cbp_mgr = CacheBlendPromptOnTheFlyManager(
        device=device, R=0.0, llm=llm, tkn=tokenizer)

    for ds in pending:
        out_path = out_path_for(ds)
        fname = DATASETS[ds]
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"[skip] {fpath} not found", file=sys.stderr); continue
        data = json.loads(fpath.read_text())
        # shard filter — keep (orig_index, sample) for samples this worker owns
        shard_items = [(i, s) for i, s in enumerate(data)
                       if i % args.num_shards == args.shard]
        print(f"\n{'='*60}\n  {ds}: {len(data)} queries  "
              f"(this shard: {len(shard_items)})\n{'='*60}")

        results = []
        t_total = time.time()
        for local_i, (orig_i, sample) in enumerate(shard_items):
            print(f"\n  [{local_i+1}/{len(shard_items)}] "
                  f"orig#{orig_i}  {sample.get('query_id', orig_i)}",
                  flush=True)
            r = run_query(
                sample, tokenizer, baseline, cb_mgr, fr_mgr, cbp_mgr,
                args.recomp_ratios, args.max_new_tokens,
            )
            r["orig_index"] = orig_i
            results.append(r)
            elapsed = time.time() - t_total
            eta = elapsed / (local_i+1) * (len(shard_items) - local_i - 1)
            print(f"    eta={eta/60:.1f}m", flush=True)

        out = {
            "config": {
                "model":   args.model,
                "dataset": ds,
                "recomp_ratios": args.recomp_ratios,
                "methods": [m[0] for m in METHODS],
                "n_queries":   len(data),
                "n_in_shard":  len(shard_items),
                "shard":       args.shard,
                "num_shards":  args.num_shards,
                "elapsed_s":   time.time() - t_total,
            },
            "per_query": results,
        }

        if args.merge and out_path.exists():
            existing = json.loads(out_path.read_text())
            ex_by_qid = {q.get("query_id"): q for q in existing.get("per_query", [])
                         if q.get("query_id") is not None}
            for r in results:
                qid = r.get("query_id")
                if qid is None: continue
                if qid in ex_by_qid:
                    # additive: keep existing keys; add any that are missing
                    for k, v in r.items():
                        if k not in ex_by_qid[qid]:
                            ex_by_qid[qid][k] = v
                else:
                    ex_by_qid[qid] = r
            # rebuild per_query, preserving existing dataset order for known
            # qids and appending any brand-new ones at the end
            seen = set()
            merged_pq = []
            for q in existing.get("per_query", []):
                qid = q.get("query_id")
                if qid is None: continue
                merged_pq.append(ex_by_qid[qid]); seen.add(qid)
            for qid, q in ex_by_qid.items():
                if qid not in seen:
                    merged_pq.append(q)
            ex_cfg = existing.get("config", {}) or {}
            old_ratios = ex_cfg.get("recomp_ratios") or []
            union_ratios = sorted({float(x) for x in old_ratios}
                                  | {float(x) for x in args.recomp_ratios})
            ex_cfg["recomp_ratios"] = union_ratios
            ex_cfg["model"]   = args.model
            ex_cfg["dataset"] = ds
            ex_cfg["methods"] = sorted(
                set(ex_cfg.get("methods") or []) | {m[0] for m in METHODS})
            ex_cfg["merged_elapsed_s"] = (
                ex_cfg.get("merged_elapsed_s", 0.0) + (time.time() - t_total))
            ex_cfg["n_queries"] = len(data)
            existing["config"] = ex_cfg
            existing["per_query"] = merged_pq
            out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
            print(f"\nMerged into: {out_path}  "
                  f"(recomp_ratios={union_ratios})")
        else:
            out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
            print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
