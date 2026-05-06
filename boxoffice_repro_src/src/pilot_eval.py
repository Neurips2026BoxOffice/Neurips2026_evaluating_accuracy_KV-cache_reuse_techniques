#!/usr/bin/env python3
"""Pilot evaluator for Phase-2 multi-hop synthetic KB queries."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cachebend.qa_metrics import compute_f1
from cachebend.ncf.adapters import build_adapter
from cachebend.ncf.flop_utils import compute_flop_metrics, dump_model_constants_json, model_constants_from_model_path
from cachebend.ncf.method_factory import SUPPORTED_METHODS, RuntimeConfig


WINNER_ONLY_RE = re.compile(r"Winner\s*=\s*(?P<winner>[^;\n]+)", re.IGNORECASE)
AWARD_RE = re.compile(r"Winner\s*=\s*(?P<winner>[^;]+)\s*;\s*Margin\s*=\s*(?P<margin>-?\d+)", re.IGNORECASE)
REV_RE = re.compile(r"Winner\s*=\s*(?P<winner>[^;]+)\s*;\s*RevenueGapM\s*=\s*(?P<gap>-?\d+)", re.IGNORECASE)
ANSWER_ONLY_RE = re.compile(r"Answer\s*=\s*(?P<answer>[^\n]+)", re.IGNORECASE)
PUNCT_RE = re.compile(r"[^\w\s]")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_answer_text(text: str) -> str:
    text = text.lower()
    text = PUNCT_RE.sub(" ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def parse_answer(text: str) -> Optional[Dict[str, Any]]:
    m = ANSWER_ONLY_RE.search(text)
    if m and ";" not in text:
        return {"type": "answer_only", "answer": m.group("answer").strip()}
    m = WINNER_ONLY_RE.search(text)
    if m and ";" not in text:
        return {"type": "winner_only", "winner": m.group("winner").strip(), "value": 0}
    m = AWARD_RE.search(text)
    if m:
        return {"type": "award", "winner": m.group("winner").strip(), "value": int(m.group("margin"))}
    m = REV_RE.search(text)
    if m:
        return {"type": "revenue", "winner": m.group("winner").strip(), "value": int(m.group("gap"))}
    stripped = text.strip().splitlines()[0].strip() if text.strip() else ""
    if stripped and stripped != "ADAPTER_RUNTIME_ERROR":
        return {"type": "answer_only", "answer": stripped}
    return None


def metric_candidates(
    gold: Dict[str, Any],
    accepted_answers: Optional[List[str]] = None,
) -> List[str]:
    if accepted_answers:
        return [str(x) for x in accepted_answers if str(x).strip()]
    if gold["type"] == "answer_only":
        answer = gold["answer"]
        return [answer, f"Answer {answer}"]
    if gold["type"] == "winner_only":
        winner = gold["winner"]
        return [winner, f"Winner {winner}"]
    value_label = "Margin" if gold["type"] == "award" else "RevenueGapM"
    winner = gold["winner"]
    value = gold["value"]
    return [
        f"{winner} {value}",
        f"Winner {winner} {value_label} {value}",
    ]


def metric_text(parsed: Optional[Dict[str, Any]], raw_text: str) -> str:
    if parsed is None:
        return raw_text
    if parsed["type"] == "answer_only":
        return parsed["answer"]
    if parsed["type"] == "winner_only":
        return parsed["winner"]
    return f"{parsed['winner']} {parsed['value']}"


def score(
    gold: str,
    pred: str,
    tokenizer: Any,
    accepted_answers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    g = parse_answer(gold)
    p = parse_answer(pred)
    if g is None:
        raise ValueError(f"Invalid gold answer format: {gold}")
    candidates = metric_candidates(g, accepted_answers)
    answer_f1 = compute_f1(metric_text(p, pred), candidates, tokenizer) if tokenizer is not None else 0.0
    if p is None:
        return {"exact_match": 0, "field_match": 0.0, "parse_error": 1, "answer_f1": answer_f1}
    if g["type"] != p["type"]:
        return {"exact_match": 0, "field_match": 0.0, "parse_error": 0, "answer_f1": answer_f1}
    if g["type"] == "answer_only":
        pred_norm = normalize_answer_text(p["answer"])
        matched = int(any(normalize_answer_text(answer) == pred_norm for answer in candidates))
        return {"exact_match": matched, "field_match": float(matched), "parse_error": 0, "answer_f1": answer_f1}
    if g["type"] == "winner_only":
        w = int(g["winner"] == p["winner"])
        return {"exact_match": w, "field_match": float(w), "parse_error": 0, "answer_f1": answer_f1}
    w = int(g["winner"] == p["winner"])
    v = int(g["value"] == p["value"])
    return {"exact_match": int(w and v), "field_match": (w + v) / 2.0, "parse_error": 0, "answer_f1": answer_f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pilot multi-hop query evaluation.")
    parser.add_argument("--method", choices=SUPPORTED_METHODS, default="baseline")
    parser.add_argument("--queries", type=Path, default=Path("data/phase2_pilot_queries.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/phase2_pilot_eval.json"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"])
    parser.add_argument("--max-queries", type=int, default=10)
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split input rows into N round-robin shards. "
                             "Combine with --shard-i to process only one "
                             "shard. Output written to --out as usual; merge "
                             "shards downstream with merge_pilot_shards.py.")
    parser.add_argument("--shard-i", type=int, default=0,
                        help="0-based shard index; must satisfy 0 ≤ i < N.")
    parser.add_argument("--max-atom-copies", type=int, default=1, help="ZCF/CacheCraft M")
    parser.add_argument("--mchunk-size", type=int, default=3, help="ZCF mc size")
    parser.add_argument("--cachecraft-n", type=int, default=12, help="CacheCraft N")
    parser.add_argument(
        "--trace-id",
        action="append",
        default=[],
        help="Evaluate only the specified trace_id (repeat flag to pass multiple).",
    )
    parser.add_argument("--max-context-tokens", type=int, default=12000)
    parser.add_argument("--recompute-ratio", type=float, default=0.05)
    parser.add_argument(
        "--warm-cache-path",
        default="",
        help="Path to warm cache .pt (required for cacheblend_warm/fusionrag_warm).",
    )
    parser.add_argument(
        "--single-prompt",
        action="store_true",
        help="Evaluate each query as one combined prompt (turn1 + turn2). Recommended for Phase-2 sequence benchmarks.",
    )
    parser.add_argument(
        "--reset-between-queries",
        action="store_true",
        help="Reset adapter/cache after each query (debug mode). Default keeps cache persistent across query sequence.",
    )
    parser.add_argument(
        "--blend-includes-ffn",
        action="store_true",
        help="Assume blend/fusion path includes FFN cost in FLOP estimation.",
    )
    parser.add_argument(
        "--dump-model-constants",
        type=Path,
        default=Path("model_constants.json"),
        help="Path to write verified model constants used by FLOP metrics.",
    )
    parser.add_argument(
        "--timing-mode",
        action="store_true",
        help="Run per-query baseline-vs-method timing loop and log relative prefill proxy timing.",
    )
    parser.add_argument("--timing-repetitions", type=int, default=3)
    parser.add_argument("--timing-warmup-discard", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.queries)
    if args.trace_id:
        selected = set(args.trace_id)
        rows = [r for r in rows if r.get("trace_id") in selected]
    rows = rows[: args.max_queries]
    if args.num_shards > 1:
        if not (0 <= args.shard_i < args.num_shards):
            raise SystemExit(f"shard_i {args.shard_i} out of range for num_shards {args.num_shards}")
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_i]

    config = RuntimeConfig(
        method=args.method,
        model_path=args.model_path,
        device=args.device,
        torch_dtype=args.dtype,
        recompute_ratio=args.recompute_ratio,
        max_atom_copies=args.max_atom_copies,
        mchunk_size=args.mchunk_size,
        cachecraft_n=args.cachecraft_n,
        warm_cache_path=args.warm_cache_path,
    )
    adapter = build_adapter(config, max_context_tokens=args.max_context_tokens)
    tokenizer = getattr(adapter.manager, "tokenizer", None)
    model_constants = adapter.model_constants
    if model_constants is None:
        model_constants = model_constants_from_model_path(args.model_path)
    dump_model_constants_json(model_constants.to_dict(), args.dump_model_constants)

    out_rows: List[Dict[str, Any]] = []
    timing_rows: List[Dict[str, Any]] = []

    def _sync_device() -> None:
        try:
            import torch
        except Exception:
            return

    def _torch_env_info() -> Dict[str, str]:
        try:
            import torch  # type: ignore
        except Exception:
            return {"gpu_model": args.device, "cuda_version": "", "pytorch_version": ""}
        cuda_ver = str(getattr(torch.version, "cuda", "") or "")
        torch_ver = str(getattr(torch, "__version__", "") or "")
        gpu_model = args.device
        try:
            if args.device.startswith("cuda") and torch.cuda.is_available():
                gpu_model = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return {"gpu_model": gpu_model, "cuda_version": cuda_ver, "pytorch_version": torch_ver}
        try:
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elif args.device.startswith("npu") and hasattr(torch, "npu"):
                torch.npu.synchronize()
        except Exception:
            return

    def _row_prompt_segments(row_obj: Dict[str, Any]) -> Optional[List[str]]:
        prompt_segments = row_obj.get("prompt_segments")
        if isinstance(prompt_segments, list) and prompt_segments:
            segs = [str(seg).strip() for seg in prompt_segments if str(seg).strip()]
            # Match paper2/exp/boxoffice/run_boxoffice.build_cb_prompt_tensor:
            # trailing "\n\n" on every non-last segment so the
            # post-DSEP-strip token stream is byte-identical to the test's
            # baseline path. Without this the filter sees
            # `chunk1<DSEP>chunk2…question` while the test sees
            # `chunk1\n\n<DSEP>chunk2\n\n…question`, and the model gives
            # different answers for the same row in ~3% of cases.
            return ([s + "\n\n" for s in segs[:-1]] + [segs[-1]]) if segs else None
        ctxs = row_obj.get("ctxs")
        if isinstance(ctxs, list) and ctxs:
            metadata = row_obj.get("metadata") or {}
            instruction = str(metadata.get("instruction_chunk") or "").strip()
            if not instruction:
                turn_1 = str(row_obj.get("turn_1_poison_prompt") or "")
                first_ctx = str(ctxs[0]).strip()
                if first_ctx and first_ctx in turn_1:
                    instruction = turn_1.split(first_ctx, 1)[0].strip()
            question = str(row_obj.get("turn_2_eval_prompt") or row_obj.get("question") or "").strip()
            segments = [instruction] if instruction else []
            segments.extend(str(ctx).strip() for ctx in ctxs if str(ctx).strip())
            if question:
                segments.append(question)
            if len(segments) >= 2:
                return segments
        return None

    def _run_once(eval_adapter, row_obj: Dict[str, Any], force_full: bool = False) -> Any:
        if args.single_prompt:
            segments = _row_prompt_segments(row_obj)
            if segments:
                return eval_adapter.query_segments(segments, force_full=force_full)
            prompt = f"{row_obj['turn_1_poison_prompt']}\n\n{row_obj['turn_2_eval_prompt']}"
            return eval_adapter.query(prompt, force_full=force_full)
        return eval_adapter.poison_then_query(row_obj["turn_1_poison_prompt"], row_obj["turn_2_eval_prompt"])

    for row_idx, row in enumerate(rows):
        result = _run_once(adapter, row)
        cost_metrics = (result.stats or {}).get("cost_metrics", {})
        flop_metrics = compute_flop_metrics(
            cost_metrics=cost_metrics,
            constants=model_constants,
            blend_includes_ffn=args.blend_includes_ffn,
            computation_notes=f"method={args.method}",
        )
        result.stats["flop_metrics"] = flop_metrics
        if args.timing_mode and row_idx >= int(args.timing_warmup_discard):
            full_times: List[float] = []
            cached_times: List[float] = []
            for _ in range(max(1, args.timing_repetitions)):
                adapter.reset()
                _sync_device()
                t0 = time.perf_counter()
                _run_once(adapter, row, force_full=True)
                _sync_device()
                t1 = time.perf_counter()
                full_times.append((t1 - t0) * 1000.0)

                adapter.reset()
                _sync_device()
                t0 = time.perf_counter()
                _run_once(adapter, row)
                _sync_device()
                t1 = time.perf_counter()
                cached_times.append((t1 - t0) * 1000.0)
            full_times_sorted = sorted(full_times)
            cached_times_sorted = sorted(cached_times)
            mid = len(full_times_sorted) // 2
            t_full = float(full_times_sorted[mid])
            t_cached = float(cached_times_sorted[mid])
            t_rel = (t_cached / t_full) if t_full > 0 else 1.0
            env = _torch_env_info()
            timing_payload = {
                "t_prefill_full_ms": t_full,
                "t_prefill_cached_ms": t_cached,
                "T_rel": float(t_rel),
                "repetitions": int(args.timing_repetitions),
                "warmup_discarded": int(args.timing_warmup_discard),
                "gpu_model": env["gpu_model"],
                "cuda_version": env["cuda_version"],
                "pytorch_version": env["pytorch_version"],
                "timing_mode_note": "Wall-clock around method query path; includes method overhead and decode-side constant costs.",
            }
            result.stats["timing_metrics"] = timing_payload
            timing_rows.append({"trace_id": row["trace_id"], **timing_payload})
        accepted_answers = row.get("answers")
        if not isinstance(accepted_answers, list):
            accepted_answers = (row.get("metadata", {}) or {}).get("accepted_answers")
        if isinstance(accepted_answers, list):
            accepted_answers = [str(x) for x in accepted_answers]
        else:
            accepted_answers = None
        sc = score(row["gold_answer"], result.text, tokenizer=tokenizer, accepted_answers=accepted_answers)
        out_rows.append(
            {
                "id": row.get("id"),
                "question": row.get("question"),
                "answers": row.get("answers"),
                "ctxs": row.get("ctxs"),
                "trace_id": row["trace_id"],
                "run_name": row.get("run_name"),
                "domain": row.get("domain"),
                "metadata": row.get("metadata", {}),
                "gold_answer": row["gold_answer"],
                "predicted_answer": result.text,
                "score": sc,
                "stats": result.stats,
            }
        )
        if args.reset_between_queries:
            adapter.reset()

    n = len(out_rows)
    mean_r_actual = 0.0
    mean_f_norm = 0.0
    if n:
        mean_r_actual = (
            sum(float(((x.get("stats") or {}).get("cost_metrics") or {}).get("R_actual", 1.0)) for x in out_rows)
            / n
        )
        mean_f_norm = (
            sum(float(((x.get("stats") or {}).get("flop_metrics") or {}).get("F_norm", 1.0)) for x in out_rows)
            / n
        )
    summary = {
        "num_queries": n,
        "method": args.method,
        "exact_match": sum(x["score"]["exact_match"] for x in out_rows) / n if n else 0.0,
        "field_match": sum(x["score"]["field_match"] for x in out_rows) / n if n else 0.0,
        "parse_error_rate": sum(x["score"]["parse_error"] for x in out_rows) / n if n else 1.0,
        "answer_f1": sum(float(x["score"].get("answer_f1", 0.0)) for x in out_rows) / n if n else 0.0,
        "mean_R_actual": mean_r_actual,
        "mean_F_norm": mean_f_norm,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "model_constants": model_constants.to_dict(),
                "rows": out_rows,
                "timing_rows": timing_rows if args.timing_mode else [],
            },
            f,
            indent=2,
        )
    print(f"Wrote results to {args.out}")
    print(
        f"exact_match={summary['exact_match']:.4f} "
        f"answer_f1={summary['answer_f1']:.4f} "
        f"parse_error_rate={summary['parse_error_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
