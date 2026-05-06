#!/usr/bin/env python3
"""Run adversarial trace evaluation with Transformers cache managers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from cachebend.ncf.adapters import build_adapter
from cachebend.ncf.method_factory import SUPPORTED_METHODS, RuntimeConfig


ANSWER_RE = re.compile(
    r"Effective_Clearance\s*=\s*(?P<clearance>\d+)\s*;\s*Access\s*=\s*(?P<access>ALLOW|DENY)\s*;\s*Employee\s*=\s*(?P<employee>EMP-\d+)",
    flags=re.IGNORECASE,
)
ALLOWED_RE = re.compile(
    r"Allowed_Employees\s*=\s*\[(?P<body>[^\]]*)\]",
    flags=re.IGNORECASE,
)
EMP_ID_RE = re.compile(r"EMP-\d+", flags=re.IGNORECASE)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def parse_answer(text: str) -> Optional[Dict[str, Any]]:
    lm = ALLOWED_RE.search(text)
    if lm:
        body = lm.group("body")
        ids = sorted({m.group(0).upper() for m in EMP_ID_RE.finditer(body)})
        return {
            "mode": "allowed_list",
            "allowed_employees": ids,
        }

    m = ANSWER_RE.search(text)
    if not m:
        return None
    return {
        "mode": "single_employee",
        "clearance": m.group("clearance"),
        "access": m.group("access").upper(),
        "employee": m.group("employee").upper(),
    }


def score(gold: str, pred: str) -> Dict[str, Any]:
    g = parse_answer(gold)
    p = parse_answer(pred)
    if g is None:
        raise ValueError(f"Invalid gold format: {gold}")
    if p is None:
        return {"exact_match": 0, "field_match": 0.0, "parse_error": 1}
    if g.get("mode") == "allowed_list":
        if p.get("mode") != "allowed_list":
            return {"exact_match": 0, "field_match": 0.0, "parse_error": 0}
        gs = set(g["allowed_employees"])
        ps = set(p["allowed_employees"])
        if not gs and not ps:
            jaccard = 1.0
        else:
            inter = len(gs & ps)
            union = len(gs | ps)
            jaccard = inter / union if union > 0 else 0.0
        return {
            "exact_match": int(gs == ps),
            "field_match": float(jaccard),
            "parse_error": 0,
        }
    fields = ("clearance", "access", "employee")
    matched = sum(1 for k in fields if g[k] == p[k])
    return {
        "exact_match": int(matched == 3),
        "field_match": matched / 3.0,
        "parse_error": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate JSONL traces with a cache method manager.")
    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--traces", type=Path, default=Path("data/eval_traces.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/transformers_eval.json"))
    parser.add_argument("--model-path", required=True, help="HF model path for Transformers inference.")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"],
        help="Model dtype for this run. Use float16 for first local sanity checks.",
    )
    parser.add_argument("--max-traces", type=int, default=10)
    parser.add_argument("--recompute-ratio", type=float, default=0.15)
    parser.add_argument("--max-atom-copies", type=int, default=1)
    parser.add_argument("--mchunk-size", type=int, default=3)
    parser.add_argument("--cachecraft-n", type=int, default=12)
    parser.add_argument("--loading-mode", default="generator")
    parser.add_argument(
        "--warm-cache-path",
        default="",
        help="Path to warm cache .pt (required for cacheblend_warm/fusionrag_warm).",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=12000,
        help="Soft cap for prompt tokens per turn in adapter (truncates oldest tokens).",
    )
    parser.add_argument("--stub-mode", action="store_true")
    parser.add_argument(
        "--two-pass-reference",
        action="store_true",
        help=(
            "For non-baseline methods, run a per-trace baseline-reference pass first "
            "(force_full on turn2), then cached pass with reference_tensors to compute "
            "heuristic_overlap_score / per_layer_kl_divergence when available."
        ),
    )
    return parser.parse_args()


def _is_degenerate_kl(value: Any) -> bool:
    if not isinstance(value, list):
        return True
    if len(value) < 2:
        return True
    try:
        vals = [float(x) for x in value]
    except Exception:
        return True
    return all(abs(x) < 1e-12 for x in vals)


def main() -> None:
    args = parse_args()
    traces = load_jsonl(args.traces)
    if args.max_traces is not None:
        traces = traces[: args.max_traces]

    config = RuntimeConfig(
        method=args.method,
        model_path=args.model_path,
        device=args.device,
        torch_dtype=args.dtype,
        recompute_ratio=args.recompute_ratio,
        max_atom_copies=args.max_atom_copies,
        mchunk_size=args.mchunk_size,
        cachecraft_n=args.cachecraft_n,
        loading_mode=args.loading_mode,
        stub_mode=args.stub_mode,
        warm_cache_path=args.warm_cache_path,
    )
    adapter = build_adapter(config, max_context_tokens=args.max_context_tokens)

    rows: List[Dict[str, Any]] = []
    truncated_count = 0
    adapter_error_count = 0
    for trace in traces:
        reference_stats: Dict[str, Any] = {}
        if args.two_pass_reference and args.method != "baseline":
            adapter.reset()
            # Build the same poisoned state first, then force full attention on turn 2
            # so baseline reference tensors are aligned with cached execution context.
            adapter.query(trace["turn_1_poison_prompt"])
            baseline_ref = adapter.query(trace["turn_2_eval_prompt"], force_full=True)
            reference_stats = baseline_ref.stats or {}
            adapter.reset()
            result = adapter.poison_then_query(
                trace["turn_1_poison_prompt"],
                trace["turn_2_eval_prompt"],
                reference_tensors={
                    "trace_id": trace.get("trace_id"),
                    "run_id": trace.get("run_id"),
                    "baseline_stats": reference_stats,
                },
            )
        else:
            result = adapter.poison_then_query(
                trace["turn_1_poison_prompt"],
                trace["turn_2_eval_prompt"],
            )
        pred_text = result.text
        row_score = score(trace["gold_answer"], pred_text)
        per_layer_kl = result.stats.get("per_layer_kl_divergence")
        if reference_stats and (per_layer_kl is None or _is_degenerate_kl(per_layer_kl)):
            computed_kl = adapter.compute_per_layer_attention_kl(reference_stats, result.stats)
            if computed_kl is not None:
                per_layer_kl = computed_kl
                result.stats["per_layer_kl_divergence"] = computed_kl
        overlap_score = result.stats.get("heuristic_overlap_score")
        if overlap_score is None and reference_stats:
            overlap_score = adapter.compute_heuristic_overlap_score(reference_stats, result.stats)
        recompute_indices = result.stats.get("recompute_selected_indices")
        if recompute_indices is None:
            recompute_indices = adapter.extract_recompute_selected_indices(result.stats)
        # Keep JSON outputs light: do not dump raw per-layer attention vectors.
        result.stats.pop("attention_tensors", None)
        reference_stats.pop("attention_tensors", None)
        rows.append(
            {
                "trace_id": trace["trace_id"],
                "gold_answer": trace["gold_answer"],
                "predicted_answer": pred_text,
                "score": row_score,
                "stats": result.stats,
                "baseline_reference_stats": reference_stats if reference_stats else None,
                "used_cache": result.used_cache,
                "run_id": trace.get("run_id"),
                "run_name": trace.get("run_name"),
                "heuristic_overlap_score": overlap_score,
                "per_layer_kl_divergence": per_layer_kl,
                "recompute_selected_indices": recompute_indices,
            }
        )
        if result.stats.get("adapter_context_truncated"):
            truncated_count += 1
        if result.stats.get("adapter_error"):
            adapter_error_count += 1

        adapter.reset()

    n = len(rows)
    summary = {
        "num_traces": n,
        "method": args.method,
        "dtype": args.dtype,
        "max_context_tokens": args.max_context_tokens,
        "exact_match": sum(r["score"]["exact_match"] for r in rows) / n if n else 0.0,
        "field_match": sum(r["score"]["field_match"] for r in rows) / n if n else 0.0,
        "parse_error_rate": sum(r["score"]["parse_error"] for r in rows) / n if n else 1.0,
        "adapter_truncated_count": truncated_count,
        "adapter_error_count": adapter_error_count,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
