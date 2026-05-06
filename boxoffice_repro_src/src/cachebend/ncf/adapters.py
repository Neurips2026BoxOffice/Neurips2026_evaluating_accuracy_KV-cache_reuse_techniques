"""Unified adapter interface over heterogeneous legacy manager classes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from cachebend.ncf.method_factory import RuntimeConfig, build_manager_from_config
from cachebend.ncf.flop_utils import ModelConstants, model_constants_from_config, model_constants_from_model_path

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class QueryResult:
    text: str
    used_cache: bool
    stats: Dict[str, Any]


@dataclass(frozen=True)
class CostMetrics:
    tokens_total_prompt: int
    tokens_fresh_miss: int
    tokens_cached_hit: int
    tokens_recomputed: int
    tokens_blend_or_fusion: int
    tokens_pure_cache: int
    R_config: Optional[float]
    R_actual: float
    cache_hit_count: int
    cache_miss_count: int
    cache_hit_rate: float
    tokens_invariant_ok: bool
    recompute_invariant_ok: bool
    method_family: str
    zcf_tokens_recomputed_analytical: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MethodAdapter:
    """Normalize prompt encode/query/reset across managers."""

    def __init__(
        self,
        manager: Any,
        attention_export_dir: Optional[Path] = None,
        max_attention_exports_per_run: int = 3,
        max_context_tokens: int = 12000,
    ):
        self._manager = manager
        self._attention_export_dir = attention_export_dir
        self._max_attention_exports_per_run = max_attention_exports_per_run
        self._attention_export_counts: Dict[str, int] = {}
        self._max_context_tokens = max_context_tokens
        self._force_deterministic_generation()
        self._model_constants: Optional[ModelConstants] = self._resolve_model_constants()

    @property
    def manager(self) -> Any:
        return self._manager

    @property
    def model_constants(self) -> Optional[ModelConstants]:
        return self._model_constants

    def encode(self, prompt_text: str) -> Any:
        segments = [s.strip() for s in prompt_text.split("\n\n") if s.strip()]
        if len(segments) <= 1:
            segments = [prompt_text.strip(), "Provide the requested output format only."]
        return self.encode_segments(segments)

    def encode_segments(self, segments: Sequence[str]) -> Any:
        segments = [str(s).strip() for s in segments if str(s).strip()]
        if not segments:
            segments = ["Provide the requested output format only."]
        prompt = self._manager.to_str_prompt(segments)
        # `to_str_prompt()` already returns the fully formatted chat prompt.
        # Re-adding special tokens here changes the first chunk CID (BOS path)
        # and breaks warm-cache alignment relative to the old benchmark flow.
        return self._manager.tokenize(prompt, use_special_tokens=False)

    def query(
        self,
        prompt_text: str,
        reference_tensors: Optional[Dict[str, Any]] = None,
        force_full: bool = False,
    ) -> QueryResult:
        tokenized = self.encode(prompt_text)
        return self.query_tokenized(tokenized, reference_tensors=reference_tensors, force_full=force_full)

    def query_segments(
        self,
        segments: Sequence[str],
        reference_tensors: Optional[Dict[str, Any]] = None,
        force_full: bool = False,
    ) -> QueryResult:
        tokenized = self.encode_segments(segments)
        return self.query_tokenized(tokenized, reference_tensors=reference_tensors, force_full=force_full)

    def query_tokenized(
        self,
        tokenized: Any,
        reference_tensors: Optional[Dict[str, Any]] = None,
        force_full: bool = False,
    ) -> QueryResult:
        original_tokens = int(tokenized.numel()) if hasattr(tokenized, "numel") else None
        truncated = False
        if original_tokens is not None and original_tokens > self._max_context_tokens:
            tokenized = tokenized[-self._max_context_tokens :]
            truncated = True
        try:
            output = used_cache = stats = None
            called = False
            # Try most-informative signature first, then gracefully fall back.
            for kwargs in (
                {"force_full": force_full, "reference_tensors": reference_tensors},
                {"force_full": force_full},
                {"reference_tensors": reference_tensors},
                {},
            ):
                try:
                    output, used_cache, stats = self._manager.new_query(tokenized, **kwargs)
                    called = True
                    break
                except TypeError:
                    continue
            if not called:
                output, used_cache, stats = self._manager.new_query(tokenized)
            stats = stats or {}
            stats["adapter_input_tokens_original"] = original_tokens
            stats["adapter_input_tokens_used"] = int(tokenized.numel()) if hasattr(tokenized, "numel") else None
            stats["adapter_context_truncated"] = truncated
            stats["adapter_context_cap"] = self._max_context_tokens
            stats["cost_metrics"] = self.get_cost_metrics(stats).to_dict()
            if reference_tensors and isinstance(reference_tensors, dict):
                baseline_stats = reference_tensors.get("baseline_stats") or {}
                if isinstance(baseline_stats, dict):
                    if stats.get("heuristic_overlap_score") is None:
                        stats["heuristic_overlap_score"] = self.compute_heuristic_overlap_score(
                            baseline_stats, stats
                        )
                    existing_kl = stats.get("per_layer_kl_divergence")
                    computed_kl = self.compute_per_layer_attention_kl(baseline_stats, stats)
                    if computed_kl is not None and (
                        existing_kl is None or self._is_degenerate_kl(existing_kl)
                    ):
                        stats["per_layer_kl_divergence"] = computed_kl
                    if stats.get("recompute_selected_indices") is None:
                        stats["recompute_selected_indices"] = self.extract_recompute_selected_indices(stats)
            return QueryResult(text=str(output), used_cache=bool(used_cache), stats=stats)
        except Exception as exc:
            err_stats = {
                "adapter_error": repr(exc),
                "adapter_error_traceback": traceback.format_exc(limit=3),
                "adapter_input_tokens_original": original_tokens,
                "adapter_input_tokens_used": int(tokenized.numel()) if hasattr(tokenized, "numel") else None,
                "adapter_context_truncated": truncated,
                "adapter_context_cap": self._max_context_tokens,
            }
            return QueryResult(text="ADAPTER_RUNTIME_ERROR", used_cache=False, stats=err_stats)

    def _resolve_model_constants(self) -> Optional[ModelConstants]:
        llm = getattr(self._manager, "llm", None)
        cfg = getattr(llm, "config", None)
        if cfg is not None:
            try:
                return model_constants_from_config(cfg)
            except Exception:
                pass
        model_path = getattr(llm, "name_or_path", None) or getattr(self._manager, "model_path", None)
        if not model_path:
            return None
        try:
            return model_constants_from_model_path(str(model_path))
        except Exception:
            return None

    def _method_family(self) -> str:
        name = self._manager.__class__.__name__.lower()
        if "baseline" in name:
            return "baseline"
        if "lmcache" in name:
            return "lmcache_online"
        if "fusionrag" in name:
            return "fusionrag"
        if "zcf" in name:
            return "zcf"
        if "cacheblend" in name:
            return "cacheblend"
        return "unknown"

    def _r_config(self) -> Optional[float]:
        cache = getattr(self._manager, "cache", None)
        if cache is None:
            return None
        val = getattr(cache, "recomp_ratio", None)
        if isinstance(val, (int, float)):
            return float(val)
        return None

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def get_cost_metrics(self, stats: Dict[str, Any]) -> CostMetrics:
        tc = (stats.get("token_composition") or {})
        hit_tokens = self._safe_int(tc.get("hit_tokens"), 0)
        miss_tokens = self._safe_int(tc.get("miss_tokens"), 0)
        total_raw = tc.get("total_tokens", None)
        total_tokens = self._safe_int(total_raw, 0) if total_raw is not None else 0
        if total_tokens <= 0:
            total_tokens = hit_tokens + miss_tokens
        if total_tokens <= 0:
            total_tokens = self._safe_int(stats.get("adapter_input_tokens_used"), 0)

        # Normalize token split against total.
        if hit_tokens + miss_tokens <= 0:
            miss_tokens = total_tokens
            hit_tokens = 0
        elif hit_tokens + miss_tokens != total_tokens and total_tokens > 0:
            # Keep miss stable, repair hit to preserve invariant.
            hit_tokens = max(0, total_tokens - miss_tokens)

        hit_chunks = self._safe_int(tc.get("hit_chunks"), self._safe_int(stats.get("num_cached_chunks"), 0))
        miss_chunks = self._safe_int(tc.get("miss_chunks"), self._safe_int(stats.get("num_fresh_chunks"), 0))
        total_chunks = hit_chunks + miss_chunks
        if total_chunks <= 0:
            total_chunks = self._safe_int(stats.get("total_chunks"), 0)
            if total_chunks > 0 and (hit_chunks + miss_chunks) == 0:
                miss_chunks = total_chunks
                hit_chunks = 0

        family = self._method_family()
        r_cfg = self._r_config()

        blend_tokens = 0
        pure_cache_tokens = 0
        recomputed_tokens = total_tokens

        if family == "baseline":
            blend_tokens = 0
            pure_cache_tokens = 0
            recomputed_tokens = total_tokens
            r_cfg = None
        elif family == "lmcache_online":
            r_eff = max(0.0, min(1.0, float(r_cfg if r_cfg is not None else 0.0)))
            selected_hit_tokens = int(round(hit_tokens * r_eff))
            blend_tokens = 0
            recomputed_tokens = miss_tokens + selected_hit_tokens
            pure_cache_tokens = max(0, total_tokens - recomputed_tokens)
        elif family == "cacheblend":
            # CacheBlend precomputes fresh misses fully, then partially
            # recomputes cached-hit tokens according to R before blending.
            r_eff = max(0.0, min(1.0, float(r_cfg if r_cfg is not None else 0.0)))
            selected_hit_tokens = int(round(hit_tokens * r_eff))
            blend_tokens = hit_tokens
            recomputed_tokens = miss_tokens + selected_hit_tokens
            pure_cache_tokens = max(0, total_tokens - recomputed_tokens)
        elif family == "fusionrag":
            selected = self._safe_int(stats.get("fusionrag_selected_count"), 0)
            if selected <= 0:
                selected = hit_tokens
            blend_tokens = max(0, min(hit_tokens, selected))
            recomputed_tokens = miss_tokens + blend_tokens
            pure_cache_tokens = max(0, total_tokens - recomputed_tokens)
        elif family == "zcf":
            zcf_used_mcids = self._safe_int(stats.get("num_used_mcids"), self._safe_int(stats.get("zcf_used_mcids"), 0))
            chunk_lens = stats.get("zcf_chunk_lens") if isinstance(stats.get("zcf_chunk_lens"), dict) else {}
            if chunk_lens:
                lens = [self._safe_int(v, 0) for v in chunk_lens.values()]
                avg_chunk_len = int(round(sum(lens) / max(1, len(lens))))
            else:
                avg_chunk_len = int(round(total_tokens / max(1, self._safe_int(stats.get("zcf_total_chunks"), max(total_chunks, 1)))))
            recompute_est = zcf_used_mcids * max(1, avg_chunk_len)
            recomputed_tokens = max(miss_tokens, min(total_tokens, recompute_est))
            blend_tokens = 0
            pure_cache_tokens = max(0, total_tokens - recomputed_tokens)
        else:
            # Conservative fallback: all prompt tokens consumed as compute.
            blend_tokens = 0
            pure_cache_tokens = 0
            recomputed_tokens = total_tokens

        if recomputed_tokens > total_tokens:
            recomputed_tokens = total_tokens
        if pure_cache_tokens + blend_tokens + miss_tokens != total_tokens:
            pure_cache_tokens = max(0, total_tokens - (blend_tokens + miss_tokens))
        tokens_invariant_ok = (pure_cache_tokens + blend_tokens + miss_tokens) == total_tokens
        recompute_invariant_ok = recomputed_tokens >= miss_tokens and recomputed_tokens <= total_tokens
        r_actual = (float(recomputed_tokens) / float(total_tokens)) if total_tokens > 0 else 1.0
        hit_rate = (float(hit_chunks) / float(hit_chunks + miss_chunks)) if (hit_chunks + miss_chunks) > 0 else 0.0

        return CostMetrics(
            tokens_total_prompt=int(total_tokens),
            tokens_fresh_miss=int(miss_tokens),
            tokens_cached_hit=int(hit_tokens),
            tokens_recomputed=int(recomputed_tokens),
            tokens_blend_or_fusion=int(blend_tokens),
            tokens_pure_cache=int(pure_cache_tokens),
            R_config=(float(r_cfg) if r_cfg is not None else None),
            R_actual=float(r_actual),
            cache_hit_count=int(hit_chunks),
            cache_miss_count=int(miss_chunks),
            cache_hit_rate=float(hit_rate),
            tokens_invariant_ok=bool(tokens_invariant_ok),
            recompute_invariant_ok=bool(recompute_invariant_ok),
            method_family=family,
            zcf_tokens_recomputed_analytical=(int(recompute_est) if family == "zcf" else None),
        )

    def poison_then_query(
        self,
        turn_1: str,
        turn_2: str,
        reference_tensors: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        self.query(turn_1)
        return self.query(turn_2, reference_tensors=reference_tensors)

    def reset(self) -> None:
        if hasattr(self._manager, "begin_fresh_query"):
            self._manager.begin_fresh_query()

    def _force_deterministic_generation(self) -> None:
        """Hard-force greedy decoding controls for publishable deterministic runs."""
        llm = getattr(self._manager, "llm", None)
        if llm is None:
            return
        gen_cfg = getattr(llm, "generation_config", None)
        if gen_cfg is not None:
            gen_cfg.temperature = 0.0
            gen_cfg.top_p = 1.0
            gen_cfg.do_sample = False
        setattr(llm, "_deterministic_eval", {"temperature": 0.0, "top_p": 1.0, "do_sample": False})

    def _extract_attention_tensors(self, stats: Dict[str, Any]) -> Optional[List[Any]]:
        # Best-effort extraction across possible stats layouts.
        for key in ("attention_tensors", "attentions", "per_layer_attentions", "raw_attentions"):
            value = stats.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                return value
            if torch is not None and torch.is_tensor(value):
                return [value]
        return None

    def _extract_index_set(self, stats: Dict[str, Any], candidates: Tuple[str, ...]) -> Optional[Set[int]]:
        for key in candidates:
            value = stats.get(key)
            if not isinstance(value, list):
                continue
            out: Set[int] = set()
            try:
                for item in value:
                    out.add(int(item))
            except Exception:
                continue
            return out
        return None

    def _is_degenerate_kl(self, value: Any) -> bool:
        if not isinstance(value, list):
            return True
        if len(value) < 2:
            return True
        try:
            vals = [float(x) for x in value]
        except Exception:
            return True
        return all(abs(x) < 1e-12 for x in vals)

    def compute_per_layer_attention_kl(
        self,
        baseline_stats: Dict[str, Any],
        cached_stats: Dict[str, Any],
    ) -> Optional[List[float]]:
        if torch is None:
            return None
        b_attn = self._extract_attention_tensors(baseline_stats)
        c_attn = self._extract_attention_tensors(cached_stats)
        if not b_attn or not c_attn:
            return None
        n = min(len(b_attn), len(c_attn))
        kls: List[float] = []
        eps = 1e-8
        for i in range(n):
            b = torch.as_tensor(b_attn[i], dtype=torch.float32)
            c = torch.as_tensor(c_attn[i], dtype=torch.float32)
            if b.dim() >= 3:
                # Average heads if present, then flatten token grid.
                if b.dim() > 2:
                    b = b.mean(dim=0)
                if c.dim() > 2:
                    c = c.mean(dim=0)
            b = b.reshape(-1)
            c = c.reshape(-1)
            b = b / (b.sum() + eps)
            c = c / (c.sum() + eps)
            kl = torch.sum(b * torch.log((b + eps) / (c + eps))).item()
            kls.append(float(kl))
        return kls

    def _derive_changed_indices_from_attention(
        self,
        baseline_stats: Dict[str, Any],
        cached_stats: Dict[str, Any],
        top_k: int,
    ) -> Optional[Set[int]]:
        if torch is None or top_k <= 0:
            return None
        b_attn = self._extract_attention_tensors(baseline_stats)
        c_attn = self._extract_attention_tensors(cached_stats)
        if not b_attn or not c_attn:
            return None
        n_layers = min(len(b_attn), len(c_attn))
        if n_layers == 0:
            return None
        deltas = None
        for i in range(n_layers):
            b = torch.as_tensor(b_attn[i], dtype=torch.float32).reshape(-1)
            c = torch.as_tensor(c_attn[i], dtype=torch.float32).reshape(-1)
            n = min(b.numel(), c.numel())
            if n == 0:
                continue
            d = torch.abs(b[:n] - c[:n])
            deltas = d if deltas is None else deltas[:n] + d[: min(deltas.numel(), n)]
        if deltas is None or deltas.numel() == 0:
            return None
        k = min(top_k, int(deltas.numel()))
        if k <= 0:
            return None
        idx = torch.topk(deltas, k).indices.tolist()
        return {int(x) for x in idx}

    def _derive_true_attention_oracle_indices(
        self,
        baseline_stats: Dict[str, Any],
        top_k: int,
    ) -> Optional[Set[int]]:
        """Top-K prompt indices by baseline attention mass (independent oracle)."""
        if torch is None or top_k <= 0:
            return None
        b_attn = self._extract_attention_tensors(baseline_stats)
        if not b_attn:
            return None
        acc = None
        for layer in b_attn:
            v = torch.as_tensor(layer, dtype=torch.float32).reshape(-1)
            if v.numel() == 0:
                continue
            acc = v if acc is None else acc[: min(acc.numel(), v.numel())] + v[: min(acc.numel(), v.numel())]
        if acc is None or acc.numel() == 0:
            return None
        k = min(top_k, int(acc.numel()))
        if k <= 0:
            return None
        idx = torch.topk(acc, k).indices.tolist()
        return {int(x) for x in idx}

    def extract_recompute_selected_indices(self, cached_stats: Dict[str, Any]) -> Optional[List[int]]:
        idx = self._extract_index_set(
            cached_stats,
            ("recompute_selected_indices", "selected_indices", "recompute_indices", "recomp_indices", "xattn_ids"),
        )
        if idx is None:
            return None
        return sorted(idx)

    def compute_heuristic_overlap_score(
        self,
        baseline_stats: Dict[str, Any],
        cached_stats: Dict[str, Any],
    ) -> Optional[float]:
        selected = self._extract_index_set(
            cached_stats,
            ("recompute_selected_indices", "selected_indices", "recompute_indices", "recomp_indices", "xattn_ids"),
        )
        changed = self._extract_index_set(
            baseline_stats,
            ("baseline_changed_indices", "changed_indices", "delta_indices", "changed_token_indices", "xattn_ids"),
        )
        if selected and not changed:
            changed = self._derive_true_attention_oracle_indices(
                baseline_stats, top_k=len(selected)
            )
            if changed is not None:
                baseline_stats["baseline_changed_indices"] = sorted(changed)
        if not selected or not changed:
            return None
        if len(changed) == 0:
            return None
        return float(len(selected & changed) / len(changed))

    def maybe_export_attention_maps(
        self,
        run_id: int,
        r_percent: float,
        trace_id: str,
        trace_index: int,
        baseline_stats: Dict[str, Any],
        cached_stats: Dict[str, Any],
    ) -> Optional[str]:
        """Save raw attention tensors for first N traces of each (run_id, r%)."""
        if torch is None or self._attention_export_dir is None:
            return None
        run_key = f"run{run_id}_r{int(r_percent)}"
        count = self._attention_export_counts.get(run_key, 0)
        if count >= self._max_attention_exports_per_run:
            return None

        b_attn = self._extract_attention_tensors(baseline_stats)
        c_attn = self._extract_attention_tensors(cached_stats)
        if not b_attn and not c_attn:
            return None

        self._attention_export_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._attention_export_dir / f"{run_key}_trace{trace_index:03d}_{trace_id}.pt"
        torch.save(
            {
                "run_id": run_id,
                "r_percent": r_percent,
                "trace_id": trace_id,
                "baseline_attentions": b_attn,
                "cached_attentions": c_attn,
            },
            out_path,
        )
        self._attention_export_counts[run_key] = count + 1
        return str(out_path)


def build_adapter(
    config: RuntimeConfig,
    attention_export_dir: Optional[Path] = None,
    max_attention_exports_per_run: int = 3,
    max_context_tokens: int = 12000,
) -> MethodAdapter:
    return MethodAdapter(
        build_manager_from_config(config),
        attention_export_dir=attention_export_dir,
        max_attention_exports_per_run=max_attention_exports_per_run,
        max_context_tokens=max_context_tokens,
    )
