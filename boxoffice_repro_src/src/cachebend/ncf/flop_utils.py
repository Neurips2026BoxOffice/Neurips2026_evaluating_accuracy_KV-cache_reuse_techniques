"""Phase-4 FLOP utilities (GQA-aware) and model-constant helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ModelConstants:
    model_type: str
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    head_dim: int
    max_position_embeddings: Optional[int] = None
    torch_dtype: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _infer_kv_heads(cfg: Any, n_heads: int) -> int:
    kv = getattr(cfg, "num_key_value_heads", None)
    if kv is None:
        # Some configs expose grouped-query count with different names.
        kv = getattr(cfg, "multi_query_group_num", None)
    if kv is None:
        kv = n_heads
    kv = _to_int(kv, n_heads)
    if kv <= 0:
        kv = n_heads
    return kv


def model_constants_from_config(cfg: Any) -> ModelConstants:
    n_layers = _to_int(getattr(cfg, "num_hidden_layers", 0))
    d_model = _to_int(getattr(cfg, "hidden_size", 0))
    n_heads = _to_int(getattr(cfg, "num_attention_heads", 0))
    n_kv_heads = _infer_kv_heads(cfg, n_heads)
    d_ffn = _to_int(getattr(cfg, "intermediate_size", 0))
    head_dim = _to_int(getattr(cfg, "head_dim", 0))
    if head_dim <= 0 and n_heads > 0 and d_model > 0:
        head_dim = d_model // n_heads
    return ModelConstants(
        model_type=str(getattr(cfg, "model_type", "unknown")),
        num_hidden_layers=n_layers,
        hidden_size=d_model,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        intermediate_size=d_ffn,
        head_dim=head_dim,
        max_position_embeddings=getattr(cfg, "max_position_embeddings", None),
        torch_dtype=str(getattr(cfg, "torch_dtype", None)) if getattr(cfg, "torch_dtype", None) is not None else None,
    )


def model_constants_from_model_path(model_path: str) -> ModelConstants:
    from transformers import AutoConfig  # local import to keep import-light module load

    cfg = AutoConfig.from_pretrained(model_path)
    return model_constants_from_config(cfg)


def dump_model_constants_json(model_constants: Dict[str, Any], out_path: Path) -> None:
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(model_constants, f, indent=2, sort_keys=True)


def _full_components(n_tokens: int, c: ModelConstants) -> Dict[str, int]:
    n = max(0, int(n_tokens))
    l = c.num_hidden_layers
    d_model = c.hidden_size
    n_heads = c.num_attention_heads
    n_kv = c.num_key_value_heads
    d_head = c.head_dim
    d_ffn = c.intermediate_size

    flops_proj_layer = 2 * n * d_model * (2 * n_heads * d_head + 2 * n_kv * d_head)
    flops_attn_layer = 4 * n * n * n_heads * d_head
    flops_ffn_layer = 6 * n * d_model * d_ffn

    return {
        "proj": int(l * flops_proj_layer),
        "attn": int(l * flops_attn_layer),
        "ffn": int(l * flops_ffn_layer),
    }


def _cached_components(
    total_tokens: int,
    recompute_tokens: int,
    blend_tokens: int,
    c: ModelConstants,
    blend_includes_ffn: bool,
) -> Dict[str, int]:
    n = max(0, int(total_tokens))
    r = max(0, int(recompute_tokens))
    b = max(0, int(blend_tokens))

    l = c.num_hidden_layers
    d_model = c.hidden_size
    n_heads = c.num_attention_heads
    n_kv = c.num_key_value_heads
    d_head = c.head_dim
    d_ffn = c.intermediate_size

    # Full-forward tokens (R): projections + full-sequence attention + FFN.
    proj_r = l * (2 * r * d_model * (2 * n_heads * d_head + 2 * n_kv * d_head))
    attn_r = l * (4 * r * n * n_heads * d_head)
    ffn_r = l * (6 * r * d_model * d_ffn)

    # Blend/fusion tokens (B): attention-only by default, optionally include FFN.
    proj_b = 0
    attn_b = l * (4 * b * n * n_heads * d_head)
    ffn_b = l * (6 * b * d_model * d_ffn) if blend_includes_ffn else 0

    return {
        "proj": int(proj_r + proj_b),
        "attn": int(attn_r + attn_b),
        "ffn": int(ffn_r + ffn_b),
    }


def compute_flop_metrics(
    cost_metrics: Dict[str, Any],
    constants: ModelConstants,
    blend_includes_ffn: bool = False,
    computation_notes: str = "",
) -> Dict[str, Any]:
    n = _to_int(cost_metrics.get("tokens_total_prompt"), 0)
    r = _to_int(cost_metrics.get("tokens_recomputed"), 0)
    b = _to_int(cost_metrics.get("tokens_blend_or_fusion"), 0)

    full = _full_components(n, constants)
    cached = _cached_components(
        total_tokens=n,
        recompute_tokens=r,
        blend_tokens=b,
        c=constants,
        blend_includes_ffn=blend_includes_ffn,
    )
    full_total = full["proj"] + full["attn"] + full["ffn"]
    cached_total = cached["proj"] + cached["attn"] + cached["ffn"]
    f_norm = (float(cached_total) / float(full_total)) if full_total > 0 else 1.0

    notes = computation_notes.strip() or "GQA-aware FLOPs from token-level cost metrics."
    return {
        "flops_full_recompute": int(full_total),
        "flops_cached_method": int(cached_total),
        "flops_attn_component": int(cached["attn"]),
        "flops_ffn_component": int(cached["ffn"]),
        "flops_proj_component": int(cached["proj"]),
        "F_norm": float(f_norm),
        "blend_includes_ffn": bool(blend_includes_ffn),
        "gqa_n_kv_heads": int(constants.num_key_value_heads),
        "computation_notes": notes,
    }
