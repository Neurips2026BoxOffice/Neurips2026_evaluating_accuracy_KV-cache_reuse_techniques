"""Factory helpers to instantiate legacy cache managers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Tuple

from cachebend.ncf.cutils import (
    build_model_eager,
    build_model_sdpa,
    build_tokenizer,
    ensure_tokenizer_model_alignment,
)
from cachebend.ncf.cblend_cache_faster import CacheBlendCache, CacheBlendCacheManager
from cachebend.ncf.cacheblend_warm import CacheBlendWarmCacheManager
from cachebend.ncf.cc_cache import CacheCraftCache, CacheCraftCacheManager, CacheCraftCacheV2
from cachebend.ncf.zcf_v10 import Baseline, BaselineNosep, ZCF, ZCFCache_V1
from cachebend.ncf.pseudo_LMCache_online_v4 import (
    LMCacheAlt,
    LMCacheAltBalanced,
    LMCacheAltMiddle,
    LMCacheAltMiddleBalanced,
    LMCacheAltTunable,
    LMCacheAltManager,
    LMCacheGuarded,
    LMCacheGuardedManager,
    LMCacheHybrid,
    LMCacheHybridManager,
    LMCacheMix,
    LMCacheMixManager,
    LMCacheOnline,
    LMCacheOnlineManager,
)
from cachebend.ncf.fusionrag import FusionRAGCache, FusionRAGCacheManager, FusionRAGWarmCacheManager


SUPPORTED_METHODS: Tuple[str, ...] = (
    "baseline",
    "cacheblend",
    "cacheblend_warm",
    "zcf",
    "cachecraft",
    "cachecraft_v2",
    "ccv2",
    "lmcache_online",
    "lmcache_alt",
    "lmcache_alt_balanced",
    "lmcache_alt_middle",
    "lmcache_alt_middle_balanced",
    "lmcache_alt_tunable",
    "lmcache_hybrid",
    "lmcache_mix",
    "lmcache_guarded",
    "fusionrag",
    "fusionrag_warm",
)


@dataclass(frozen=True)
class RuntimeConfig:
    method: str
    model_path: str
    device: str
    torch_dtype: str = "bfloat16"
    recompute_ratio: float = 0.15
    max_atom_copies: int = 1
    mchunk_size: int = 3
    cachecraft_n: int = 12
    loading_mode: str = "generator"
    stub_mode: bool = False
    warm_cache_path: str = ""


def build_runtime(
    model_path: str,
    device: str,
    torch_dtype: str = "bfloat16",
    require_eager: bool = False,
    attention_impl: str | None = None,
):
    force_eager = os.getenv("CACHEBEND_FORCE_EAGER_ATTENTION", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if force_eager or require_eager:
        llm = build_model_eager(model_path)
    else:
        llm = build_model_sdpa(model_path, torch_dtype=torch_dtype, attention_impl=attention_impl)
    tkn = build_tokenizer(model_path)
    ensure_tokenizer_model_alignment(llm, tkn)
    llm = llm.to(device)
    return llm, tkn


def build_manager(
    method: str,
    model_path: str,
    device: str,
    recompute_ratio: float,
    max_atom_copies: int,
    mchunk_size: int,
    cachecraft_n: int,
    loading_mode: str,
    stub_mode: bool,
) -> Any:
    cfg = RuntimeConfig(
        method=method.lower(),
        model_path=model_path,
        device=device,
        torch_dtype="bfloat16",
        recompute_ratio=recompute_ratio,
        max_atom_copies=max_atom_copies,
        mchunk_size=mchunk_size,
        cachecraft_n=cachecraft_n,
        loading_mode=loading_mode,
        stub_mode=stub_mode,
    )
    return build_manager_from_config(cfg)


def build_manager_from_config(cfg: RuntimeConfig) -> Any:
    if cfg.method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method {cfg.method}. Expected one of {SUPPORTED_METHODS}."
        )

    sdpa_methods = {
        "cacheblend",
        "cacheblend_warm",
        "fusionrag_warm",
        "lmcache_online",
        "lmcache_alt",
        "lmcache_alt_balanced",
        "lmcache_alt_middle",
        "lmcache_alt_middle_balanced",
        "lmcache_alt_tunable",
        "lmcache_hybrid",
        "lmcache_mix",
        "lmcache_guarded",
    }
    attention_impl = os.getenv("CACHEBEND_ATTENTION_IMPL", "").strip() or (
        "sdpa" if cfg.method in sdpa_methods else "flash_attention_2"
    )
    # CacheCraft still relies on the eager model path. Plain CacheBlend uses
    # SDPA because its partial recompute path needs explicit chunked masks that
    # flash attention rejects, while the current eager Ministral path is brittle.
    require_eager = cfg.method in {"cachecraft", "cachecraft_v2", "ccv2"}
    llm, tkn = build_runtime(
        cfg.model_path,
        cfg.device,
        torch_dtype=cfg.torch_dtype,
        require_eager=require_eager,
        attention_impl=attention_impl,
    )

    if cfg.method == "baseline":
        # Wired to BaselineNosep — the canonical paper2 baseline used by
        # boxoffice/intro_lb/fr_extended runners. The legacy `Baseline`
        # class (also exported above) is kept only because it's still
        # referenced by other code paths in this snapshot; nothing in
        # the boxoffice generation pipeline reaches it.
        return BaselineNosep(device=cfg.device, llm=llm, tkn=tkn)
    if cfg.method == "cacheblend":
        cache = CacheBlendCache(R=cfg.recompute_ratio)
        return CacheBlendCacheManager(
            device=cfg.device,
            cache=cache,
            llm=llm,
            tkn=tkn,
            stub_mode=cfg.stub_mode,
            loading_mode=cfg.loading_mode,
        )
    if cfg.method == "cacheblend_warm":
        if not cfg.warm_cache_path:
            raise ValueError("cacheblend_warm requires warm_cache_path")
        return CacheBlendWarmCacheManager(
            device=cfg.device,
            warm_cache_path=cfg.warm_cache_path,
            R=cfg.recompute_ratio,
            llm=llm,
            tkn=tkn,
            stub_mode=cfg.stub_mode,
            loading_mode=cfg.loading_mode,
        )
    if cfg.method == "zcf":
        cache = ZCFCache_V1(
            M=cfg.max_atom_copies,
            R=cfg.recompute_ratio,
            mchunk_size=cfg.mchunk_size,
        )
        return ZCF(
            device=cfg.device,
            cache=cache,
            llm=llm,
            tkn=tkn,
            stub_mode=cfg.stub_mode,
            loading_mode=cfg.loading_mode,
        )
    if cfg.method == "cachecraft":
        cache = CacheCraftCache(
            N=cfg.cachecraft_n,
            M=cfg.max_atom_copies,
            R=cfg.recompute_ratio,
        )
        return CacheCraftCacheManager(
            device=cfg.device,
            cache=cache,
            llm=llm,
            tkn=tkn,
            stub_mode=cfg.stub_mode,
            loading_mode=cfg.loading_mode,
        )
    if cfg.method in {"cachecraft_v2", "ccv2"}:
        cache = CacheCraftCacheV2(
            N=cfg.cachecraft_n,
            M=cfg.max_atom_copies,
            R=cfg.recompute_ratio,
        )
        return CacheCraftCacheManager(
            device=cfg.device,
            cache=cache,
            llm=llm,
            tkn=tkn,
            stub_mode=cfg.stub_mode,
            loading_mode=cfg.loading_mode,
        )
    if cfg.method == "lmcache_online":
        cache = LMCacheOnline(R=cfg.recompute_ratio)
        return LMCacheOnlineManager(
            device=cfg.device,
            cache=cache,
            llm=llm,
            tkn=tkn,
            stub_mode=cfg.stub_mode,
            loading_mode=cfg.loading_mode,
        )
    if cfg.method == "lmcache_alt":
        cache = LMCacheAlt(R=cfg.recompute_ratio)
        return LMCacheAltManager(
            device=cfg.device,
            cache=cache,
            llm=llm,
            tkn=tkn,
            stub_mode=cfg.stub_mode,
            loading_mode=cfg.loading_mode,
        )
    if cfg.method == "lmcache_alt_balanced":
        cache = LMCacheAltBalanced(R=cfg.recompute_ratio)
        return LMCacheAltManager(
            device=cfg.device,
            cache=cache,
            llm=llm,
            tkn=tkn,
            stub_mode=cfg.stub_mode,
            loading_mode=cfg.loading_mode,
        )
    if cfg.method == "lmcache_alt_middle":
        cache = LMCacheAltMiddle(R=cfg.recompute_ratio)
        return LMCacheAltManager(
            device=cfg.device,
            cache=cache,
            llm=llm,
            tkn=tkn,
            stub_mode=cfg.stub_mode,
            loading_mode=cfg.loading_mode,
        )
    if cfg.method == "lmcache_alt_middle_balanced":
        cache = LMCacheAltMiddleBalanced(R=cfg.recompute_ratio)
        return LMCacheAltManager(
            device=cfg.device,
            cache=cache,
            llm=llm,
            tkn=tkn,
            stub_mode=cfg.stub_mode,
            loading_mode=cfg.loading_mode,
        )
    if cfg.method == "lmcache_alt_tunable":
        cache = LMCacheAltTunable(R=cfg.recompute_ratio)
        return LMCacheAltManager(
            device=cfg.device,
            cache=cache,
            llm=llm,
            tkn=tkn,
            stub_mode=cfg.stub_mode,
            loading_mode=cfg.loading_mode,
        )
    if cfg.method == "lmcache_hybrid":
        cache = LMCacheHybrid(R=cfg.recompute_ratio)
        return LMCacheHybridManager(
            device=cfg.device,
            cache=cache,
            llm=llm,
            tkn=tkn,
            stub_mode=cfg.stub_mode,
            loading_mode=cfg.loading_mode,
        )
    if cfg.method == "lmcache_mix":
        cache = LMCacheMix(R=cfg.recompute_ratio)
        return LMCacheMixManager(
            device=cfg.device,
            cache=cache,
            llm=llm,
            tkn=tkn,
            stub_mode=cfg.stub_mode,
            loading_mode=cfg.loading_mode,
        )
    if cfg.method == "lmcache_guarded":
        cache = LMCacheGuarded(R=cfg.recompute_ratio)
        return LMCacheGuardedManager(
            device=cfg.device,
            cache=cache,
            llm=llm,
            tkn=tkn,
            stub_mode=cfg.stub_mode,
            loading_mode=cfg.loading_mode,
        )
    if cfg.method == "fusionrag_warm":
        if not cfg.warm_cache_path:
            raise ValueError("fusionrag_warm requires warm_cache_path")
        return FusionRAGWarmCacheManager(
            device=cfg.device,
            warm_cache_path=cfg.warm_cache_path,
            R=cfg.recompute_ratio,
            llm=llm,
            tkn=tkn,
            stub_mode=cfg.stub_mode,
            loading_mode=cfg.loading_mode,
        )
    cache = FusionRAGCache(R=cfg.recompute_ratio)
    return FusionRAGCacheManager(
        device=cfg.device,
        cache=cache,
        llm=llm,
        tkn=tkn,
        stub_mode=cfg.stub_mode,
        loading_mode=cfg.loading_mode,
    )
