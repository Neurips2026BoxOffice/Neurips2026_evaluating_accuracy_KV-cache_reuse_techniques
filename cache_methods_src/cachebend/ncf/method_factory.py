"""Factory helpers to instantiate legacy cache managers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Tuple

from cachebend.ncf.cutils import build_model_eager, build_model_sdpa, build_tokenizer
from cachebend.ncf.cblend_cache_faster import CacheBlendCache, CacheBlendCacheManager
from cachebend.ncf.cacheblend_warm import CacheBlendWarmCacheManager
from cachebend.ncf.cc_cache import (
    CacheCraftCache,
    CacheCraftCacheManager,
    CacheCraftCacheV3,
    CacheCraftCacheV3DiffKVManager,
    CacheCraftCacheV3QManager,
)
from cachebend.ncf.zcf_v10 import Baseline, BaselineNosep, ZCF, ZCFCache_V1
from cachebend.ncf.pseudo_LMCache_online_v4 import LMCacheOnline, LMCacheOnlineManager
from cachebend.ncf.fusionrag import FusionRAGCache, FusionRAGCacheManager, FusionRAGWarmCacheManager
from cachebend.ncf.onthefly_infusion import CacheBlendOnTheFlyManager, FusionRAGOnTheFlyManager


SUPPORTED_METHODS: Tuple[str, ...] = (
    "baseline",
    "baseline_nosep",
    "cacheblend",
    "cacheblend_warm",
    "cacheblend_onthefly",
    "zcf",
    "cachecraft",
    "cachecraft_v3_diffkv",
    "cachecraft_v3_q",
    "lmcache_online",
    "fusionrag",
    "fusionrag_warm",
    "fusionrag_onthefly",
)


@dataclass(frozen=True)
class RuntimeConfig:
    method: str
    model_path: str
    device: str
    torch_dtype: str = "float16"
    recompute_ratio: float = 0.15
    max_atom_copies: int = 1
    mchunk_size: int = 3
    cachecraft_n: int = 12
    loading_mode: str = "generator"
    stub_mode: bool = False
    warm_cache_path: str = ""
    top_k_neighbors: int = 5


def build_runtime(model_path: str, device: str, torch_dtype: str = "float16"):
    force_eager = os.getenv("CACHEBEND_FORCE_EAGER_ATTENTION", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if force_eager:
        llm = build_model_eager(model_path).to(device)
    else:
        llm = build_model_sdpa(model_path, torch_dtype=torch_dtype).to(device)
    tkn = build_tokenizer(model_path)
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
        torch_dtype="float16",
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

    llm, tkn = build_runtime(cfg.model_path, cfg.device, torch_dtype=cfg.torch_dtype)

    if cfg.method == "baseline":
        return Baseline(device=cfg.device, llm=llm, tkn=tkn)
    if cfg.method == "baseline_nosep":
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
    if cfg.method == "cacheblend_onthefly":
        return CacheBlendOnTheFlyManager(
            device=cfg.device,
            R=cfg.recompute_ratio,
            top_k=cfg.top_k_neighbors,
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
    if cfg.method == "cachecraft_v3_diffkv":
        cache = CacheCraftCacheV3(
            N=cfg.cachecraft_n,
            M=cfg.max_atom_copies,
            R=cfg.recompute_ratio,
        )
        return CacheCraftCacheV3DiffKVManager(
            device=cfg.device,
            cache=cache,
            llm=llm,
            tkn=tkn,
            stub_mode=cfg.stub_mode,
            loading_mode=cfg.loading_mode,
        )
    if cfg.method == "cachecraft_v3_q":
        cache = CacheCraftCacheV3(
            N=cfg.cachecraft_n,
            M=cfg.max_atom_copies,
            R=cfg.recompute_ratio,
        )
        return CacheCraftCacheV3QManager(
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
    if cfg.method == "fusionrag_onthefly":
        return FusionRAGOnTheFlyManager(
            device=cfg.device,
            R=cfg.recompute_ratio,
            top_k=cfg.top_k_neighbors,
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
