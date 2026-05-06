"""
LMCache Online Implementation

A unified caching strategy that handles both hits and misses in a single blending pass.
- Hits: Use cached KV, recompute top R% globally by ΔK
- Misses: 100% recompute, save after blending

Usage:
    cache = LMCacheOnline(R=0.1)
    manager = LMCacheOnlineManager(device="npu:0", cache=cache, llm=llm, tkn=tokenizer)
    response, used_cache, stats = manager.new_query(prompt)
"""

from typing import List, Dict, Optional
import logging
import gc

import torch
from transformers import (
    DynamicCache,
    AutoModelForCausalLM,
    AutoConfig,
)

try:
    import torch_npu
except ImportError:
    pass

from cachebend.utils import Timer
from cachebend.ncf.cutils import (
    Chunk, PosChunk, CachedChunk, 
    to_str_prompt, build_tokenizer, do_query_with_state,
    chash, chunks_from_tokenss, full_prefill_cache, save_snapshot
)
from cachebend.ncf.cblend_cache_faster import CacheBlendCache
from cachebend.llm_docs import generator_build_blent_cache5, generator_build_blent_cache_unified  # unified blender



model_path = "/data/weights/llama3.1-8BI"

cache_log = logging.getLogger("cacheblend")

def isdbg():
    return cache_log.isEnabledFor(logging.DEBUG)


def _dynamic_cache_layer_kv(cache: DynamicCache, layer_idx: int):
    """Compat helper for older/newer DynamicCache layouts."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]

    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        if hasattr(layer, "keys") and hasattr(layer, "values"):
            return layer.keys, layer.values

    layer = cache[layer_idx]
    if isinstance(layer, (tuple, list)) and len(layer) >= 2:
        return layer[0], layer[1]

    raise TypeError(f"Unsupported DynamicCache layer format at index {layer_idx}: {type(layer)!r}")


class LMCacheOnline(CacheBlendCache):
    """
    LMCache Online: CacheBlend with online cache population.
    - Hits: Use cached KV, recompute top R% globally by ΔK
    - Misses: 100% recompute, save after blending
    """
    
    def __init__(self, R: float):
        super().__init__(R)
        self.use_piaffe = False

    def maybe_add(self, chunks: List[PosChunk], tokenizer, stats=None) -> List[PosChunk]:
        """No prefill step — misses are handled during blending."""
        if stats is not None:
            stats["added_chunk_details"] = []
            stats["count_added"] = 0
        return []

    def recomp_tkns_for_chunks(self, chunks: List[PosChunk]) -> Dict[int, float]:
        """
        Returns per-chunk ratios:
        - Misses: 1.0 (100% recompute)
        - Hits: self.recomp_ratio
        """
        ratios = {}
        for chunk in chunks:
            if chunk.cid in self.chunk_to_cchunk:
                ratios[chunk.cid] = self.recomp_ratio
            else:
                ratios[chunk.cid] = 1.0
        return ratios

    def reuse_cache(self, chunks, llm, tokenizer, stats, stub_mode=False, loading_mode: str = "generator"):
        if stub_mode:
            return None
        
        
        cache_size_before = len(self.chunk_to_cchunk)
        missing_cids = {c.cid for c in chunks if c.cid not in self.chunk_to_cchunk}
        
        if stats:
            stats["cache_size_before_query"] = len(self.chunk_to_cchunk)
            stats["num_used_cids"] = len(chunks)
            stats["used_cids"] = [c.cid for c in chunks]
            stats["num_cached_chunks"] = len(chunks) - len(missing_cids)
            stats["num_fresh_chunks"] = len(missing_cids)
            stats["total_chunks"] = len(chunks)
            
            stats["hit_cids"] = [c.cid for c in chunks if c.cid in self.chunk_to_cchunk]
            stats["miss_cids"] = list(missing_cids)
            
            total_tokens = sum(len(c) for c in chunks)
            miss_tokens = sum(len(c) for c in chunks if c.cid in missing_cids)
            hit_tokens = total_tokens - miss_tokens
            
            stats["token_composition"] = {
                "total_tokens": total_tokens,
                "miss_tokens": miss_tokens,
                "hit_tokens": hit_tokens,
                "miss_token_ratio": miss_tokens / total_tokens if total_tokens > 0 else 0,
                "chunk_hit_ratio": (len(chunks) - len(missing_cids)) / len(chunks) if len(chunks) > 0 else 0
            }

        recompute_ratios = self.recomp_tkns_for_chunks(chunks)
        
        if stats:
            stats["recompute_ratios_sent"] = recompute_ratios

        doc_ids = torch.cat([c.tokens_() for c in chunks], dim=0).unsqueeze(0).to(llm.device, non_blocking=True)

        cachedb = []
        for chunk in chunks:
            if chunk.cid in missing_cids:
                cws = self._create_dummy_chunk(chunk, llm)
            else:
                cws = self.chunk_to_cchunk[chunk.cid]
            
            
            
            orig_pos = torch.arange(cws.start, cws.end)

            cachedb.append((
                chunk.cid,
                chunk.tokens_(),
                cws,
                cws,
                orig_pos 
            ))

        with Timer("blend", stats=stats):
            blended_cache = generator_build_blent_cache_unified(
                llm.model,
                cachedb,
                doc_ids,
                recompute_ratios,
                past_key_value=None,
                stats=stats,
                use_piaffe=self.use_piaffe
            )

        if missing_cids:
            with Timer("snapshot_misses", stats=stats):
                self._save_miss_chunks(chunks, missing_cids, blended_cache, llm, stats=stats)
            
            if stats:
                cache_size_after = len(self.chunk_to_cchunk)
                stats["cache_size_after_query"] = cache_size_after
                expected_growth = len(missing_cids)
                actual_growth = cache_size_after - cache_size_before
                
                stats["cache_cids_after_query"] = list(self.chunk_to_cchunk.keys())
                
                if actual_growth != expected_growth:
                    stats["cache_growth_mismatch"] = {
                        "expected": expected_growth,
                        "actual": actual_growth,
                        "deficit": expected_growth - actual_growth
                    }
                if cache_size_after < cache_size_before:
                    stats["CACHE_SHRINKAGE_DETECTED"] = {
                        "before": cache_size_before,
                        "after": cache_size_after,
                        "lost_chunks": cache_size_before - cache_size_after
                    }

        return blended_cache

    def _create_dummy_chunk(self, chunk: PosChunk, llm) -> CachedChunk:
        """Create zeros placeholder for misses."""
        num_layers = llm.config.num_hidden_layers
        n_kv_heads = getattr(llm.config, "num_key_value_heads", llm.config.num_attention_heads)
        head_dim = llm.config.hidden_size // llm.config.num_attention_heads

        c_states = []
        for _ in range(num_layers):
            k = torch.zeros((n_kv_heads, len(chunk), head_dim), dtype=llm.dtype, device="cpu")
            v = torch.zeros((n_kv_heads, len(chunk), head_dim), dtype=llm.dtype, device="cpu")
            c_states.append((k, v))

        return CachedChunk(chunk.tokens_(), 0, len(chunk), c_states)

    def _save_miss_chunks(
        self, 
        chunks: List[PosChunk], 
        missing_cids: set, 
        blended_cache: DynamicCache, 
        llm,
        stats = None
    ):
        """Extract and save miss chunks from blended cache with correct positional metadata."""
       
        # TRACK ABSOLUTE POSITION
        
        current_idx = 0 
        num_layers = llm.config.num_hidden_layers
        
        if stats is not None:
            stats["save_attempt_count"] = len(missing_cids)
            stats["save_attempt_cids"] = list(missing_cids)

        # Start all copies 
        pending_chunks = [] 

        for chunk in chunks:
            chunk_len = len(chunk)
            
            if chunk.cid in missing_cids:
                c_states = []
                for layer_idx in range(num_layers):
                    k_layer, v_layer = _dynamic_cache_layer_kv(blended_cache, layer_idx)
                    
                    # Extract slice 
                    k_slice = k_layer[:, :, current_idx:current_idx+chunk_len, :].to("cpu", non_blocking=True)
                    v_slice = v_layer[:, :, current_idx:current_idx+chunk_len, :].to("cpu", non_blocking=True)
                    
                    if k_slice.dim() == 4:
                        k_slice = k_slice.squeeze(0)
                        v_slice = v_slice.squeeze(0)
                    
                    c_states.append((k_slice, v_slice))
                
                # Store current_idx alongside the data
                pending_chunks.append((chunk, chunk.cid, chunk_len, c_states, current_idx))
            
            current_idx += chunk_len
        
        #  Sync
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        elif hasattr(torch, "cuda"):
            torch.cuda.synchronize()
        
        # Save
        saved_count = 0
        for chunk, cid, chunk_len, c_states, abs_start in pending_chunks:
        
            new_cchunk = CachedChunk(chunk.tokens_(), abs_start, abs_start + chunk_len, c_states)
            
            self.chunk_to_cchunk[cid] = new_cchunk
            saved_count += 1
        
        # Final verification
        if stats is not None:
            stats["chunks_actually_saved"] = saved_count
            missing_after_save = [cid for cid in missing_cids if cid not in self.chunk_to_cchunk]
            if missing_after_save:
                stats["save_failure_cids"] = missing_after_save
                stats["save_failure_count"] = len(missing_after_save)
            else:
                stats["save_success"] = True


class LMCacheOnlineManager:
    """
    Manager for LMCache Online — same interface as CacheBlendCacheManager.
    """
    
    def __init__(
        self, 
        device: str, 
        cache: LMCacheOnline, 
        llm=None, 
        tkn=None, 
        stub_mode=False, 
        loading_mode: str = "generator", 
        no_reuse=False
    ):
        self.loading_mode = loading_mode
        self.device = torch.device(device)
        self.stub_mode = stub_mode
        self.no_reuse = no_reuse
        
        if self.stub_mode:
            cache_log.info("LMCacheOnline running in STUB MODE (Inference Disabled)")
        cache_log.info(f"LMCacheOnline initialized with loading mode: {self.loading_mode}")
        
        if not llm:
            assert not tkn
            config = AutoConfig.from_pretrained(model_path)
            config.sliding_window = None
            config._attn_implementation = "eager"
            self.llm = AutoModelForCausalLM.from_pretrained(model_path, config=config)
            self.llm.to(self.device)
            self.llm.to(torch.bfloat16)
            self.tokenizer = build_tokenizer(model_path)
        else:
            self.llm = llm
            self.tokenizer = tkn
            
        self.llm.eval()
        self.model = self.llm.model
        self.cache = cache

        self.sep = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.int64)
        self.ascii_sep = self.tokenizer.sep_token
        self.n_q = 0
        self.n_r = 0
        self._last_stats = None

    def enable_piaffe(self):
        self.cache.use_piaffe = True
        
    def tokenize(self, doc: str, use_special_tokens: bool = False) -> torch.Tensor:
        return self.tokenizer(
            doc, return_tensors="pt", add_special_tokens=use_special_tokens
        )["input_ids"][0].to(torch.int64)

    def to_str_prompt(self, lis) -> str:
        return to_str_prompt(self.tokenizer, self.ascii_sep, lis)

    @torch.no_grad()
    def new_query(self, prompt: torch.Tensor, force_full=False, snapshot_path=None, reference_tensors=None):
        self.n_q += 1
        stats = {}
        self._last_stats = stats
        
        if isdbg():
            cache_log.debug(f"New query {len(prompt)=} {self.tokenizer.decode(prompt[:15])=}")
        
        chunks = chunks_from_tokenss(prompt, self.sep)
        chunks, query = chunks[:-1], chunks[-1].tokens
        
        if isdbg():
            cache_log.debug(f"{chunks=} {query=} {len(query)=}")

        attn_capture = bool(force_full or reference_tensors is not None)
        if force_full:
            if self.stub_mode:
                return "STUB_FULL", False, stats
            cache, _ = full_prefill_cache(chunks, self.llm, output_attentions=attn_capture)
            return do_query_with_state(
                self.llm, self.tokenizer, cache, query, 
                output_attentions=attn_capture, stats=stats
            ), False, stats

        with Timer("maybe_add", stats):
            _ = self.cache.maybe_add(chunks, self.tokenizer, stats=stats)

        if self.stub_mode:
            query_cids = [c.cid for c in chunks]
            cached_cids = [cid for cid in query_cids if cid in self.cache.chunk_to_cchunk]
            fresh_cids = [cid for cid in query_cids if cid not in self.cache.chunk_to_cchunk]
            
            stats.update({
                "num_fresh": len(fresh_cids),
                "num_cached": len(cached_cids),
                "hit_rate": len(cached_cids) / len(chunks) if chunks else 0,
                "total_chunks": len(chunks),
                "num_fresh_chunks": len(fresh_cids),
                "num_cached_chunks": len(cached_cids),
            })
            return "STUB_RESPONSE", True, stats

        with Timer("reuse_cache", stats):
            cache = self.cache.reuse_cache(
                chunks, self.llm, self.tokenizer, stats,
                stub_mode=self.stub_mode, loading_mode=self.loading_mode
            )

        with Timer("query", stats):
            if cache:
                try:
                    if snapshot_path:
                        try:
                            meta = [{"cid": c.cid, "len": len(c)} for c in chunks]
                            save_snapshot(snapshot_path, "lmcache_online", cache, query, meta)
                        except Exception:
                            cache_log.warning("Cache snapshot saving failed.")
                    
                    return do_query_with_state(
                        self.llm, self.tokenizer, cache, query,
                        output_attentions=attn_capture, stats=stats
                    ), True, stats
                    
                finally:
                    if hasattr(torch, "npu"):
                        torch.npu.empty_cache()
                    elif hasattr(torch, "cuda"):
                        torch.cuda.empty_cache()
                    gc.collect()
            else:
                cache, _ = full_prefill_cache(chunks, self.llm, output_attentions=False)
                return do_query_with_state(
                    self.llm, self.tokenizer, cache, query,
                    output_attentions=attn_capture, stats=stats
                ), False, stats

    def begin_fresh_query(self):
        """
        Start a new query without discarding the online cache state.

        LMCache-online is meant to accumulate reusable chunks across queries in
        a run; resetting here would turn the method into an all-miss baseline.
        """
        self._last_stats = None
