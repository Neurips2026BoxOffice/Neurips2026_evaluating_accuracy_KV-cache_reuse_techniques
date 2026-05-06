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

from typing import List, Dict, Optional, Tuple
import logging
import gc
import os
from functools import partial

import torch
import torch.nn.functional as F
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
from cachebend.ncf.fusionrag import QueryGuidedSelector
from cachebend.llm_docs import (
    generator_build_blent_cache5,
    generator_build_blent_cache_unified,
    positional_encoder,
    reverse_positional_encoder,
    apply_rotary_pos_emb,
)  # unified blender



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


class LMCacheAlt(LMCacheOnline):
    """
    LMCache online with FusionRAG-style query-guided hit-token selection.

    Miss chunks are still fully recomputed and saved exactly as in LMCacheOnline.
    The only intended change is cached-hit token selection: top R% is selected
    by final-layer query-to-context attention instead of layer-1 ΔK.
    """

    method_name = "lmcache_alt"
    selection_layer = "final"
    balance_per_chunk = False

    def __init__(self, R: float):
        super().__init__(R)
        self.selector = QueryGuidedSelector(recomp_ratio=R)
        selector_layer = os.getenv("CACHEBEND_LMCACHE_SELECTOR_LAYER", "").strip()
        if selector_layer:
            self.selection_layer = selector_layer
        balance = os.getenv("CACHEBEND_LMCACHE_BALANCE_PER_CHUNK", "").strip().lower()
        if balance:
            self.balance_per_chunk = balance in {"1", "true", "yes", "on"}

    def set_positional_encoders(self, model):
        self.selector.set_positional_encoders(
            positional_encoder=partial(positional_encoder, model),
            reverse_positional_encoder=partial(reverse_positional_encoder, model),
        )

    def selection_layer_index(self, llm) -> int:
        num_layers = int(getattr(llm.config, "num_hidden_layers"))
        if self.selection_layer == "final":
            return num_layers - 1
        if self.selection_layer == "middle":
            return num_layers // 2
        if isinstance(self.selection_layer, str) and self.selection_layer.isdigit():
            return max(0, min(num_layers - 1, int(self.selection_layer)))
        if isinstance(self.selection_layer, int):
            return max(0, min(num_layers - 1, int(self.selection_layer)))
        raise RuntimeError(f"Unsupported LMCacheAlt selection layer {self.selection_layer!r}")

    def _select_recompute_indices_query_guided(
        self,
        chunks: List[PosChunk],
        missing_cids: set,
        query_layer_q: torch.Tensor,
        llm,
        stats=None,
    ) -> List[int]:
        device = llm.device
        n_q_heads = query_layer_q.shape[0]
        head_dim = query_layer_q.shape[-1]
        layer_idx = self.selection_layer_index(llm)

        selected_indices: List[int] = []
        processed_keys = []
        local_to_abs: List[int] = []
        processed_ranges: List[Tuple[int, int, int]] = []
        per_chunk_distribution: List[int] = []
        processed_token_offset = 0

        current_idx = 0
        hit_tokens = 0
        miss_tokens = 0
        hit_chunks = 0
        miss_chunks = 0

        for chunk in chunks:
            chunk_len = len(chunk)
            start = current_idx
            end = current_idx + chunk_len

            if chunk.cid in missing_cids:
                selected_indices.extend(range(start, end))
                per_chunk_distribution.append(chunk_len)
                miss_tokens += chunk_len
                miss_chunks += 1
                current_idx = end
                continue

            cws = self.chunk_to_cchunk[chunk.cid]
            chunk_k = cws.states[layer_idx][0].to(device, non_blocking=True)
            while chunk_k.dim() > 3:
                chunk_k = chunk_k.squeeze(0)

            orig_pos = torch.arange(int(cws.start), int(cws.end), device=device)
            if orig_pos.numel() != chunk_len:
                orig_pos = torch.arange(0, chunk_len, device=device)
            new_pos = torch.arange(start, end, device=device)
            chunk_k = self.selector.rerope_key(chunk_k, orig_pos, new_pos)

            n_kv_heads = chunk_k.shape[0]
            if n_q_heads != n_kv_heads:
                repeat_factor = n_q_heads // n_kv_heads
                chunk_k = chunk_k.repeat_interleave(repeat_factor, dim=0)

            processed_keys.append(chunk_k)
            local_to_abs.extend(range(start, end))
            processed_ranges.append((processed_token_offset, processed_token_offset + chunk_len, len(per_chunk_distribution)))
            processed_token_offset += chunk_len
            per_chunk_distribution.append(0)
            hit_tokens += chunk_len
            hit_chunks += 1
            current_idx = end

        hit_budget = int(hit_tokens * self.recomp_ratio)
        selected_hit_indices: List[int] = []
        if processed_keys and hit_budget > 0:
            all_keys = torch.cat(processed_keys, dim=1)
            attn_logits = torch.matmul(
                query_layer_q.float(),
                all_keys.float().transpose(-2, -1),
            ) / (head_dim ** 0.5)
            attn_weights = F.softmax(attn_logits, dim=-1)
            critical_scores = attn_weights.sum(dim=(0, 1))
            if self.balance_per_chunk:
                top_indices = []
                for local_start, local_end, _chunk_idx in processed_ranges:
                    chunk_len = local_end - local_start
                    chunk_budget = min(chunk_len, int(chunk_len * self.recomp_ratio))
                    if chunk_budget <= 0:
                        continue
                    local_scores = critical_scores[local_start:local_end]
                    chunk_top = torch.topk(local_scores, chunk_budget).indices.cpu().tolist()
                    top_indices.extend(local_start + int(i) for i in chunk_top)
                hit_budget = len(top_indices)
            else:
                hit_budget = min(hit_budget, critical_scores.shape[0])
                top_indices = torch.topk(critical_scores, hit_budget).indices.cpu().tolist()
            selected_hit_indices = [local_to_abs[int(i)] for i in top_indices]
            selected_indices.extend(selected_hit_indices)

            # Convert absolute selected hit indices back into per-prompt-chunk counts.
            for abs_idx in selected_hit_indices:
                running = 0
                for chunk_idx, chunk in enumerate(chunks):
                    next_running = running + len(chunk)
                    if running <= abs_idx < next_running:
                        per_chunk_distribution[chunk_idx] += 1
                        break
                    running = next_running

        selected_indices = sorted(set(selected_indices))
        if stats is not None:
            stats["mode"] = self.method_name
            stats["num_miss_tokens"] = miss_tokens
            stats["num_hit_tokens"] = hit_tokens
            stats["num_miss_chunks"] = miss_chunks
            stats["num_hit_chunks"] = hit_chunks
            stats["hit_budget"] = hit_budget
            stats["num_selected"] = len(selected_indices)
            stats["recompute_selected_indices"] = [int(x) for x in selected_indices]
            stats[f"{self.method_name}_selected_hit_indices"] = [int(x) for x in sorted(selected_hit_indices)]
            stats[f"{self.method_name}_per_chunk_distribution"] = per_chunk_distribution
            # Keep the original diagnostic key for existing analysis scripts.
            if self.method_name == "lmcache_alt":
                stats["lmcache_alt_selected_hit_indices"] = [int(x) for x in sorted(selected_hit_indices)]
                stats["lmcache_alt_per_chunk_distribution"] = per_chunk_distribution
            layer_label = "final" if self.selection_layer == "final" else str(self.selection_layer)
            balance_label = "per-chunk" if self.balance_per_chunk else "global"
            stats[f"{self.method_name}_selection_rule"] = (
                f"top R% cached-hit tokens by {layer_label}-layer query-to-context attention "
                f"with {balance_label} selection; all miss tokens recomputed"
            )
            stats[f"{self.method_name}_selection_layer_idx"] = int(layer_idx)
            stats[f"{self.method_name}_balance_per_chunk"] = bool(self.balance_per_chunk)

        return selected_indices

    def reuse_cache_with_query(
        self,
        chunks,
        query_layer_q,
        llm,
        tokenizer,
        stats,
        stub_mode=False,
        loading_mode: str = "generator",
    ):
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
                "chunk_hit_ratio": (len(chunks) - len(missing_cids)) / len(chunks) if len(chunks) > 0 else 0,
            }

        recompute_ratios = self.recomp_tkns_for_chunks(chunks)
        if stats:
            stats["recompute_ratios_sent"] = recompute_ratios

        with Timer("select_recompute_tokens", stats=stats):
            recompute_indices = self._select_recompute_indices_query_guided(
                chunks=chunks,
                missing_cids=missing_cids,
                query_layer_q=query_layer_q,
                llm=llm,
                stats=stats,
            )

        doc_ids = torch.cat([c.tokens_() for c in chunks], dim=0).unsqueeze(0).to(llm.device, non_blocking=True)
        cachedb = []
        for chunk in chunks:
            if chunk.cid in missing_cids:
                cws = self._create_dummy_chunk(chunk, llm)
            else:
                cws = self.chunk_to_cchunk[chunk.cid]
            orig_pos = torch.arange(cws.start, cws.end)
            cachedb.append((chunk.cid, chunk.tokens_(), cws, cws, orig_pos))

        with Timer("blend", stats=stats):
            blended_cache = generator_build_blent_cache5(
                llm.model,
                cachedb,
                doc_ids,
                recompute_ratios,
                past_key_value=None,
                stats=stats,
                use_piaffe=self.use_piaffe,
                precomputed_recompute_indices=recompute_indices,
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
                        "deficit": expected_growth - actual_growth,
                    }
                if cache_size_after < cache_size_before:
                    stats["CACHE_SHRINKAGE_DETECTED"] = {
                        "before": cache_size_before,
                        "after": cache_size_after,
                        "lost_chunks": cache_size_before - cache_size_after,
                    }

        return blended_cache


class LMCacheAltBalanced(LMCacheAlt):
    """LMCacheAlt with query-guided selection balanced within each cached chunk."""

    method_name = "lmcache_alt_balanced"
    balance_per_chunk = True


class LMCacheAltMiddle(LMCacheAlt):
    """LMCacheAlt using middle-layer query/key relevance instead of final-layer relevance."""

    method_name = "lmcache_alt_middle"
    selection_layer = "middle"


class LMCacheAltMiddleBalanced(LMCacheAltMiddle):
    """Middle-layer query-guided selection with per-chunk budget balancing."""

    method_name = "lmcache_alt_middle_balanced"
    balance_per_chunk = True


class LMCacheAltTunable(LMCacheAlt):
    """LMCacheAlt whose layer/balancing are controlled by CACHEBEND_LMCACHE_* env vars."""

    method_name = "lmcache_alt_tunable"


class LMCacheHybrid(LMCacheAlt):
    """
    LMCache online with a hybrid stale-and-relevant token selector.

    The LMCache blending path still computes fresh layer-1 K and cached layer-1
    K. Selection changes from pure diffK to:
        minmax(layer-1 diffK) * minmax(final-layer query attention)
    over cached-hit tokens. Miss chunks remain 100% recomputed and saved.
    """

    method_name = "lmcache_hybrid"
    selection_mode = "hybrid"

    def _compute_query_relevance_scores(
        self,
        chunks: List[PosChunk],
        missing_cids: set,
        query_final_q: torch.Tensor,
        llm,
        stats=None,
    ) -> torch.Tensor:
        device = llm.device
        n_q_heads = query_final_q.shape[0]
        head_dim = query_final_q.shape[-1]
        total_tokens = sum(len(chunk) for chunk in chunks)

        score_tensor = torch.zeros(total_tokens, device=device, dtype=torch.float32)
        processed_keys = []
        local_to_abs: List[int] = []
        per_chunk_hit_tokens: List[int] = []

        current_idx = 0
        for chunk in chunks:
            chunk_len = len(chunk)
            start = current_idx
            end = current_idx + chunk_len
            current_idx = end

            if chunk.cid in missing_cids:
                per_chunk_hit_tokens.append(0)
                continue

            cws = self.chunk_to_cchunk[chunk.cid]
            chunk_k = cws.states[-1][0].to(device, non_blocking=True)
            while chunk_k.dim() > 3:
                chunk_k = chunk_k.squeeze(0)

            orig_pos = torch.arange(int(cws.start), int(cws.end), device=device)
            if orig_pos.numel() != chunk_len:
                orig_pos = torch.arange(0, chunk_len, device=device)
            new_pos = torch.arange(start, end, device=device)
            chunk_k = self.selector.rerope_key(chunk_k, orig_pos, new_pos)

            n_kv_heads = chunk_k.shape[0]
            if n_q_heads != n_kv_heads:
                repeat_factor = n_q_heads // n_kv_heads
                chunk_k = chunk_k.repeat_interleave(repeat_factor, dim=0)

            processed_keys.append(chunk_k)
            local_to_abs.extend(range(start, end))
            per_chunk_hit_tokens.append(chunk_len)

        if processed_keys:
            all_keys = torch.cat(processed_keys, dim=1)
            attn_logits = torch.matmul(
                query_final_q.float(),
                all_keys.float().transpose(-2, -1),
            ) / (head_dim ** 0.5)
            attn_weights = F.softmax(attn_logits, dim=-1)
            critical_scores = attn_weights.sum(dim=(0, 1)).float()
            score_tensor[local_to_abs] = critical_scores

        if stats is not None:
            hit_scores = score_tensor[score_tensor > 0]
            stats["lmcache_hybrid_query_score_tokens"] = int(score_tensor.numel())
            stats["lmcache_hybrid_query_score_nonzero"] = int((score_tensor > 0).sum().item())
            stats["lmcache_hybrid_per_chunk_hit_tokens"] = per_chunk_hit_tokens
            stats["lmcache_hybrid_query_score_mean_nonzero"] = (
                float(hit_scores.mean().item()) if hit_scores.numel() else 0.0
            )
            stats["lmcache_hybrid_query_score_max"] = (
                float(hit_scores.max().item()) if hit_scores.numel() else 0.0
            )

        return score_tensor

    def reuse_cache_with_query(
        self,
        chunks,
        query_final_q,
        llm,
        tokenizer,
        stats,
        stub_mode=False,
        loading_mode: str = "generator",
    ):
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
                "chunk_hit_ratio": (len(chunks) - len(missing_cids)) / len(chunks) if len(chunks) > 0 else 0,
            }

        recompute_ratios = self.recomp_tkns_for_chunks(chunks)
        if stats:
            stats["recompute_ratios_sent"] = recompute_ratios
            stats["mode"] = self.method_name

        with Timer("query_relevance_scores", stats=stats):
            query_scores = self._compute_query_relevance_scores(
                chunks=chunks,
                missing_cids=missing_cids,
                query_final_q=query_final_q,
                llm=llm,
                stats=stats,
            )

        doc_ids = torch.cat([c.tokens_() for c in chunks], dim=0).unsqueeze(0).to(llm.device, non_blocking=True)
        cachedb = []
        for chunk in chunks:
            if chunk.cid in missing_cids:
                cws = self._create_dummy_chunk(chunk, llm)
            else:
                cws = self.chunk_to_cchunk[chunk.cid]
            orig_pos = torch.arange(cws.start, cws.end)
            cachedb.append((chunk.cid, chunk.tokens_(), cws, cws, orig_pos))

        with Timer("blend", stats=stats):
            blended_cache = generator_build_blent_cache5(
                llm.model,
                cachedb,
                doc_ids,
                recompute_ratios,
                past_key_value=None,
                stats=stats,
                use_piaffe=self.use_piaffe,
                precomputed_recompute_scores=query_scores,
                precomputed_selection_mode=self.selection_mode,
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
                        "deficit": expected_growth - actual_growth,
                    }
                if cache_size_after < cache_size_before:
                    stats["CACHE_SHRINKAGE_DETECTED"] = {
                        "before": cache_size_before,
                        "after": cache_size_after,
                        "lost_chunks": cache_size_before - cache_size_after,
                    }

        return blended_cache


class LMCacheMix(LMCacheHybrid):
    """
    LMCache online with an epsilon-gated stale-and-relevant selector.

    Selection score is:
        minmax(layer-1 diffK) * (0.5 + minmax(final-layer query attention))

    This keeps the original diffK signal as the base ranking while still
    boosting tokens that the query attends to.
    """

    method_name = "lmcache_mix"
    selection_mode = "mix"


class LMCacheGuarded(LMCacheHybrid):
    """
    LMCache online with a guarded hybrid selector.

    It uses the product hybrid score unless that top-R set would spend a
    noticeable part of the sparse budget on the first prompt chunk, in which
    case it falls back to original diffK for that query.
    """

    method_name = "lmcache_guarded"
    selection_mode = "guarded"


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


class LMCacheAltManager(LMCacheOnlineManager):
    """Manager for LMCacheAlt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self.cache, "set_positional_encoders"):
            self.cache.set_positional_encoders(self.model)
        cache_log.info("LMCacheAlt initialized with FusionRAG-style selection")

    def _compute_query_selection_layer_q(self, query_tokens, context_length, stats=None):
        with Timer("query_selection_q_computation", stats=stats):
            query_ids = query_tokens.unsqueeze(0).to(self.device)
            query_len = query_ids.shape[1]
            position_ids = torch.arange(
                context_length,
                context_length + query_len,
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(0)
            attention_mask = torch.ones_like(query_ids, device=self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=query_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                if outputs.hidden_states is None or len(outputs.hidden_states) < 2:
                    raise RuntimeError("LMCacheAlt query prefill did not return hidden states.")

                layer_idx = self.cache.selection_layer_index(self.llm)
                hidden_before_layer = outputs.hidden_states[layer_idx]
                layer = self.model.layers[layer_idx]
                normed_hidden = layer.input_layernorm(hidden_before_layer)
                q_proj = layer.self_attn.q_proj(normed_hidden)

                config = self.llm.config
                n_heads = getattr(config, "num_attention_heads", None)
                head_dim = getattr(config, "head_dim", None)
                if head_dim is None:
                    head_dim = config.hidden_size // n_heads

                q_proj = q_proj.view(1, query_len, n_heads, head_dim).transpose(1, 2)
                cos, sin = self.model.rotary_emb(q_proj, position_ids)
                q_proj, _ = apply_rotary_pos_emb(q_proj, q_proj, cos, sin)
                if stats is not None:
                    stats["query_selection_layer_idx"] = int(layer_idx)
                return q_proj.squeeze(0)

    @torch.no_grad()
    def new_query(self, prompt: torch.Tensor, force_full=False, snapshot_path=None, reference_tensors=None):
        self.n_q += 1
        stats = {}
        self._last_stats = stats

        chunks = chunks_from_tokenss(prompt, self.sep)
        chunks, query = chunks[:-1], chunks[-1].tokens
        context_length = sum(len(c) for c in chunks)

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

        query_layer_q = self._compute_query_selection_layer_q(query, context_length, stats)

        with Timer("reuse_cache", stats):
            cache = self.cache.reuse_cache_with_query(
                chunks, query_layer_q, self.llm, self.tokenizer, stats,
                stub_mode=self.stub_mode, loading_mode=self.loading_mode,
            )

        with Timer("query", stats):
            if cache:
                try:
                    if snapshot_path:
                        try:
                            meta = [{"cid": c.cid, "len": len(c)} for c in chunks]
                            save_snapshot(snapshot_path, "lmcache_alt", cache, query, meta)
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

            cache, _ = full_prefill_cache(chunks, self.llm, output_attentions=False)
            return do_query_with_state(
                self.llm, self.tokenizer, cache, query,
                output_attentions=attn_capture, stats=stats
            ), False, stats


class LMCacheHybridManager(LMCacheAltManager):
    """Manager for LMCacheHybrid."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cache_log.info("LMCacheHybrid initialized with diffK * query-attention selection")


class LMCacheMixManager(LMCacheAltManager):
    """Manager for LMCacheMix."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cache_log.info("LMCacheMix initialized with epsilon-gated diffK/query-attention selection")


class LMCacheGuardedManager(LMCacheAltManager):
    """Manager for LMCacheGuarded."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cache_log.info("LMCacheGuarded initialized with guarded hybrid/diffK selection")
