"""
FusionRAG Implementation - 
- Uses RoPE helpers from cachebend.llm_docs like CacheBlend
- Does Re-Ropping 
- Implemented Global Token Selection 
"""

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import DynamicCache
from typing import Dict, List, Optional, Tuple, Union
import logging
import gc
import os
from functools import partial

from cachebend.utils import Timer
from cachebend.ncf.cutils import (
    Chunk, PosChunk, CachedChunk, to_str_prompt, build_tokenizer,
    do_query_with_state, chash, full_prefill_cache, save_snapshot, split_prompt_for_warm_chunks
)
from cachebend.llm_docs import (
    generator_build_blent_cache4,
    positional_encoder,
    reverse_positional_encoder,
    apply_rotary_pos_emb,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
cache_log = logging.getLogger("fusionrag")
cache_log.setLevel(logging.INFO)


def isdbg():
    return cache_log.isEnabledFor(logging.DEBUG)


def _is_zero_recompute_ratio(value) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False




class FusionRAGCachedChunk:
    def __init__(
        self,
        tokens: torch.Tensor,
        start: int,
        end: int,
        states: List[Tuple[torch.Tensor, torch.Tensor]],
        final_layer_key: Optional[torch.Tensor] = None,
    ):
        self.tokens = tokens
        self.start = start
        self.end = end
        self.states = states
        
        # Extract final layer key if not provided
        if final_layer_key is None and states:
            self.final_layer_key = states[-1][0].clone()
        else:
            self.final_layer_key = final_layer_key
            
        self._cid = None

    @property
    def cid(self) -> int:
        if self._cid is None:
            self._cid = chash(self.tokens)
        return self._cid
    
    def tokens_(self) -> torch.Tensor:
        return self.tokens

    def __len__(self):
        return self.end - self.start
    
    def get_layer(self, layer_idx: int, kv_idx: int, device: torch.device):
        return self.states[layer_idx][kv_idx].to(device, non_blocking=True)



#  QUERY-GUIDED SELECTOR 


class QueryGuidedSelector:
    """
    Implements Global FusionRAG Selection with Re-Ropping.
    """
    
    def __init__(self, recomp_ratio: float, aggregate_method: str = "sum"):
        self.recomp_ratio = recomp_ratio
        self.aggregate_method = aggregate_method
        self.positional_encoder = None
        self.reverse_positional_encoder = None
    
    def set_positional_encoders(self, positional_encoder, reverse_positional_encoder):
        self.positional_encoder = positional_encoder
        self.reverse_positional_encoder = reverse_positional_encoder
    
    def rerope_key(
        self,
        k: torch.Tensor,  # [n_kv_heads, seq_len, head_dim]
        original_positions: torch.Tensor,
        new_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Un-rotates K from original positions and Re-rotates to new positions.
        Similar to CacheBlendImpl.rescale32() but for the selection phase.
        """
        if self.reverse_positional_encoder is None:
            return k
        
        # Save original dtype for restoration (like rescale32 does)
        original_dtype = k.dtype
            
        # Shape Safeguard: Ensure 4D [1, H, S, D] for RoPE kernels
        was_3d = k.dim() == 3
        if was_3d:
            k = k.unsqueeze(0)
            
        device = k.device
        
        # Work in float32 for precision (like CacheBlend's rescale32)
        k = k.float()
        
        # Prepare Dummy Qit is required by encoder signature
        dumb_q = torch.zeros_like(k)
        
        #  Un-Rotate 
        with torch.no_grad():
            _, k_no_rope = self.reverse_positional_encoder(
                original_positions.to(device=device, dtype=torch.long),
                dumb_q, k
            )
            
            #  Re-Rotate 
            _, k_reroped = self.positional_encoder(
                new_positions.to(device=device, dtype=torch.long),
                dumb_q, k_no_rope
            )
        
        ###  Restore original shape and dtype
        if was_3d:
            k_reroped = k_reroped.squeeze(0)
        
        return k_reroped.to(original_dtype)
    
    def compute_critical_scores(
        self,
        query_final_q: torch.Tensor,
        chunk_final_keys: List[torch.Tensor],
        chunk_lengths: List[int],
        chunk_original_position_ranges: Optional[List[Tuple[int, int]]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Compute global attention scores with:
        - Re-roping K to correct positions
        - Global softmax across all chunks
        """
        # Calculate boundaries
        chunk_boundaries = []
        offset = 0
        for chunk_len in chunk_lengths:
            chunk_boundaries.append((offset, offset + chunk_len))
            offset += chunk_len
        
        n_q_heads = query_final_q.shape[0]
        head_dim = query_final_q.shape[-1]
        
        processed_keys = []
        
        # PREPARE CHUNKS (Re-RoPE & GQA) 
        for chunk_idx, (chunk_k, (start, end)) in enumerate(zip(chunk_final_keys, chunk_boundaries)):
            chunk_k = chunk_k.to(device, non_blocking=True)
            chunk_len = end - start
            
            # Normalize to [Heads, Seq, Dim]
            while chunk_k.dim() > 3:
                chunk_k = chunk_k.squeeze(0)
            
            # Warm per-query caches preserve the original prompt offsets used when
            # the contextualized chunk was exported. Those offsets must be used to
            # undo RoPE correctly before re-rotating into the current prompt.
            if chunk_original_position_ranges and chunk_idx < len(chunk_original_position_ranges):
                orig_start, orig_end = chunk_original_position_ranges[chunk_idx]
                if (orig_end - orig_start) == chunk_len:
                    orig_pos = torch.arange(orig_start, orig_end, device=device)
                else:
                    orig_pos = torch.arange(0, chunk_len, device=device)
            else:
                orig_pos = torch.arange(0, chunk_len, device=device)
            new_pos = torch.arange(start, end, device=device)
            
            chunk_k = self.rerope_key(chunk_k, orig_pos, new_pos)
            
            # GQA Expansion (repeat K heads to match Q heads)
            n_kv_heads = chunk_k.shape[0]
            if n_q_heads != n_kv_heads:
                repeat_factor = n_q_heads // n_kv_heads
                chunk_k = chunk_k.repeat_interleave(repeat_factor, dim=0)
            
            processed_keys.append(chunk_k)
            
        
        # Concatenate all keys: [n_q_heads, Total_Tokens, Head_Dim]
        all_keys = torch.cat(processed_keys, dim=1)
        
        # Compute Attention: [n_q_heads, Query_Len, Total_Tokens]
       
        attn_logits = torch.matmul(
            query_final_q.float(),
            all_keys.float().transpose(-2, -1)
        ) / (head_dim ** 0.5)
        
        # GLOBAL Softmax across all tokens
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        # Aggregate scores: sum over heads and query positions
        if self.aggregate_method == "max":
            critical_scores = attn_weights.amax(dim=(0, 1))
        elif self.aggregate_method == "mean":
            critical_scores = attn_weights.mean(dim=(0, 1))
        else:
            critical_scores = attn_weights.sum(dim=(0, 1))
            
        return critical_scores, chunk_boundaries
    
    def select_tokens(
        self,
        critical_scores: torch.Tensor,
        chunk_boundaries: List[Tuple[int, int]],
        stats: Optional[dict] = None,
    ) -> List[int]:
        """Select top-k tokens based on critical scores."""
        total_tokens = critical_scores.shape[0]
        if self.recomp_ratio == 0:
            num_to_select = 0
        else:
            num_to_select = max(1, int(total_tokens * self.recomp_ratio))
        num_to_select = min(num_to_select, total_tokens)
        
        
        
        _, top_indices = torch.topk(critical_scores, num_to_select)
        selected = sorted(top_indices.cpu().tolist())
        
        if stats is not None:
            stats["fusionrag_total_tokens"] = total_tokens
            stats["fusionrag_selected_count"] = len(selected)
            stats["fusionrag_selection_ratio"] = len(selected) / total_tokens if total_tokens > 0 else 0
            
            # Per-chunk distribution for analysis
            per_chunk = []
            for start, end in chunk_boundaries:
                count = sum(1 for idx in selected if start <= idx < end)
                per_chunk.append(count)
            stats["fusionrag_per_chunk_distribution"] = per_chunk
            
        return selected

    def __call__(
        self,
        query_final_q: torch.Tensor,
        chunk_final_keys: List[torch.Tensor],
        chunk_lengths: List[int],
        chunk_original_position_ranges: Optional[List[Tuple[int, int]]],
        device: torch.device,
        stats: Optional[dict] = None,
    ) -> List[int]:
        """Main entry point for Query-Guided Selection."""
        if _is_zero_recompute_ratio(self.recomp_ratio):
            total_tokens = sum(int(x) for x in chunk_lengths)
            if stats is not None:
                stats["fusionrag_total_tokens"] = total_tokens
                stats["fusionrag_selected_count"] = 0
                stats["fusionrag_selection_ratio"] = 0.0
                stats["fusionrag_per_chunk_distribution"] = [
                    0 for _ in chunk_lengths
                ]
                stats["fusionrag_selector_skipped"] = True
            return []

        with Timer("fusionrag_compute_scores", stats=stats):
            scores, boundaries = self.compute_critical_scores(
                query_final_q, chunk_final_keys, chunk_lengths, chunk_original_position_ranges, device
            )
        
        with Timer("fusionrag_select_tokens", stats=stats):
            selected = self.select_tokens(scores, boundaries, stats)
            
        return selected



#  FUSIONRAG CACHE


class FusionRAGCache:
    """
    Cache storage for FusionRAG.
    Stores FusionRAGCachedChunk objects which include final layer keys for selection.
    """
    
    def __init__(self, R: float):
        self.chunk_to_cchunk: Dict[int, FusionRAGCachedChunk] = dict()
        self.recomp_ratio = R
        self._seen_cids = set()
        self.selector = QueryGuidedSelector(recomp_ratio=R)

    def maybe_add(self, chunks: List[PosChunk], tokenizer, stats=None) -> List[PosChunk]:
        """Return chunks that need to be added to cache."""
        to_process = [c for c in chunks if c.cid not in self.chunk_to_cchunk]
        positions = [PosChunk(c.tokens, 0, len(c.tokens)) for c in to_process]
        if stats is not None:
            stats["_internal_added_set"] = {c.cid for c in to_process}
            stats["fusionrag_chunks_added"] = len(to_process)
        return positions

    def populate_cache_one_shot(self, poschunks: List[PosChunk], llm, tokenizer):
        """Populate cache with KV states and final layer keys."""
        if not poschunks: 
            return
            
        ids = [chunk.tokens_() for chunk in poschunks]
        input_ids = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device=llm.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            # We only need hidden-state KV caches here; routing through the full
            # LM head materializes large logits tensors and can OOM on long
            # prompts during warm evaluation.
            outputs = llm.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        
        past_kv = outputs.past_key_values
        num_layers = len(past_kv)
        
        for pid, chunk in enumerate(poschunks):
            curr_len = len(chunk.tokens)
            c_states = []
            for layer_idx in range(num_layers):
                k_slice = past_kv[layer_idx][0][pid, :, :curr_len, :]
                v_slice = past_kv[layer_idx][1][pid, :, :curr_len, :]
                k_cpu = torch.empty_like(k_slice, device="cpu", pin_memory=True)
                v_cpu = torch.empty_like(v_slice, device="cpu", pin_memory=True)
                k_cpu.copy_(k_slice, non_blocking=True)
                v_cpu.copy_(v_slice, non_blocking=True)
                c_states.append((k_cpu, v_cpu))

            cchunk = FusionRAGCachedChunk(
                tokens=chunk.tokens_(), start=0, end=curr_len,
                states=c_states, final_layer_key=c_states[-1][0]
            )
            if cchunk.cid not in self.chunk_to_cchunk:
                self.chunk_to_cchunk[cchunk.cid] = cchunk
        
        torch.npu.synchronize()

    def get_final_layer_keys_and_lengths(self, chunks: List[PosChunk]) -> Tuple[List[torch.Tensor], List[int], List[Tuple[int, int]]]:
        """Extract final layer keys, lengths, and original position ranges for selection."""
        keys = []
        lengths = []
        original_ranges = []
        for chunk in chunks:
            cws = self.chunk_to_cchunk[chunk.cid]
            keys.append(cws.final_layer_key)
            lengths.append(len(cws))
            original_ranges.append((int(cws.start), int(cws.end)))
        return keys, lengths, original_ranges

    def reuse_cache(self, chunks, llm, tokenizer, stats, recompute_indices, stub_mode=False, loading_mode="generator"):
        """Build blended cache using pre-computed recompute indices."""
        stats["num_used_cids"] = len(chunks)
        stats["used_cids"] = [c.cid for c in chunks]
        
        doc_ids = torch.cat([chunk.tokens_() for chunk in chunks], dim=0).unsqueeze(0).to(llm.device, non_blocking=True)
        
        cachedb = []
        added_set = stats.get("_internal_added_set", set())
        
        count_fresh = 0
        count_cached = 0
        breakdown = []
        
        for chunk in chunks:
            cws = self.chunk_to_cchunk[chunk.cid]
            
            is_new = chunk.cid in added_set
            if is_new:
                count_fresh += 1
                status = "NEW"
            else:
                count_cached += 1
                status = "CACHED"
            
            breakdown.append(f"CID:{chunk.cid}|{status}")
            
            cachedb.append((
                chunk.cid, chunk.tokens_(), cws, cws, torch.arange(cws.start, cws.end)
            ))
            
        stats["chunk_breakdown"] = breakdown
        stats["num_fresh_chunks"] = count_fresh
        stats["num_cached_chunks"] = count_cached
            
        if "_internal_added_set" in stats: 
            del stats["_internal_added_set"]
            
        if stub_mode: 
            return None
        
        # Pass recompute_indices as a list - this triggers _apply_tokens_all_queries
        # in CacheBlendImpl.blend() which uses the pre-computed indices
        return generator_build_blent_cache4(
            llm.model, cachedb, doc_ids, recompute_indices,
            past_key_value=None, stats=stats, use_piaffe=False
        )



#  the FUSIONRAG CACHE MANAGER


class FusionRAGCacheManager:
    """
    Manager implementing FusionRAG's Query-Guided Selection.
    
    Flow:
    1. Prefill query to get final layer Q
    2. Re-rope cached K to correct positions
    3. Compute global attention scores
    4. Select top-k tokens for recomputation
    5. Use generator_build_blent_cache4 with pre-computed indices
    """
    
    def __init__(
        self, device: str, cache: FusionRAGCache, llm=None, tkn=None,
        stub_mode: bool = False, loading_mode: str = "generator", no_reuse: bool = False,
    ):
        self.device = torch.device(device)
        self.stub_mode = stub_mode
        self.loading_mode = loading_mode
        self.no_reuse = no_reuse
        self.llm = llm
        self.tokenizer = tkn
        self.llm.eval()
        self.model = self.llm.model
        self.cache = cache
        self.sep = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.int64)
        self.ascii_sep = self.tokenizer.sep_token
        self.num_layers = self.llm.config.num_hidden_layers
        self.final_layer_idx = self.num_layers - 1
        self.n_q = 0

       
        self.cache.selector.set_positional_encoders(
            positional_encoder=partial(positional_encoder, self.model),
            reverse_positional_encoder=partial(reverse_positional_encoder, self.model),
        )

        if self.stub_mode: 
            cache_log.info("FusionRAG running in STUB MODE")
        cache_log.info(f"FusionRAG initialized (R={cache.recomp_ratio}, loading={loading_mode})")

    def tokenize(self, doc, use_special_tokens=False):
        return self.tokenizer(doc, return_tensors="pt", add_special_tokens=use_special_tokens)["input_ids"][0].to(torch.int64)
    
    def to_str_prompt(self, lis):
        return to_str_prompt(self.tokenizer, self.ascii_sep, lis)

    def split_prompt_chunks(self, prompt):
        return split_prompt_for_warm_chunks(prompt, self.sep, self.tokenizer)

    def _compute_query_final_layer_q(self, query_tokens, context_length, stats=None):
        """
        Compute the final layer's Q matrix for the query.
        Use the model's native forward pass to obtain the hidden state right
        before the final decoder layer, then project the final-layer Q with the
        same RoPE convention as the installed Transformers stack.
        """
        with Timer("query_final_q_computation", stats=stats):
            query_ids = query_tokens.unsqueeze(0).to(self.device)
            query_len = query_ids.shape[1]

            # Query positions are offset by the number of context tokens that
            # will already be present in the final prompt.
            position_ids = torch.arange(
                context_length, context_length + query_len, device=self.device, dtype=torch.long
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
                    raise RuntimeError("FusionRAG query prefill did not return hidden states.")

                # hidden_states[-2] is the input to the final decoder layer.
                hidden_before_final = outputs.hidden_states[-2]
                final_layer = self.model.layers[self.final_layer_idx]
                normed_hidden = final_layer.input_layernorm(hidden_before_final)
                q_proj = final_layer.self_attn.q_proj(normed_hidden)

                # Get head configuration from config.
                config = self.llm.config
                n_heads = getattr(config, "num_attention_heads", None)
                head_dim = getattr(config, "head_dim", None)
                if head_dim is None:
                    head_dim = config.hidden_size // n_heads

                # Reshape to [batch, n_heads, seq_len, head_dim]
                q_proj = q_proj.view(1, query_len, n_heads, head_dim).transpose(1, 2)

                # Apply the same rotary embedding convention used by the model.
                cos, sin = self.model.rotary_emb(q_proj, position_ids)
                q_proj, _ = apply_rotary_pos_emb(q_proj, q_proj, cos, sin)

                return q_proj.squeeze(0)  # [n_heads, query_len, head_dim]

    @torch.no_grad()
    def new_query(self, prompt, force_full=False, snapshot_path=None):
        """Process a new query with FusionRAG's  Selection."""
        self.n_q += 1
        stats = {}
        
        # Split prompt into chunks and query
        chunks = self.split_prompt_chunks(prompt)
        chunks, query = chunks[:-1], chunks[-1].tokens
        context_length = sum(len(c) for c in chunks)

        # Force full prefill path
        if force_full:
            if self.stub_mode: 
                return "STUB_FULL", False, stats
            cache, _ = full_prefill_cache(chunks, self.llm, output_attentions=False)
            return do_query_with_state(self.llm, self.tokenizer, cache, query, output_attentions=False, stats=stats), False, stats

        # Check for new chunks to add
        with Timer("maybe_add", stats):
            positions = self.cache.maybe_add(chunks, self.tokenizer, stats=stats)
        
        # Populate cache for new chunks
        if positions:
            if self.stub_mode: 
                return "STUB_RESPONSE", True, stats
            with Timer("populate_cache", stats):
                self.cache.populate_cache_one_shot(positions, self.llm, self.tokenizer)

        if _is_zero_recompute_ratio(self.cache.recomp_ratio):
            stats["fusionrag_selector_skipped"] = True
            stats["fusionrag_total_tokens"] = context_length
            stats["fusionrag_selected_count"] = 0
            stats["fusionrag_selection_ratio"] = 0.0
            stats["fusionrag_per_chunk_distribution"] = [0 for _ in chunks]
            stats["recompute_indices_count"] = 0
            stats["recompute_ratio_actual"] = 0.0
            recompute_arg = 0.0
        else:
            # === FUSIONRAG: Query-Guided Selection ===

            # Step 1: Compute query's final layer Q matrix
            query_final_q = self._compute_query_final_layer_q(
                query, context_length, stats
            )

            # Step 2: Get final layer keys and lengths for all chunks
            (
                chunk_final_keys,
                chunk_lengths,
                chunk_original_position_ranges,
            ) = self.cache.get_final_layer_keys_and_lengths(chunks)

            # Step 3: Compute critical scores and select tokens
            recompute_indices = self.cache.selector(
                query_final_q=query_final_q,
                chunk_final_keys=chunk_final_keys,
                chunk_lengths=chunk_lengths,
                chunk_original_position_ranges=chunk_original_position_ranges,
                device=self.device,
                stats=stats,
            )

            stats["recompute_indices_count"] = len(recompute_indices)
            stats["recompute_ratio_actual"] = (
                len(recompute_indices) / context_length
                if context_length > 0
                else 0
            )

            # When no token is selected, use gen4's float-0.0 fast path.
            recompute_arg = recompute_indices if recompute_indices else 0.0

        # Step 4: Build blended cache with pre-computed indices
        with Timer("reuse_cache", stats):
            cache = self.cache.reuse_cache(
                chunks, self.llm, self.tokenizer, stats, recompute_arg,
                stub_mode=self.stub_mode, loading_mode=self.loading_mode
            )

        # Step 5: Generate response
        with Timer("query_generation", stats):
            if cache:
                try:
                    if self.stub_mode: 
                        return "STUB_RESPONSE", True, stats
                    if snapshot_path:
                        meta = [{"cid": c.cid, "len": len(c)} for c in chunks]
                        save_snapshot(snapshot_path, "fusionrag", cache, query, meta)
                    
                    return do_query_with_state(self.llm, self.tokenizer, cache, query, output_attentions=False, stats=stats), True, stats
                finally:
                    torch.npu.empty_cache()
                    gc.collect()
            else:
                cache, _ = full_prefill_cache(chunks, self.llm, output_attentions=False)
                return do_query_with_state(self.llm, self.tokenizer, cache, query, output_attentions=False, stats=stats), False, stats
    
    def begin_fresh_query(self):
        """Reset cache state."""
        self.cache = FusionRAGCache(R=self.cache.recomp_ratio)



#  WARM CACHE, cioe caricare le cache contestualizzate prima 


class FusionRAGWarmCache(FusionRAGCache):
    """FusionRAG cache that pre-loads chunks from a .pt file."""
    
    def __init__(self, R: float, warm_cache_path: str):
        super().__init__(R)
        self.warm_cache_path = warm_cache_path
        self.expected_prefix_cids = None
        self.warm_unexpected_miss_total = 0
        self._warm_base_chunk_to_cchunk = {}
        self._load_warm_cache()
    
    def _load_warm_cache(self):
        if not self.warm_cache_path or not os.path.exists(self.warm_cache_path):
            cache_log.warning(f"Warm cache not found: {self.warm_cache_path}")
            return
        try:
            cache_log.info(f"Loading FusionRAG warm cache from {self.warm_cache_path}...")
            data = torch.load(self.warm_cache_path, map_location="cpu", weights_only=False)
            
            for cid, chunk in data.items():
                cid = str(cid)
                if isinstance(chunk, FusionRAGCachedChunk):
                    self.chunk_to_cchunk[cid] = chunk
                elif hasattr(chunk, 'states'):
                    # Convert from CachedChunk to FusionRAGCachedChunk
                    final_k = chunk.states[-1][0] if chunk.states else None
                    tokens = chunk.tokens if hasattr(chunk, 'tokens') else chunk.tokens_()
                    fusionrag_chunk = FusionRAGCachedChunk(
                        tokens=tokens, start=chunk.start, end=chunk.end,
                        states=chunk.states, final_layer_key=final_k
                    )
                    self.chunk_to_cchunk[cid] = fusionrag_chunk
            self._warm_base_chunk_to_cchunk = dict(self.chunk_to_cchunk)
                    
            cache_log.info(f"Loaded {len(self.chunk_to_cchunk)} warm chunks")
        except Exception as e:
            cache_log.error(f"Failed to load warm cache: {e}")
            raise

    def load_warm_cache_path(self, warm_cache_path: str):
        """Replace the loaded warm cache with a new per-query cache file."""
        self.warm_cache_path = warm_cache_path
        self.chunk_to_cchunk = {}
        self._warm_base_chunk_to_cchunk = {}
        self.expected_prefix_cids = None
        self._load_warm_cache()

    def reset_to_warm_base(self):
        self.chunk_to_cchunk = dict(self._warm_base_chunk_to_cchunk)
        self.expected_prefix_cids = None

    def prepare_next_query(self):
        # Preserve any newly learned chunks (for example the prompt chunk learned
        # on query 1) and only reset per-query miss bookkeeping.
        self.expected_prefix_cids = None

    def warm_miss_check(self, chunks, missing_cids, stats=None):
        missing_cids = [str(x) for x in missing_cids]
        missing_set = set(missing_cids)
        if self.expected_prefix_cids is None:
            expected = []
            for chunk in chunks:
                cid = str(chunk.cid)
                if cid in missing_set:
                    expected.append(cid)
                else:
                    break
            self.expected_prefix_cids = expected
        unexpected = []
        for cid in missing_cids:
            if cid not in set(self.expected_prefix_cids or []):
                unexpected.append(cid)
        if unexpected:
            self.warm_unexpected_miss_total += len(unexpected)
        if stats is not None:
            stats["warm_expected_prefix_cids"] = list(self.expected_prefix_cids or [])
            stats["warm_expected_prefix_count"] = len(self.expected_prefix_cids or [])
            stats["warm_missing_cids"] = [str(x) for x in missing_cids]
            stats["warm_unexpected_miss_cids"] = [str(x) for x in unexpected]
            stats["warm_unexpected_miss_count"] = len(unexpected)
            stats["warm_unexpected_miss_total"] = self.warm_unexpected_miss_total
        return len(unexpected) == 0


class FusionRAGWarmCacheManager(FusionRAGCacheManager):
    """
     replacement for CacheBlendWarmCacheManager. this initialize with the preocmputed contextualized caches
    
     Usage:
        caches["fusionrag"] = FusionRAGWarmCacheManager(
            device=dev, warm_cache_path=args.warm_cache, R=args.recomp,
            llm=llm, tkn=tkn, stub_mode=args.stub
        )
    """
    
    def __init__(
        self, device: str, warm_cache_path: str, R: float, 
        llm=None, tkn=None, stub_mode: bool = False, 
        loading_mode: str = "generator", no_reuse: bool = False
    ):
        cache = FusionRAGWarmCache(R=R, warm_cache_path=warm_cache_path)
        super().__init__(
            device=device, cache=cache, llm=llm, tkn=tkn, 
            stub_mode=stub_mode, loading_mode=loading_mode, no_reuse=no_reuse
        )
        cache_log.info(f"FusionRAGWarmCacheManager ready (warm_chunks={len(cache.chunk_to_cchunk)})")

    def begin_fresh_query(self):
        self.cache.prepare_next_query()

    def set_warm_cache_path(self, warm_cache_path: str):
        self.cache.load_warm_cache_path(warm_cache_path)

    @torch.no_grad()
    def new_query(self, prompt, force_full=False, snapshot_path=None):
        chunks = self.split_prompt_chunks(prompt)
        chunks = chunks[:-1] if len(chunks) > 1 else chunks
        missing_set = {str(c.cid) for c in chunks if c.cid not in self.cache.chunk_to_cchunk}
        missing = list(missing_set)
        hit_chunks = [c for c in chunks if str(c.cid) not in missing_set]
        miss_chunks = [c for c in chunks if str(c.cid) in missing_set]
        hit_tokens = int(sum(len(c) for c in hit_chunks))
        miss_tokens = int(sum(len(c) for c in miss_chunks))
        total_tokens = hit_tokens + miss_tokens
        total_chunks = len(chunks)
        output, used_cache, stats = super().new_query(prompt, force_full=force_full, snapshot_path=snapshot_path)
        if isinstance(stats, dict):
            self.cache.warm_miss_check(chunks, missing, stats=stats)
            stats["mode"] = "fusionrag_warm"
            stats["token_composition"] = {
                "hit_tokens": hit_tokens,
                "miss_tokens": miss_tokens,
                "hit_ratio": (float(hit_tokens) / total_tokens) if total_tokens else None,
                "total_tokens": total_tokens,
                "hit_chunks": len(hit_chunks),
                "miss_chunks": len(miss_chunks),
                "total_chunks": total_chunks,
            }
        return output, used_cache, stats
