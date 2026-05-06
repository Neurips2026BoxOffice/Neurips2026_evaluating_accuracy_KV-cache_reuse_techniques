#this is tentativo version 
#for saving the caches for querstions
#for quesions for which, for some reason,
#zcf actually answer better thna baseline


from time import perf_counter as now
import inspect
import torch
import torch_npu  
#for now import all 3, so that we can choose
from cachebend.llm_docs import build_blent_cache, seq_build_blent_cache, overlap_build_blent_cache, generator_build_blent_cache4, generator_build_blent_cache_unified 
from cachebend.cacheblend import CacheBlendImpl, rope32
from transformers import (
    AutoTokenizer,
    DynamicCache,
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedTokenizer,
)

#from cacheblend.snapshot import save_snapshot
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple, Optional
from .cutils import (
    lcs_length,
    Chunk,
    PosChunk,
    ccid_base_hash,
    mcid_base_hash,
    chunks_from_tokenss,
    hash_strings,
    generate_triplets,
    to_str_prompt,
    build_tokenizer,
    full_prefill_cache,
    do_query_with_state,
    longest_prefix,
    prefix_info_order_free,
    prefix_info_order_sensitive,
    save_snapshot,
)
from cachebend.utils import Timer
import logging
from itertools import groupby
import gc
import os

#model_path = "/data/weights/mistral"
model_path = "/data/weights/llama3.1-8BI"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
cache_log = logging.getLogger("zcf")
greedy_log = logging.getLogger("greedy")
cache_log.setLevel(logging.DEBUG)
greedy_log.setLevel(logging.DEBUG)


def isdbg():
    return cache_log.isEnabledFor(logging.DEBUG)


class ChunkWithState:
    def __init__(self, tokens: torch.tensor, cid, ccid, mcid, states, start, end):
        self.start = start
        self.end = end
        # states = 0=k 1=v // [0][layer][batch,num_h,seq_len,head_dim]
        self.states = states  
        assert all(
            x.device == torch.device("cpu") for tup in states for x in tup
        ), "states are not on cpu"
        self.cid = cid
        self.ccid = ccid
        self.mcid = mcid
        self.tokens = tokens
        self._pinned_states = None

    
    def get_layer(self, layer_idx: int, kv_idx: int, device: torch.device):
        tensor = self.states[layer_idx][kv_idx]
        if os.getenv("ZCF_REUSE_PINNED", "0") == "0":
            return tensor
        if tensor.is_pinned():
            return tensor
        if self._pinned_states is None:
            self._pinned_states = {}
        key = (layer_idx, kv_idx)
        if key in self._pinned_states:
            return self._pinned_states[key]
        pinned = torch.empty_like(tensor, device="cpu", pin_memory=True)
        pinned.copy_(tensor, non_blocking=False)
        self._pinned_states[key] = pinned
        return pinned

    def state(self, i, device):
        return [state[i].to(device, non_blocking=True) for state in self.states]

    def ks(self, device):
        return self.state(0, device)

    def vs(self, device):
        return self.state(1, device)

    def __len__(self):
        return self.end - self.start

    def __str__(self):
        lv = 0
        k = 0
        return f"CWS: {self.ccid=} {self.mcid=} {self.start=} {self.end=} {self.states[k][lv].shape=}"

    def __repr__(self):
        return str(self)


def _mcid(chunks: list[Chunk] | List[str]) -> int:
    if isinstance(chunks[0], str):
        return hash_strings([mcid_base_hash] + [c for c in chunks])
    else:
        return hash_strings([mcid_base_hash] + [c.cid for c in chunks])


def _ccid(chunks: list[Chunk] | List[str]) -> int:
    if isinstance(chunks[0], str):
        return hash_strings([ccid_base_hash] + [c for c in chunks])
    else:
        return hash_strings([ccid_base_hash] + [c.cid for c in chunks])


class MultiChunkInfo:
    def __init__(self, chunks: list[PosChunk]):
        self.cids = [c.cid for c in chunks]
        self.mcid = _mcid(self.cids)
        self.chunks: list[PosChunk] = chunks

    def __str__(self):
        return f"MCI {self.mcid=} {self.cids=}"

    def __repr__(self):
        return str(self)


class MultiChunk:
    def __init__(self, mcid, cwss: list[ChunkWithState]):
        self.mcid = mcid
        self.cids = [c.cid for c in cwss]
        self.chunks_with_state = cwss

    def __str__(self):
        return f"MC {self.mcid=} {self.cids=}"

    def __repr__(self):
        return str(self)

    def to_log(self):
        cids = "/".join([str(x) for x in self.cids])
        return f"{self.mcid};{len(self.cids)};{cids}"


class ZChunk:
    def __init__(
        self,
        chunk: PosChunk,
        prefix: list[PosChunk],
        tokens: torch.Tensor,
        states: list[Tuple[torch.Tensor, torch.Tensor]],
        mcid: int,
    ):
        self.cid = chunk.cid
        self.prefix = prefix
        self.states = states
        self.tokens = tokens
        self.start = chunk.start
        self.end = chunk.end
        self.mcid = mcid

    def __len__(self):
        return self.end - self.start

    def __str__(self):
        return f"ZC {self.mcid=}  {self.prefix=}"


class ZMChunk:
    def __init__(self, mcid: int, chunks: list[ZChunk]):
        self.mcid = mcid
        self.chunks = chunks
        self.cids = [c.cid for c in self.chunks]


class ZCFCache_V1:
    def __init__(self, M, R, mchunk_size = 3):
        self.chunk_to_mchunk: Dict[int, List[int]] = dict()
        self.mchunk_to_cchunk: Dict[int, MultiChunk] = dict()
        self.chunk_to_cchunk: Dict[int, List[ChunkWithState]] = dict()
        self.max_atom_copies = M
        self.recomp_ratio = R
        self.mchunk_size = mchunk_size
        
        #addition for a better stub mode
        self._seen_cids = set()
        
    def _copy_tensor_maybe_sliced(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        non_blocking: bool,
        slice_tokens: int,
    ) -> None:
        if slice_tokens and src.ndim in (3, 4) and src.shape[-2] > slice_tokens:
            for s in range(0, src.shape[-2], slice_tokens):
                e = min(s + slice_tokens, src.shape[-2])
                if src.ndim == 4:
                    dst[:, :, s:e, :].copy_(src[:, :, s:e, :], non_blocking=non_blocking)
                elif src.ndim == 3:
                    dst[:, s:e, :].copy_(src[:, s:e, :], non_blocking=non_blocking)
        else:
            dst.copy_(src, non_blocking=non_blocking)

    def _copy_kv_to_cpu_with_fallback(
        self,
        k_src: torch.Tensor,
        v_src: torch.Tensor,
        use_pinned: bool,
        slice_tokens: int,
        stats: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Try fast path first, then progressively safer paths.
        copy_plans = [
            (use_pinned, use_pinned, slice_tokens),  # preferred path
            (False, False, slice_tokens),            # disable pinned + async
            (False, False, 0),                       # full blocking copy
        ]
        last_err = None
        for attempt, (pin_memory, non_blocking, stoks) in enumerate(copy_plans):
            try:
                k_cpu = torch.empty_like(k_src, device="cpu", pin_memory=pin_memory)
                v_cpu = torch.empty_like(v_src, device="cpu", pin_memory=pin_memory)
                self._copy_tensor_maybe_sliced(k_cpu, k_src, non_blocking=non_blocking, slice_tokens=stoks)
                self._copy_tensor_maybe_sliced(v_cpu, v_src, non_blocking=non_blocking, slice_tokens=stoks)
                if attempt > 0:
                    if stats is not None:
                        stats["zcf_copy_fallback_count"] = int(stats.get("zcf_copy_fallback_count", 0)) + 1
                        stats["zcf_copy_fallback_last_attempt"] = attempt
                    cache_log.warning(
                        f"KV copy recovered with fallback attempt={attempt} pin_memory={pin_memory} "
                        f"non_blocking={non_blocking} slice_tokens={stoks}"
                    )
                return k_cpu, v_cpu
            except RuntimeError as err:
                last_err = err
                if stats is not None:
                    stats["zcf_copy_failures"] = int(stats.get("zcf_copy_failures", 0)) + 1
                cache_log.warning(
                    f"KV copy failed on attempt={attempt} pin_memory={pin_memory} "
                    f"non_blocking={non_blocking} slice_tokens={stoks}: {err}"
                )
                try:
                    torch.npu.synchronize()
                except Exception:
                    pass
                if os.getenv("ZCF_PREFILL_EMPTY_CACHE_ON_COPY_FAIL", "1") == "1":
                    try:
                        torch.npu.empty_cache()
                    except Exception:
                        pass
                gc.collect()

        raise last_err


    
    def greediest_cover(self, chunks: List[PosChunk], llm, tokenizer, stats, stub_mode = False, loading_mode="generator"):
        Q: List[int] = [c.cid for c in chunks]
        S: List[Tuple[int, List[int]]] = []
        S_s = set()
        for c in Q:
            if c not in self.chunk_to_mchunk: continue
            mcids = self.chunk_to_mchunk[c]
            for m in mcids:
                if m in S_s: continue
                cids = self.mchunk_to_cchunk[m].cids
                S.append((m, cids))
                S_s.add(m)
        
        chosen = LongestUsefulPrefixCF().fabricate(Q, S)
        cachedb = []
        doc_ids = []
        
        for mcid, covered, _ in chosen:
            mchunk = self.mchunk_to_cchunk[mcid]
            used = [x for x in mchunk.chunks_with_state if x.cid in covered]
            for cws in used:
                # Task B: Lazy Loading (Object pass)
                cachedb.append(
                    (
                        cws.cid,
                        cws.tokens,
                        cws, 
                        cws, 
                        torch.arange(cws.start, cws.end),
                    )
                )
                doc_ids.append(cws.tokens)

        if not doc_ids:
            return None, [], []

        doc_ids = (
            torch.cat(doc_ids, dim=0).unsqueeze(0).to(llm.device, non_blocking=True)
        )
        used_mcids = [x[0] for x in chosen]
        used_chunks = [x for (_, cov, _) in chosen for x in cov]
        stats["zcf_reuse"] = [
            (mcid, len(cov), wasted) for (mcid, cov, wasted) in chosen
        ]
        if stats is not None:
            stats["zcf_total_chunks"] = len(chunks)
            stats["zcf_used_chunks"] = len(used_chunks)
            stats["zcf_used_mcids"] = len(used_mcids)
            stats["zcf_doc_tokens"] = int(doc_ids.shape[1])
            try:
                stats["zcf_chunk_lens"] = {
                    "min": int(min(len(c) for c in chunks)),
                    "max": int(max(len(c) for c in chunks)),
                    "sum": int(sum(len(c) for c in chunks)),
                }
            except Exception:
                pass

        fake = False
        #Skip blending entirely in stub mode
        if stub_mode:
            return (None, used_mcids, used_chunks)

        #Only run blending in real mode
        if not fake:
            assert loading_mode in ("", "sequential", "overlap", "generator"), "invalid cache loading mode passed"
            target_func = build_blent_cache
            if loading_mode == "sequential":
                target_func = seq_build_blent_cache
            elif loading_mode == "overlap":
                target_func = overlap_build_blent_cache
            elif loading_mode == "generator":
                target_func = generator_build_blent_cache_unified

            if isdbg():
                cache_log.debug(f"Using blending function: {target_func.__name__}")

            ret = target_func(
                llm.model,
                cachedb,
                doc_ids,
                self.recomp_ratio,
                past_key_value=None,
                stats=stats,
            )
        else:
            ret = None

        return (ret, used_mcids, used_chunks)

    # this is dead Code for now
    def greedy_cover(self, chunks: List[PosChunk], llm, tokenizer, stats):
        # #
        pass

    def reuse_cache(self, chunks, llm, tokenizer, stats, stub_mode = False,  loading_mode="generator"):
        torch.npu.synchronize() 
        with Timer("cover( < reuse_cache)", stats):
            return self.greediest_cover(chunks, llm, tokenizer, stats, stub_mode, loading_mode)

    def maybe_add(self, chunks: list[PosChunk], tokenizer) -> List[MultiChunkInfo]:
        
        
        def rm(c: PosChunk):
            try:
                ret = len(self.chunk_to_cchunk[c.cid]) == self.max_atom_copies
            except KeyError:
                ret = False
            return ret
        mc_group = [list(g) for k, g in groupby(chunks, rm) if not k]
        positions: List[MultiChunkInfo] = []
        for cgroup in mc_group:
            mcs: list[list[PosChunk]] = [cgroup[i : i + self.mchunk_size] for i in range(0, len(cgroup), self.mchunk_size)]
            keeps = []
            for mc in mcs:
                anchor = mc[0].cid
                def is_prefix(A, B): return len(A) <= len(B) and A == B[: len(A)]
                def mchunk_seq(mcid): return self.mchunk_to_cchunk[mcid].cids
                curr_seq = [c.cid for c in mc]
                if anchor in self.chunk_to_mchunk and any(is_prefix(curr_seq, mchunk_seq(c)) for c in self.chunk_to_mchunk[anchor]):
                    continue
                keeps = mcs
            for mc in keeps:
                start = 0
                for c in mc:
                    clen = len(c)
                    c.start = start
                    c.end = start + clen
                    start = c.end
                positions.append(MultiChunkInfo(mc))
        
        
        return positions if (positions and len(positions)) else None

    # Fixed: Merged fill_states and fill_cache into one function to kill the 
    # double-copy overhead. Also fixed the random crashes by adding sync.
    def populate_cache_one_shot(self, mcinfos: list[MultiChunkInfo], llm, tokenizer, stats=None):
        if not mcinfos:
            return

        safe_copy = os.getenv("ZCF_PREFILL_SAFE_COPY", "1") != "0"
        # Optional: control CPU pinned allocation (safer on RAM pressure)
        use_pinned = os.getenv("ZCF_PREFILL_PINNED", "1") != "0"
        # Optional: slice copy along sequence length to reduce peak transfer size
        env_slice = os.getenv("ZCF_PREFILL_SLICE_TOKENS", "").strip()
        try:
            slice_tokens = int(env_slice) if env_slice else 0
        except ValueError:
            slice_tokens = 0
        if safe_copy:
            # Safer path for NPU->CPU transfer stability.
            use_pinned = False
            slice_tokens = 0

        # Optional micro-batching to reduce peak NPU memory.
        # Set ZCF_PREFILL_BATCH to a small integer (e.g., 1 or 2) to force smaller batches.
        env_bs = os.getenv("ZCF_PREFILL_BATCH", "").strip()
        try:
            batch_size = int(env_bs) if env_bs else 0
        except ValueError:
            batch_size = 0
        if batch_size <= 0:
            batch_size = len(mcinfos)

        for batch_start in range(0, len(mcinfos), batch_size):
            batch = mcinfos[batch_start:batch_start + batch_size]

            # Prepare Input
            ids = [torch.cat([chunk.tokens for chunk in inner.chunks]) for inner in batch]
            input_ids = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device=llm.device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            # Forward pass to get the heavy tensors
            # Keep 'outputs' alive! If this variable dies, GC kills the tensors on NPU
            # while we are still trying to DMA them out.
            with torch.no_grad():
                outputs = llm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )

            past_key_values = outputs.past_key_values
            num_layers = len(past_key_values)

            # Direct NPU -> CPU copy
            for mic_id, mci in enumerate(batch):
                cchunk_list = []
                seq = []

                for curr, cid in enumerate(mci.cids):
                    seq.append(cid)
                    ccid = _ccid(seq)
                    start = mci.chunks[curr].start
                    end = mci.chunks[curr].end

                    c_states = []
                    for i in range(num_layers):
                        # Grab slice from NPU tensor
                        k_npu_slice = past_key_values[i][0][mic_id, :, start:end, :].contiguous()
                        v_npu_slice = past_key_values[i][1][mic_id, :, start:end, :].contiguous()

                        # Robust copy path with fallback to safer (blocking/pageable) modes.
                        k_final, v_final = self._copy_kv_to_cpu_with_fallback(
                            k_npu_slice,
                            v_npu_slice,
                            use_pinned=use_pinned,
                            slice_tokens=slice_tokens,
                            stats=stats,
                        )

                        c_states.append((k_final, v_final))

                    cws = ChunkWithState(
                        mci.chunks[curr].tokens,
                        cid=cid,
                        ccid=ccid,
                        mcid=mci.mcid,
                        states=c_states,
                        start=start,
                        end=end,
                    )

                    cchunk_list.append(cws)
                    try:
                        self.chunk_to_cchunk[cid].append(cws)
                    except KeyError:
                        self.chunk_to_cchunk[cid] = [cws]

                # Register MultiChunk
                if mci.mcid not in self.mchunk_to_cchunk:
                    self.mchunk_to_cchunk[mci.mcid] = MultiChunk(mci.mcid, cchunk_list)

                for cid in mci.cids:
                    if cid not in self.chunk_to_mchunk:
                        self.chunk_to_mchunk[cid] = []
                    self.chunk_to_mchunk[cid].append(mci.mcid)

            # Must wait for NPU to finish copying before 'outputs' goes out of scope.
            torch.npu.synchronize()
            del outputs, past_key_values, input_ids, attention_mask, ids
            if os.getenv("ZCF_PREFILL_EMPTY_CACHE", "0") == "1":
                torch.npu.empty_cache()

        return


    def deduplicate_cache(self, mcinfos: list[MultiChunkInfo]):
        def is_prefix(A, B): return len(A) <= len(B) and A == B[: len(A)]
        for mci in mcinfos:
            anchor = mci.cids[0]
            if not anchor in self.chunk_to_mchunk: continue
            mchunks: List[int] = self.chunk_to_mchunk[anchor]
            def mchunk_seq(mcid): return self.mchunk_to_cchunk[mcid].cids
            for mchunk in list(mchunks):
                mseq = mchunk_seq(mchunk)
                if is_prefix(mseq, mci.cids):
                    for cid in mseq:
                        self.chunk_to_mchunk[cid] = [x for x in self.chunk_to_mchunk[cid] if x != mchunk]
                        self.chunk_to_cchunk[cid] = [x for x in self.chunk_to_cchunk[cid] if x.mcid != mchunk]
                    del self.mchunk_to_cchunk[mchunk]
                    
                    
    def compute_temp_chunk_states(self, mcinfos: list[MultiChunkInfo], llm, tokenizer, stats=None):
        """Compute ChunkWithState objects without saving them to persistent cache."""
        if not mcinfos:
            return {}
        safe_copy = os.getenv("ZCF_PREFILL_SAFE_COPY", "1") != "0"
        
        ids = [torch.cat([chunk.tokens for chunk in inner.chunks]) for inner in mcinfos]
        input_ids = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device=llm.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = llm(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, output_hidden_states=False, output_attentions=False)
        
        past_key_values = outputs.past_key_values
        num_layers = len(past_key_values)
        cid_to_cws = {}

        for mic_id, mci in enumerate(mcinfos):
            seq = []
            for curr, cid in enumerate(mci.cids):
                seq.append(cid)
                ccid = _ccid(seq)
                start = mci.chunks[curr].start
                end = mci.chunks[curr].end
                
                c_states = []
                for i in range(num_layers):
                    k_npu_slice = past_key_values[i][0][mic_id, :, start:end, :].contiguous()
                    v_npu_slice = past_key_values[i][1][mic_id, :, start:end, :].contiguous()
                    k_final, v_final = self._copy_kv_to_cpu_with_fallback(
                        k_npu_slice,
                        v_npu_slice,
                        use_pinned=False if safe_copy else True,
                        slice_tokens=0,
                        stats=stats,
                    )
                    c_states.append((k_final, v_final))

                cws = ChunkWithState(
                    mci.chunks[curr].tokens,
                    cid=cid,
                    ccid=ccid,
                    mcid=mci.mcid,
                    states=c_states,
                    start=start,
                    end=end,
                )
                cid_to_cws[cid] = cws

        torch.npu.synchronize()
        return cid_to_cws


class ZCF:
    llm: AutoModelForCausalLM

    def to_str_prompt(self, lis) -> str:
        return to_str_prompt(self.tokenizer, self.ascii_sep, lis)

    # Added blending_mode with default "generator"
    def __init__(self, device: str, cache: ZCFCache_V1, llm=None, tkn=None, stub_mode=False, loading_mode="generator", no_reuse = False):
        self.device = torch.device(device)
        self.stub_mode = stub_mode 
        self.loading_mode = loading_mode 
        self.no_reuse = no_reuse
        
        
        
        if self.stub_mode:
            cache_log.info("ZCF running in STUB MODE (Inference Disabled)")
        cache_log.info(f"ZCF initialized with cache loading mode: {self.loading_mode}")

        if not llm:
            assert not tkn
            config = AutoConfig.from_pretrained(model_path)
            config.sliding_window = None
            config._attn_implementation = "eager"
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path, config=config
            )
            self.llm.to(self.device)
            self.llm.to(torch.bfloat16)
            self.tokenizer = build_tokenizer(model_path)
        else:
            self.llm = llm
            self.tokenizer = tkn
        self.llm.eval()
        self.model = self.llm.model
        self.cache = cache
        self.stub_mode = stub_mode
        self.loading_mode = loading_mode

        self.sep = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.int64)
        self.ascii_sep = self.tokenizer.sep_token
        self._last_stats = None

    @torch.no_grad()
    def new_query(self, prompt: torch.Tensor, force_full=False, snapshot_path = None):
        stats = {}
        self._last_stats = stats
        pre_existing_cids = set(self.cache.chunk_to_cchunk.keys())
        if isdbg(): cache_log.debug(f"New query {len(prompt)=} {self.tokenizer.decode(prompt[:10])=}")
        chunks = chunks_from_tokenss(prompt, self.sep)
        chunks, query = chunks[:-1], chunks[-1].tokens

        if force_full:
            if self.stub_mode: return "STUB_FULL", False, stats
            cache, _ = full_prefill_cache(chunks, self.llm, output_attentions=False)
            return (do_query_with_state(self.llm, self.tokenizer, cache, query, False, stats), False, stats)
        
        with Timer("maybe_add", stats=stats):
            positions: List[MultiChunkInfo] = self.cache.maybe_add(chunks, self.tokenizer)
            added_mcids = []
            
        if positions:
            added_mcids = [p.mcid for p in positions]
            # STUB MODE BRANCH !!!!!!!!!
            if self.stub_mode:
                query_cids = [c.cid for c in chunks]
                fresh_cids = [cid for cid in query_cids if cid not in pre_existing_cids]
                
                # Mock update for fresh cids to simulate cache growth
                for cid in fresh_cids:
                    if cid not in self.cache.chunk_to_cchunk:
                        self.cache.chunk_to_cchunk[cid] = [] 
                
                # Mock stats 
                # (fixed this logic to match the real branch below)
                cached_cids = [cid for cid in query_cids if cid in pre_existing_cids]
                total_chunks = len(chunks)
                
                stats.update({
                    "num_fresh": len(fresh_cids),
                    "num_cached": len(cached_cids),
                    "hit_rate": len(cached_cids) / total_chunks if total_chunks > 0 else 0,
                    "fresh_chunks_cids": fresh_cids,  
                    "chunk_breakdown": [
                        f"CID:{cid}|{'FRESH' if cid in fresh_cids else 'CACHED'}|MatchPLen:0"
                        for cid in query_cids
                    ],
                    "zcf_num_fresh_chunks": len(fresh_cids),
                    "zcf_num_cached_chunks": len(cached_cids),
                    "zcf_chunk_hit_rate": len(cached_cids) / total_chunks if total_chunks > 0 else 0,
                })
                
                cache_log.info("Stub mode: returning mock stats (no cache ops).")
                return "STUB_RESPONSE", True, stats
            else:
                #NON STUB-MODE
                #addition to have a mode with no reuse, but with the chunking
                if self.no_reuse:
                    temp_cws_map = self.cache.compute_temp_chunk_states(positions, self.llm, self.tokenizer, stats)
                    original_entries = {}
                    for cid, cws in temp_cws_map.items():
                        if cid in self.cache.chunk_to_cchunk:
                            original_entries[cid] = self.cache.chunk_to_cchunk[cid]
                        self.cache.chunk_to_cchunk[cid] = [cws]  # reuse_cache expects list
                        
                else:
                
                    
                    # Dedupe first (metadata only), then Fill (Single-Copy)
                    with Timer("deduplicate", stats=stats):
                        self.cache.deduplicate_cache(positions)
                        
                    with Timer("populate_cache", stats=stats):
                        # Replaced the old fill_states + fill_cache one-two punch
                        
                        self.cache.populate_cache_one_shot(positions, self.llm, self.tokenizer, stats)
        else:
            temp_cws_map = {}
            original_entries = {}

            
        with Timer("reuse_cache", stats=stats):
            
            #sono molto stanco
            #voglio andare a casa
            #sono affamato
            
            #meow
            
            #cache, used_mcids, used_cids = self.cache.reuse_cache(chunks, self.llm, self.tokenizer, stats, stub_mode = self.stub_mode, loading_mode=self.loading_mode)
            #addtion
            
            total_chunks = len(chunks)
            all_query_cids = [c.cid for c in chunks]

            # Fresh = not in cache BEFORE this query
            #pre_existing_cids = set(self.cache.chunk_to_cchunk.keys()) - set(temp_cws_map.keys()) if self.no_reuse else set(self.cache.chunk_to_cchunk.keys())
            fresh_cids = [cid for cid in all_query_cids if cid not in pre_existing_cids]
            cached_cids = [cid for cid in all_query_cids if cid in pre_existing_cids]
            
            
            chunk_mux_counts = {}
            for cid in all_query_cids:
                mcid_list = self.cache.chunk_to_mchunk.get(cid, [])
                chunk_mux_counts[cid] = len(mcid_list)

            # Store in stats
            stats["zcf_chunk_mux_counts"] = chunk_mux_counts  # {cid: int}
            stats["zcf_mux_per_chunk"] = [chunk_mux_counts[cid] for cid in all_query_cids]  # aligned list

            cache, used_mcids, used_cids = self.cache.reuse_cache(
                chunks, self.llm, self.tokenizer, stats,
                stub_mode=self.stub_mode, loading_mode=self.loading_mode
            )
            
            if self.no_reuse and positions:
                for cid in temp_cws_map:
                    if cid in original_entries:
                        self.cache.chunk_to_cchunk[cid] = original_entries[cid]
                    else:
                        self.cache.chunk_to_cchunk.pop(cid, None)


            # stats
            fresh_set = set(fresh_cids)
            cached_chunks = [c for c in chunks if c.cid in pre_existing_cids]
            fresh_chunks = [c for c in chunks if c.cid in fresh_set]
            hit_tokens = int(sum(len(c) for c in cached_chunks))
            miss_tokens = int(sum(len(c) for c in fresh_chunks))
            total_tokens = hit_tokens + miss_tokens
            stats.update({
                "mode": "zcf",
                "num_fresh": len(fresh_cids),
                "num_cached": len(cached_cids),
                "hit_rate": len(cached_cids) / total_chunks if total_chunks > 0 else 0,
                "fresh_chunks_cids": fresh_cids,
                "chunk_breakdown": [
                    f"CID:{cid}|{'FRESH' if cid in fresh_cids else 'CACHED'}|MatchPLen:0"
                    for cid in all_query_cids
                ],
                "zcf_num_fresh_chunks": len(fresh_cids),
                "zcf_num_cached_chunks": len(cached_cids),
                "zcf_chunk_hit_rate": len(cached_cids) / total_chunks if total_chunks > 0 else 0,
                "zcf_used_cids": used_cids,
                "zcf_fresh_chunks": len(fresh_cids),
                "zcf_cached_chunks": len(cached_cids),
                "token_composition": {
                    "hit_tokens": hit_tokens,
                    "miss_tokens": miss_tokens,
                    "hit_ratio": (float(hit_tokens) / total_tokens) if total_tokens else None,
                    "total_tokens": total_tokens,
                    "hit_chunks": len(cached_chunks),
                    "miss_chunks": len(fresh_chunks),
                    "total_chunks": total_chunks,
                },
            })

            # keep multi-chunk metadata for debugging
            stats["num_added_mcids"] = len(added_mcids)
            stats["num_used_mcids"] = len(used_mcids)
            added_set = set(added_mcids)
            used_set = set(used_mcids)
            stats["num_used_mcids_new"] = len(used_set & added_set)
            stats["num_used_mcids_cached"] = len(used_set - added_set)
            if not self.no_reuse:
                
                stats["added_mcids"] = [self.cache.mchunk_to_cchunk[c].to_log() for c in added_mcids] if added_mcids else []
            else:
                stats["added_mcids"] = []

        with Timer("query", stats=stats):
            try:
                if self.stub_mode:
                    cache_log.info("Stub mode active: returning metrics only.")
                    return "STUB_RESPONSE", True, stats
                if cache:
                    ##addition for the cache snapshot
                    if snapshot_path:
                        try:
                            meta = [{"cid": c.cid, "len": len(c)} for c in chunks]
                            save_snapshot(snapshot_path, "zcf", cache, query, meta)
                            
                        except Exception:
                            print("Warning, snapshot save failed for some reason")
                            raise
                        
                    #end addition for cache snapshot
                    
                    
                    return (do_query_with_state(self.llm, self.tokenizer, cache, query, False, stats), True, stats)
                else:
                    # Fallback, but this path should't be taken now 
                    cache_fb, _ = full_prefill_cache(chunks, self.llm, output_attentions=False)
                    return (do_query_with_state(self.llm, self.tokenizer, cache_fb, query, False, stats), False, stats)
                
            finally:
                del cache
                torch.npu.empty_cache()
                gc.collect()

    def tokenize(self, doc: str, use_special_tokens: bool = False) -> torch.Tensor:
        return self.tokenizer(
            doc, return_tensors="pt", add_special_tokens=use_special_tokens
        )["input_ids"][0].to(torch.int64)
        
    def begin_fresh_query(self):
        """Replace internal cache with a new empty one, preserving M and R."""
        self.cache = ZCFCache_V1(
            M=self.cache.max_atom_copies,
            R=self.cache.recomp_ratio,
            mchunk_size = self.cache.mchunk_size
        )

class CacheFabricator:
    def __init__(self): pass

class LongestUsefulPrefixCF(CacheFabricator):
    def __init__(self, preserve_order=True):
        super().__init__()
        self.preserve_order = preserve_order
    def longest_prefix_order_sensitive(self, S, Q, start_idx):
        Qi = Q[start_idx]
        if Qi not in S: return set(), float("inf"), 0
        covered = [Qi]
        s_index = {x: i for i, x in enumerate(S)}
        last_s_pos = s_index[Qi]
        for x in Q[start_idx + 1 :]:
            if x in s_index and s_index[x] > last_s_pos:
                covered.append(x)
                last_s_pos = s_index[x]
            else: break
        prefix_len = 1 + last_s_pos
        waste = prefix_len - len(covered)
        return covered, waste, prefix_len
    def greedy_query_cover(self, Q, Ss, keep_order=True):
        Q_set = set(Q)
        covered = set()
        plan = []
        i = 0
        while i < len(Q):
            if Q[i] in covered:
                i += 1
                continue
            best = None
            best_score = (float("inf"), -1)
            for mcid, S in Ss:
                cov, waste, _ = self.longest_prefix_order_sensitive(S, Q, i)
                if not cov: continue
                score = (-len(cov), waste)
                if score < best_score:
                    best = (mcid, cov, waste)
                    best_score = score
            if not best: break
            m, cov, waste = best
            plan.append((m, cov, waste))
            covered |= set(cov)
            while i < len(Q) and Q[i] in covered: i += 1
        return plan
    def fabricate(self, Q: List[int], S: List[Tuple[int, List[int]]]):
        return self.greedy_query_cover(Q, S, self.preserve_order)

class Baseline:
    #cleaned it up a bit
    def __init__(self, device: str, llm=None, tkn=None):
        self.device = torch.device(device)
        if not llm:
            config = AutoConfig.from_pretrained(model_path)
            config.sliding_window = None
            config._attn_implementation = "eager"
            self.llm = AutoModelForCausalLM.from_pretrained(model_path, config=config, output_attentions=True)
            self.llm.to(self.device)
            self.llm.to(torch.bfloat16)
            self.tokenizer = build_tokenizer(model_path)
        else:
            self.llm = llm
            self.tokenizer = tkn
        self.llm.eval()
        self.model = self.llm.model
        self.ascii_sep = self.tokenizer.sep_token
        self.sep = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.int64)
    def to_str_prompt(self, lis) -> str: return to_str_prompt(self.tokenizer, self.ascii_sep, lis)
    def tokenize(self, doc: str, use_special_tokens: bool = False) -> torch.Tensor:
        return self.tokenizer(doc, return_tensors="pt", add_special_tokens=use_special_tokens)["input_ids"][0].to(torch.int64)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        import re

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
        return text

    @torch.no_grad()
    def _direct_full_prefill_query(self, prompt: torch.Tensor, stats: dict, max_length: int = 50) -> str:
        """Baseline parity path: generate from the exact tokenized prompt, including DSEP tokens."""
        input_ids = prompt.to(self.llm.device).unsqueeze(0)
        forward_kwargs = {}
        try:
            if "logits_to_keep" in inspect.signature(self.llm.forward).parameters:
                forward_kwargs["logits_to_keep"] = 1
        except Exception:
            pass

        out = self.llm(input_ids=input_ids, use_cache=True, return_dict=True, **forward_kwargs)
        past = out.past_key_values
        generated = []
        logits = out.logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1)

        for _ in range(max_length):
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
            generated.append(next_token_id.item())
            out = self.llm(
                input_ids=next_token_id.unsqueeze(0),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits = out.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1)
            past = out.past_key_values

        stats["num_fresh_chunks"] = 1
        stats["num_cached_chunks"] = 0
        stats["chunks"] = []
        return self._strip_thinking(self.tokenizer.decode(generated, skip_special_tokens=True)).strip()

    @torch.no_grad()
    def _direct_query_no_context(self, query: torch.Tensor, stats: dict, max_length: int = 50):
        # Threads past_key_values across decode steps — same pattern as
        # `do_query_with_state` in cutils.py. Without this, every step
        # after the first re-encodes a single token with no history and
        # the model degenerates to "predict the next token from a token
        # in isolation" (typically high-frequency garbage).
        # Stop conditions: EOS OR a "\n\n" sequence appears in the
        # accumulating decoded text — the latter mirrors vLLM's
        # `stop=["\n\n"]` (used by the v8 gate's T1a/T1b), so this
        # path's predictions are directly comparable to the gate's.
        output = []
        accum  = ""
        input_ids = query.to(self.llm.device).unsqueeze(0)
        past = None
        for _ in range(max_length):
            outputs = self.llm(input_ids=input_ids, past_key_values=past,
                               use_cache=True, output_attentions=False)
            logits = outputs.logits
            past = outputs.past_key_values
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            decoded_token = self.tokenizer.decode(next_token_id, skip_special_tokens=True)
            output.append(decoded_token)
            accum += decoded_token
            input_ids = next_token_id.unsqueeze(0)
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
            if "\n\n" in accum:
                break
        stats["num_fresh_chunks"] = 0
        stats["num_cached_chunks"] = 0
        return "".join(output)

    @torch.no_grad()
    def new_query(self, prompt: torch.Tensor, force_full=False, snapshot_path = None):
        stats = {}
        if force_full:
            try:
                return (self._direct_full_prefill_query(prompt, stats), False, stats)
            finally:
                torch.npu.empty_cache()
                gc.collect()
        chunks = chunks_from_tokenss(prompt, self.sep)
        chunks, query = chunks[:-1], chunks[-1].tokens
        stats["chunks"] = [c.cid for c in chunks]
        if not chunks:
            try:
                return (self._direct_query_no_context(query, stats), False, stats)
            finally:
                torch.npu.empty_cache()
                gc.collect()
        cache, _ = full_prefill_cache(chunks, self.llm, output_attentions=False)
        
        try:
            # snapshot modification 
            if snapshot_path:
                try:
                    from cachebend.snapshot import save_snapshot
                    meta = [{"cid": c.cid, "len": len(c)} for c in chunks]
                    save_snapshot(snapshot_path, "baseline", cache, query, meta)
                except ImportError:
                    print("Warning: cachebend.snapshot not found. Skipping save.")
            # snapshot modification

            return (do_query_with_state(self.llm, self.tokenizer, cache, query, False, stats), False, stats)
        finally:
            #little additon of mine, just to be a bit less messy with mem managment
            del cache
            torch.npu.empty_cache()
            gc.collect()
        #return (do_query_with_state(self.llm, self.tokenizer, cache, query, False, stats), False, stats)
    def begin_fresh_query(self):
        """Replace internal cache with a new empty one, preserving M and R."""
        pass


class BaselineNosep:
    """100%-recompute baseline that runs *without* DSEP in the forward pass.

    Flow per query:
        tokenized prompt (with DSEP, single BOS from chat template)
          → dedupe leading BOS defensively
          → split on DSEP (chunks_from_tokenss also strips DSEP from output)
          → concat chunk tokens (no separator between them) for a single prefill
          → manual token-by-token decode of the last chunk (the query)
            reusing the prefill KV cache

    Differences vs. Baseline:
      • The original Baseline's primary path (force_full=True in the
        benchmark) forwarded the prompt with DSEP tokens included. This
        class always runs DSEP-stripped — the same token layout CacheBlend
        uses when it rebuilds chunk KV caches.
      • Fixes the double-BOS injection: chat templates emit BOS as text and
        tokenize to bos_token_id; a subsequent tokenize(..., add_special_tokens=True)
        prepends a second BOS. _dedupe_leading_bos collapses consecutive BOS
        ids at the start of the input.
      • Only DSEP (tokenizer.sep_token_id) defines chunk boundaries — never
        splits on "\\n\\n" or any other whitespace marker.
    """

    def __init__(self, device: str, llm=None, tkn=None):
        self.device = torch.device(device)
        if not llm:
            config = AutoConfig.from_pretrained(model_path)
            config.sliding_window = None
            config._attn_implementation = "eager"
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path, config=config, output_attentions=True)
            self.llm.to(self.device)
            self.llm.to(torch.bfloat16)
            self.tokenizer = build_tokenizer(model_path)
        else:
            self.llm = llm
            self.tokenizer = tkn
        self.llm.eval()
        self.model = self.llm.model
        self.ascii_sep = self.tokenizer.sep_token
        self.sep = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.int64)

    def to_str_prompt(self, lis) -> str:
        return to_str_prompt(self.tokenizer, self.ascii_sep, lis)

    def tokenize(self, doc: str, use_special_tokens: bool = False) -> torch.Tensor:
        # Default add_special_tokens=False: the chat template (applied inside
        # to_str_prompt) already injected BOS. Setting True here would inject
        # a second one — _dedupe_leading_bos cleans it up regardless.
        return self.tokenizer(
            doc, return_tensors="pt", add_special_tokens=use_special_tokens
        )["input_ids"][0].to(torch.int64)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
        return text

    def _dedupe_leading_bos(self, tokens: torch.Tensor) -> torch.Tensor:
        bos_id = self.tokenizer.bos_token_id
        if bos_id is None or tokens.numel() < 2:
            return tokens
        i = 0
        while i + 1 < tokens.numel() and int(tokens[i]) == bos_id and int(tokens[i + 1]) == bos_id:
            i += 1
        return tokens[i:] if i > 0 else tokens

    @torch.no_grad()
    def _direct_query_no_context(self, query: torch.Tensor, stats: dict, max_length: int = 50):
        # Threads past_key_values across decode steps — same pattern as
        # `do_query_with_state` in cutils.py. Without this, every step
        # after the first re-encodes a single token with no history and
        # the model degenerates to "predict the next token from a token
        # in isolation" (typically high-frequency garbage).
        # Stop conditions: EOS OR a "\n\n" sequence appears in the
        # accumulating decoded text — the latter mirrors vLLM's
        # `stop=["\n\n"]` (used by the v8 gate's T1a/T1b), so this
        # path's predictions are directly comparable to the gate's.
        output = []
        accum  = ""
        input_ids = query.to(self.llm.device).unsqueeze(0)
        past = None
        for _ in range(max_length):
            outputs = self.llm(input_ids=input_ids, past_key_values=past,
                               use_cache=True, output_attentions=False)
            logits = outputs.logits
            past = outputs.past_key_values
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            decoded_token = self.tokenizer.decode(next_token_id, skip_special_tokens=True)
            output.append(decoded_token)
            accum += decoded_token
            input_ids = next_token_id.unsqueeze(0)
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
            if "\n\n" in accum:
                break
        stats["num_fresh_chunks"] = 0
        stats["num_cached_chunks"] = 0
        return "".join(output)

    @torch.no_grad()
    def new_query(self, prompt: torch.Tensor, force_full: bool = False,
                  snapshot_path=None, max_new_tokens: int = 50):
        """
        force_full is accepted for API parity but ignored — this class
        always runs the DSEP-stripped path.
        max_new_tokens caps greedy decoding length (default 50).
        """
        stats = {}
        prompt = self._dedupe_leading_bos(prompt)

        chunks = chunks_from_tokenss(prompt, self.sep)
        if not chunks:
            return (self._direct_query_no_context(prompt, stats, max_length=max_new_tokens), False, stats)
        if len(chunks) == 1:
            # No DSEP in the prompt — treat the whole thing as a query.
            try:
                return (self._direct_query_no_context(chunks[0].tokens, stats, max_length=max_new_tokens), False, stats)
            finally:
                torch.npu.empty_cache()
                gc.collect()

        doc_chunks, query = chunks[:-1], chunks[-1].tokens
        stats["chunks"] = [c.cid for c in doc_chunks]
        stats["num_fresh_chunks"] = len(doc_chunks)
        stats["num_cached_chunks"] = 0

        cache, _ = full_prefill_cache(doc_chunks, self.llm, output_attentions=False)
        try:
            if snapshot_path:
                try:
                    from cachebend.snapshot import save_snapshot
                    meta = [{"cid": c.cid, "len": len(c)} for c in doc_chunks]
                    save_snapshot(snapshot_path, "baseline_nosep", cache, query, meta)
                except ImportError:
                    print("Warning: cachebend.snapshot not found. Skipping save.")
            return (do_query_with_state(
                self.llm, self.tokenizer, cache, query, False, stats,
                max_length=max_new_tokens), False, stats)
        finally:
            del cache
            torch.npu.empty_cache()
            gc.collect()

    def begin_fresh_query(self):
        """No-op: BaselineNosep holds no persistent state between queries."""
        pass


def test():
    config = AutoConfig.from_pretrained(model_path)
    config.sliding_window = None
    dev = "npu:0"
    llm = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16)
    llm.to(dev)
    tokenizer = build_tokenizer(model_path) 
    cache = ZCFCache_V1(1, 0.0) 
    ccm = ZCF(cache=cache, device=dev, llm=llm, tokenizer=tokenizer, loading_mode="generator")
    docs = generate_triplets(10)
    SP = [["SYS PROMPT"] for _ in docs]
    queries = [["wut?"] for _ in docs]
    for sp, doc, q in zip(SP, docs, queries):
        print(sp, doc, q)
        answ = ccm.new_query(ccm.tokenize(ccm.to_str_prompt(sp + doc + q), use_special_tokens=True))
        print(f"{answ=}\n\n\n\n**********************************************\n\n\n")

if __name__ == "__main__":
    test()
