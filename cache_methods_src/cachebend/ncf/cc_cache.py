from time import perf_counter as now
import numpy as np
from scipy.stats import kendalltau
import torch_npu
from cachebend.ncf.cutils import chash, Chunk, PosChunk, chunks_from_tokenss
from cachebend.ncf.cutils import dynamic_cache_layer_kv, dynamic_cache_num_layers

import torch, hashlib, copy, os
from functools import partial
from typing import List, Dict
from .cutils import to_str_prompt, build_tokenizer
from transformers import (
    AutoTokenizer,
    DynamicCache,
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedTokenizer,
)
import os, sys, gc

from cachebend.utils import Timer
from cachebend.cacheblend import CacheBlendImpl, rope32
from cachebend.llm_docs import build_blent_cache, generator_build_blent_cache4, seq_build_blent_cache, overlap_build_blent_cache
import logging
from copy import deepcopy
from .cutils import full_prefill_cache, do_query_with_state

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["ASCEND_RT_VISIBLE_DEVICES"]="7"
model_path = "/data/weights/llama3.1-70B-I"
model_path = "/data/weights/llama3.1-8BI"
model_path = "/data/weights/mistral"
model_path = "/data/weights/llama3.1-8BI"

# We need the sys prompt for cross-attn so that even chunk 1 can have recomp tookens
# We probably do not need this for the beta, but let's keep this consistent for now
SKIP_PROMPT_IN_CROSS = False

CID_TO_TKNS = {}
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
cache_log = logging.getLogger("cachecraft")
cache_log.setLevel(logging.INFO)
attn_log = logging.getLogger("attn")
attn_log.setLevel(logging.INFO)
cfo_log = logging.getLogger("cfo")
cfo_log.setLevel(logging.INFO)

chunk_log = logging.getLogger("chunk")
chunk_log.setLevel(logging.WARN)


def _is_zero_recompute_ratio(value) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def generate(
    llm: AutoModelForCausalLM,
    past_key_values: DynamicCache,
    tokenizer: PreTrainedTokenizer,
    prompt: str | torch.Tensor,
    max_length: int = 20,
    device: str = "npu",
):
    """
    past_key_value will be modified
    """
    output = []
    with torch.no_grad():
        if isinstance(prompt, str):
            assert False
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
        else:
            input_ids = prompt

        for _ in range(max_length):
            outputs = llm(
                input_ids=input_ids, past_key_values=past_key_values, use_cache=True
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            decoded_token = tokenizer.decode(next_token_id, skip_special_tokens=True)
            output.append(decoded_token)

            input_ids = next_token_id.unsqueeze(0)

            # Optionally stop on end-of-text token
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return "".join(output)


class CCChunk:

    def b_ci_bar(self) -> float:
        if self.is_stub: return 1.0
        return torch.mean(self.self_attn_info)

    def a_ci_at_layer(self, layer: int) -> float:
        if self.is_stub: return 0.0
        ci_card = self.end - self.start
        attn = 0
        for cj in self.get_prefix_for_crossattn():
            cj_card = cj.end - cj.start
            cross_attn = self.attn_to_chunk(cj.cid)[layer]
            attn += cross_attn / (ci_card * cj_card)
        return attn

    def a_ci_bar(self) -> float:
        if self.is_stub: return 0.0
        n_layers = len(self.self_attn_info)
        a_ci = 0
        for layer in range(n_layers):
            a_ci += self.a_ci_at_layer(layer)
        return a_ci

    def cci(self) -> float:
        if self.is_stub: return 0.0
        from math import exp

        abar = self.a_ci_bar()
        bbar = self.b_ci_bar()
        return 1 / (1 + exp(-abar / bbar))

    def get_prefix_for_crossattn(self):
        return self.prefix if not SKIP_PROMPT_IN_CROSS else self.prefix[1:]

    def compute_cross_attn_info(self, attn):
        attn_info_by_cid = {}
        attn_log.debug(
            f"Computing attn for {self} {attn[0].shape=} {self.prefix=}\n{attn[0][0]=}"
        )
        for layer, attn_l in enumerate(attn):
            for chunk in self.get_prefix_for_crossattn():
                # each attn has self.end (prefix_len) column and end-start (len) rows
                # attn is without batch (head, self_len, self_prefix_len)
                # all columns of prefix are unmasked
                # so the cross attnn towards chunk is all rows, and chunk start/end cols
                curr_att = torch.sum(attn_l[:, :, chunk.start : chunk.end])
                attn_info_by_cid.setdefault(chunk.cid, []).append(curr_att)
        for cid in attn_info_by_cid:
            attn_info_by_cid[cid] = torch.tensor(attn_info_by_cid[cid])
        attn_log.debug(f"{attn_info_by_cid=}")
        return attn_info_by_cid

    def compute_self_attn_info(self, attn):
        return torch.tensor(
            [torch.sum(attn_l[:, :, self.start : self.end]) for attn_l in attn]
        ).cpu()

    def __repr__(self):
        return f"{self.cid=} {self.prefix=}"

    def __str__(self):
        return f"{self.cid=} {self.prefix=}"

    def __len__(self):
        # Required by `generator_build_blent_cache4`'s recompute_ratio==0.0
        # fast path (llm_docs.py): it calls `len(k_source)` to size the
        # destination KV slice. Without this, V3 (and V1) cachecraft
        # silently crash inside the fast path → empty cache → F1=0.
        return self.end - self.start

    def top_cross_attn(self, topr, attn):
        start = min([x.start for x in self.get_prefix_for_crossattn()])
        end = max([x.end for x in self.get_prefix_for_crossattn()])
        # (heads, self_tkns, all_tkns)
        # cross_attns = [ torch.sum(torch.sum(attn_l[:,:,start:end],dim=0,keepdim=False),dim=-1,keepdim=False) for attn_l in attn]   #do the sum first across all heads, then across tkns
        # sum across dim 0 and dim 2 in one go
        cross_attns = [
            torch.sum(attn_l[:, :, start:end], dim=(0, -1), keepdim=False)
            for attn_l in attn
        ]  # do the sum first across all heads, then across tkns

        row_sums = torch.stack(cross_attns).sum(dim=0)
        k = int(max(1, topr * row_sums.shape[0]))
        _, topk_idx = torch.topk(row_sums, k)
        attn_log.debug(
            f"Computing attn for {self} {self.start=} {self.end=} {attn[0].shape=} {self.prefix=}\n{attn[0][0]=} with actul prefix {self.get_prefix_for_crossattn()}\n{start=} {end=}.  top {k=} {topk_idx=}"
        )
        return topk_idx.tolist()

    def __init__(
        self, cid, states, attn, prefix: list[PosChunk], start, end, recomp_ratio,
        is_stub=False, attn_free=False, final_layer_key=None,
    ):
        self.cid = cid
        self.states = states
        # print(f"Creating new CChunkf for {cid=}. {len(states)=} {[(k.shape,v.shape,a.shape) for k,v,a in states]} ")
        self.prefix = prefix
        self.start = start
        self.end = end
        self.is_stub = is_stub
        self.attn_free = attn_free
        self.final_layer_key = final_layer_key

        # STUB MODE SHORTCUT
        # If stub, we skip all heavy tensor computation (attention sum/sort)
        if self.is_stub:
            self.recomp_tkns = [] # Caller fill this manually if needed
            self.cross_attn_info = {} # Dummy dict
            self.self_attn_info = torch.tensor([1.0]) # Dummy tensor to avoid division by zero
            self.cci = 0.5 # Dummy value
            return
        #########

        # ATTN-FREE MODE (CacheCraft V3): real KV but no attention matrix.
        # cci/recomp_tkns are not meaningful; β-only selection is done by the
        # V3 cache class via overlap of prefix cids. Token recomputation is
        # decided at blend time (DiffKV or query-guided), not at fill time.
        if self.attn_free:
            self.recomp_tkns = []
            self.cross_attn_info = {}
            self.self_attn_info = torch.tensor([1.0])
            self.cci = 1.0
            return

        try:
            self.recomp_tkns = self.top_cross_attn(recomp_ratio, attn)
            assert len(self.recomp_tkns)
        except:
            assert (
                len(prefix) == 0
            ), f"top_cross_attn failed for {cid=} {prefix=} {start=} {end=} {len(prefix)=}. THis should only happen for the prompt (no prefix) or for the first chunk, which has cross-attn only with the prompt chunk that we discard"
            self.recomp_tkns = []
        # for each cid in prefix: cid , [ cross_attn_to_cid(l) for l in layers]
        self.cross_attn_info = self.compute_cross_attn_info(attn)
        # [self_attn(l) for l in layers]
        self.self_attn_info = self.compute_self_attn_info(attn)
        self.cci = self.cci()
        
    ##################################################################
    #new addition from Sam
    def get_layer(self, layer_idx: int, kv_idx: int, device = None):
        #like in zcf
        return self.states[layer_idx][kv_idx]

    ################################################################

    def state(self, i,device):
        return [state[i].to(device,non_blocking=True) for state in self.states]

    def ks(self,device):
        return self.state(0,device)

    def vs(self,device):
        return self.state(1,device)

    def attn_to_chunk(self, cid):
        if self.is_stub or self.attn_free: return torch.tensor([0.0]*32)
        return self.cross_attn_info[cid]

    def sum_cross_attns(self):
        if self.is_stub or self.attn_free: return 0.0
        try:
            return [
                torch.sum(torch.stack(xs))
                for xs in zip(
                    *[
                        self.attn_to_chunk(x.cid)
                        for x in self.get_prefix_for_crossattn()
                    ]
                )
            ].cpu()
        except:
            return torch.tensor([0 for _ in self.states])

class CacheCraftCache:
    ccchunk_map: Dict[int, List[CCChunk]]

    def __init__(self,N,M,R):
        self.ccchunk_map = {}
        self.cache_dev = torch.device("cpu")
        self.N = N
        self.M = M
        self.recomp_ratio = R
        cache_log.info(f"Created {N=} {M=} {R=}. Note, N is not used now")

    def beta_prime(self, ci: CCChunk, S_new: list[Chunk]):
        if ci.is_stub: return 0.0
        beta = self.beta(
            ci, S_new
        )  # computed between ci.prefix and whole S_new (as per ccraft paper)
        START = 1 if SKIP_PROMPT_IN_CROSS else 0
        gamma = self.normalized_kendall_tau(
            ci.prefix[START:], S_new[START:]
        )  # computed between ci.prefix and whole S_new
        return beta * (1 - gamma)

    """
    As per cachecraft paper, beta is computed on the intersection of the PREFIX of the current chunk, and the WHOLE new sequence
    """

    def beta(self, ci: CCChunk, S_new: list[Chunk]):
        if ci.is_stub: return 0.0
        # 1 identify which chunk ids in ci and Snew are shared
        cfo_log.debug(f"Evaluating beta for {ci=} vs {S_new}")

        # todo: elegant way for cutting rpefix also for chunk list
        START = 1 if SKIP_PROMPT_IN_CROSS else 0
        S_oldp = [c.cid for c in ci.prefix[START:]]
        S_newp = [c.cid for c in S_new[START:]]
        common = set(S_oldp) & set(S_newp)
        cfo_log.debug(f"{S_oldp=} {S_newp=} {common=}")
        if len(common) == 0:
            cfo_log.debug(f"Beta for {ci=} {S_new=} is 0")
            return 0
        # 2 grab the interattention of common and of all (one entry per layer)

        # if we skip the common prompt prefix, it can be that ci has no cross attn becaseu is was
        # [prompt Ci]. In that case beta is 0. There is no shared cross attn
        # in that case, torch will try to operate on 0-length tensors, and we return 0
        try:
            A = sum_common_crossattn = [
                torch.sum(torch.cat(xs))
                for xs in zip(*[ci.attn_to_chunk(x) for x in common])
            ]
            B = sum_all_crossattn = ci.sum_cross_attn
            
            if isinstance(B, float) and B == 0.0: return 0.0 # safety for stub
            
        except:
            return 0
        # 3 comptue the avg ratio
        cfo_log.debug(f"{A=} {B=}")
        return (A / B).mean()

    def fill_cache(
        self, chunks: List[Chunk], positions: List[tuple], cache: DynamicCache, out
    ):
        return self._fill_cache(chunks,positions,cache,out)

    def _fill_cache(
        self, chunks: List[Chunk], positions: List[tuple], cache: DynamicCache, out
    ):
        
        attns = out.attentions
        B = 0
        num_layers = dynamic_cache_num_layers(cache)
        
        cache_log.debug(f"Filling cache for {chunks=}")
        
        for pos, cid, prefix, start, end in positions:
            states = []
            for i in range(num_layers):
                k_layer, v_layer = dynamic_cache_layer_kv(cache, i)
                k_view = k_layer[B, :, start:end, :]
                v_view = v_layer[B, :, start:end, :]
                
                k_final = torch.empty(k_view.shape, dtype=k_view.dtype, device="cpu", pin_memory=True)
                v_final = torch.empty(v_view.shape, dtype=v_view.dtype, device="cpu", pin_memory=True)
                
                k_final.copy_(k_view)
                v_final.copy_(v_view)
                
                states.append((k_final, v_final))

            attn = [attns[i][B, :, start:end, :end] for i in range(num_layers)]
            
            ccc = CCChunk(cid, states, attn, prefix, start, end, self.recomp_ratio)
            
            try:
                self.ccchunk_map[cid].append(ccc)
                cache_log.debug(f"New variant for {cid}")
            except KeyError:
                self.ccchunk_map[cid] = [ccc]
                
        del out.attentions
        import gc
        gc.collect()
        torch.npu.empty_cache()
        return

    def match_first(
        self, chunks: List[Chunk],device
    ) -> dict[
        int, tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.LongTensor]
    ]:
        recomp_pos = {}
        cache_l = True
        cachedb = {} if not cache_l else []
        for chunk in chunks:
            match: CCChunk
            match = self.ccchunk_map[chunk.cid][0]
            if not cache_l:
                cachedb[chunk.cid] = (
                    chunk.tokens,
                    match.ks(device),
                    match.vs(device),
                    torch.arange(match.start, match.end),
                )
            else:
                cachedb.append((chunk.cid,
                    chunk.tokens,
                    match.ks(device),
                    match.vs(device),
                    torch.arange(match.start, match.end)
                ))
                
            recomp_pos[chunk.cid] = match.recomp_tkns
            # test
            cfo = self.cfo(match, chunks)
        return cachedb, recomp_pos

    def match_lowest_cfo(
        self, chunks: List[Chunk],device
    ) -> dict[
        int, tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.LongTensor]
    ]:
        #cached is now a list 
        cachedb_list = []  
        recomp_pos = {}
        
        for chunk in chunks:
            best_cfo = 1e9
            matches: list[CCChunk] = self.ccchunk_map.get(chunk.cid, [])
          
            if not matches:
                continue
                
            match = matches[0] 
            
            # If multiple variants, and not stub, pick best. 
            # In stub mode, cfo() returns 0.0, so this loop is harmless/equally fast.
            if len(matches) > 1:
                for m in matches:
                    cfo = self.cfo(m, chunks)
                    if cfo < best_cfo:
                        best_cfo = cfo
                        match = m

            cachedb_list.append((
                chunk.cid,
                chunk.tokens,
                match,  # k_source 
                match,  # v_souurce
                torch.arange(match.start, match.end),
            ))
            
            recomp_pos[chunk.cid] = sorted(match.recomp_tkns)

        return cachedb_list, recomp_pos
        
    def match(self, chunks: List[Chunk],device):
        return self.match_lowest_cfo(chunks,device)

    def reuse_cache(self, chunks: List[PosChunk], llm, stats, stub_mode = False, loading_mode = "generator"):
        #not sure sunchronize is useful here???
        #actually might be hurtful?
        #for now keep commented out
        #torch.npu.synchronize()
        cache_log.debug(f"{chunks=}")
        
        # Get List
        cache_db_list, recomp_tkns = self.match(chunks, llm.device)

        # Always compute stats (needed for evaluation)
        if stats is not None:
            # We use 'fresh_chunks_cids' which we populated in maybe_add
            added_sigs = set(stats.get("fresh_chunks_cids", []))
            usage_log = []
            
            # Since match processes chunks in order, cache_db_list corresponds to the matched ones.
            # If there was a cache miss (which shouldn't happen here if we prefilled), logic holds.
            for i, entry in enumerate(cache_db_list):
                # cache_db_list entry: (cid, tokens, k_src, v_src, range)
                cid = entry[0]
                matched_cc = entry[2]  # CCChunk
                
                status = "FRESH" if cid in added_sigs else "CACHED"
                
                # The length of the prefix of the variant we actually picked
                prefix_len = len(matched_cc.prefix)
                
                usage_log.append(f"CID:{cid}|{status}|MatchPLen:{prefix_len}")

            stats["chunk_breakdown"] = usage_log

        #SKIP BLENDING IN STUB MODE
        if stub_mode:
            return DynamicCache()  # empty but valid cache

        #non stub mode: run blending
        doc_ids = (
            torch.cat([chunk.tokens for chunk in chunks], dim=0)
            .unsqueeze(0)
            .to(llm.device)
        )

        if loading_mode == "generator":
            return generator_build_blent_cache4(
                llm.model, cache_db_list, doc_ids, recomp_tkns, 
                past_key_value=None, stats=stats
            )
        elif loading_mode == "overlap":
            return overlap_build_blent_cache(
                llm.model, cache_db_list, doc_ids, recomp_tkns, 
                past_key_value=None, stats=stats
            )
        else:
            return build_blent_cache(
                llm.model, cache_db_list, doc_ids, recomp_tkns, 
                past_key_value=None, stats=stats
            )

    def normalized_kendall_tau(self, list1: list[PosChunk], list2: list[PosChunk]):
        # Elements in common
        cfo_log.debug(f"{[x.cid for x in list1]=} {[x.cid for x in list2]=}")
        common = list(set([x.cid for x in list1]) & set([x.cid for x in list2]))
        if len(common) < 2:
            return 0.0  # Kendall distance is 0 if <2 items in common

        # Restrict to common elements, preserving original order
        l1 = [x.cid for x in list1 if x.cid in common]
        l2 = [x.cid for x in list2 if x.cid in common]

        # Map elements to original positions
        idx1 = {x: i for i, x in enumerate(l1)}
        idx2 = {x: i for i, x in enumerate(l2)}

        # Build aligned rank vectors
        order = sorted(common)  # consistent ordering
        cfo_log.debug(f"{common=} {order=}")
        v1 = [idx1[x] for x in order]
        v2 = [idx2[x] for x in order]

        # Compute Kendall tau correlation
        tau, _ = kendalltau(v1, v2)

        # Convert correlation, normalized distance
        # distance = (1 - tau) / 2, ranges from 0 (identical) to 1 (reversed)
        return (1 - tau) / 2

    def cfo(self, ci: CCChunk, S_new: list[Chunk]):
        if ci.is_stub: return 0.0
        alpha = 0.2
        cci = ci.cci
        beta_prime = self.beta_prime(ci, S_new)
        return alpha * cci * (1 - beta_prime)

    # returns the positions to add
    def maybe_add(self, chunks: list[PosChunk], tokenizer, stats = None):
        return self._maybe_add(chunks,tokenizer, stats = stats)
    
    def _maybe_add(self, chunks: list[PosChunk], tokenizer, stats = None):
        # Add when
        # prefix-chunk is not already in the cache AND
        # chunk has less than M occurrences
        positions = []
        added_log = []

        def dd(chunk):
            return f"{chunk.cid} -> {chunk.tokens[:5]=} {tokenizer.decode(chunk.tokens[:5])}"

        for pos, chunk in enumerate(chunks):
            chunk_vars = self.ccchunk_map.get(chunk.cid, None)
            
            #use slice instead of deepcopy, safe here as we only read
            curr_pref = chunks[:pos]
            
            def log_add(reason):
                prefix_len = len(curr_pref)
                added_log.append((chunk.cid, prefix_len))
            
            # chunk is not present. Save it
            if not chunk_vars:
                cache_log.debug(f"First time I see {dd(chunk)}. Appending {curr_pref=}")
                positions.append((pos, chunk.cid, curr_pref, chunk.start, chunk.end))
                log_add("New_Chunk")
                continue
                
            # chunk is present.
            # if there is still space left, check if an exact prefix+chunk
            # is already there. if not, add
            cache_log.debug(
                f"Evaluating alternatives: {len(chunk_vars)=} for  {dd(chunk)}"
            )
            if len(chunk_vars) < self.M:
                found = False
                for cvar in chunk_vars:
                    # Compare CIDs of prefix instead of full objects to avoid deepcopy issues
                    if [c.cid for c in cvar.prefix] == [c.cid for c in curr_pref]:
                        found = True
                        cache_log.debug(
                            f"Not adding: same combo found. {cvar.prefix=} vs {curr_pref=}..."
                        )
                        break
                if not found:
                    cache_log.debug(f"Adding!")
                    positions.append(
                        (pos, chunk.cid, curr_pref, chunk.start, chunk.end)
                    )
                    log_add("New_Variant")
            else:
                cache_log.debug(f"Not adding: M is {len(chunk_vars)=}")
                
        if stats is not None: 
            stats["num_added_variants"] = len(positions)
            stats["added_variants"] = [f"{p[1]}|PLen={len(p[2])}" for p in positions]
            stats["_internal_added_sigs"] = [f"{c}|{p}" for c, p in added_log]
            stats["added_variants_log"] = [f"CID:{c}|PLen:{p}" for c, p in added_log]
            stats["num_added"] = len(positions)
                
        return positions if len(positions) else None
    
    #new addition for stub mode
    def fill_cache_stub(self, chunks: List[Chunk], positions: List[tuple]):
        """
        In stub mode: create dummy CCChunk variants without running LLM.
        We don't have real attentions, so:
        recomp_tkns = ratio * length
        self_attn_info, cross_attn_info = skipped via is_stub=True
        Uses light tensor allocation (torch.empty) to avoid memory/time overhead.
        """
        for pos, cid, prefix, start, end in positions:
            # Dummy states: minimal allocation
            num_layers = 32  #nasty, for now HARDCODED! needs to be changed 
            states = []
            for _ in range(num_layers):
                # Use empty (fastest), no zeroing
                k_dummy = torch.empty((32, end - start, 128), dtype=torch.bfloat16)  
                v_dummy = torch.empty_like(k_dummy)
                states.append((k_dummy, v_dummy))

            # Pass attn=None and is_stub=True to skip all tensor math
            ccc = CCChunk(cid, states, attn=None, prefix=prefix, start=start, end=end, 
                          recomp_ratio=self.recomp_ratio, is_stub=True)
            
            # Manually set recomp tokens to mimic behavior
            actual_len = end - start
            recomp_count = max(1, int(self.recomp_ratio * actual_len))
            ccc.recomp_tkns = list(range(recomp_count))

            # Store in cache
            if cid not in self.ccchunk_map:
                self.ccchunk_map[cid] = []
            self.ccchunk_map[cid].append(ccc)


class CacheCraftCacheManager:
    llm: AutoModelForCausalLM

    def __init__(
        self,
        device: str,
        cache: CacheCraftCache,
        llm = None,
        tkn = None,
        stub_mode = False,
        loading_mode = "generator"
    ):
        # self.ccchunk_map : dict(int:List[CCChunk])
        assert loading_mode in ["", "sequential", "overlapping", "generator"], "passed invalid cache loading mode"
        
        self.loading_mode = loading_mode
        self.stub_mode = stub_mode
        self.ccchunk_map = {}
        self.device = torch.device(device)
        
        if self.stub_mode:
            cache_log.info("CacheCraft running in STUB MODE (Inference Disabled)")
        cache_log.info(f"CacheCraft initialized with cache loading mode: {self.loading_mode}")
        
        if not llm:
            assert not tkn
            config = AutoConfig.from_pretrained(model_path)
            config.sliding_window = None
            config._attn_implementation = "eager"  # to have attns
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path, config=config
            )  # , attn_implementation="eager")
            self.llm.to(self.device)
            self.llm.to(torch.bfloat16)
            self.tokenizer = build_tokenizer(model_path)
        else:
            #i guess i need to have eager here too?
            #i will try this out meow
            
            #this is nasty but for now it works
            ###########################################
            #TEMPORARY
            """
            config = AutoConfig.from_pretrained(model_path)
            config.sliding_window = None
            config._attn_implementation = "eager"  # to have attns
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path, config=config
            )  # , attn_implementation="eager")
            self.llm.to(self.device)
            self.llm.to(torch.bfloat16)
            self.tokenizer = build_tokenizer(model_path)
            """
            
            
            
            ############################################
            
            #the old version
            
            self.llm = llm
            self.tokenizer = tkn
            
            #end old version
            ########################################
            
            
            
        self.llm.eval()
        self.model = self.llm.model
        self.cache = cache

        self.sep = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.int64)
        self.ascii_sep = self.tokenizer.sep_token
        print(f"{self.sep=} {self.sep.shape=}")

        self.n_q = 0
        self.n_r = 0

    def to_str_prompt(self, lis) -> str:
        return to_str_prompt(self.tokenizer,self.ascii_sep,lis)

    def tokenize(self, doc: str, use_special_tokens: bool = False) -> torch.Tensor:
        # tokenizer gives me array of array...
        return self.tokenizer(
            doc, return_tensors="pt", add_special_tokens=use_special_tokens
        )["input_ids"][0].to(torch.int64)

    @torch.no_grad()
    def new_query(self, prompt: torch.Tensor, force_full: bool = False):
        # remove the query
        self.n_q += 1
        stats = {}
        cache_log.debug(f"New query {len(prompt)=} {prompt.shape=} {prompt[:15]=} {self.tokenizer.decode(prompt[:15])=}")
        chunks = chunks_from_tokenss(prompt, self.sep)
        chunks, query = chunks[:-1], chunks[-1].tokens
        cache_log.debug(
            f"\n{chunks=} {query=} {len(query)=} {sum([len(chunk) for chunk in chunks])=}"
        )
        
       
        positions = None
        if not force_full:
            with Timer("maybe_add", stats):
                positions = self.cache.maybe_add(chunks, self.tokenizer, stats=stats)
        
        
        if positions:
            stats["fresh_chunks_cids"] = [p[1] for p in positions]
        
        total_chunks = len(chunks)
        num_fresh = len(positions) if positions else 0
        num_cached = total_chunks - num_fresh
        stats["num_fresh"] = num_fresh
        stats["num_cached"] = num_cached
        stats["hit_rate"] = num_cached / total_chunks if total_chunks > 0 else 0

        cached = False
        if positions:
            if self.stub_mode:
                with Timer("fill_cache_stub", stats):
                    self.cache.fill_cache_stub(chunks, positions)
            else:
                with Timer("full_prefill_cache", stats):
                    cache, out = full_prefill_cache(chunks, self.llm, output_attentions=True)
                with Timer("fill_cache", stats):
                    self.cache.fill_cache(chunks, positions, cache, out)     
        else:
            cache_log.info(
                f"Reconstructing cache for prompt {prompt[:10]=} {len(prompt)=} {len(chunks)=}"
            )
            #this path is usually hit when everything is cached.
            #Normal logic falls through to the 'cached = True' below
            # original code called reuse_cache immediately in the else
            #
            cached = True
            self.n_r += 1
        
        #STUB MODE EXIT
        if self.stub_mode:
            # call reuse_cache here to generate the 'chunk_breakdown' stats
            # not sure this ensures ensures that we pick exactly which variants non  stub mode *would* have picked.
            with Timer("stub_reuse_logic_check", stats):
                _ = self.cache.reuse_cache(chunks, self.llm, stats, stub_mode=True, loading_mode=self.loading_mode)
            
            cache_log.info("Stub mode active: returning metrics only.")
            return "STUB_RESPONSE", cached, stats
        
        #NORMAL MODE GENERATION
        #I think If we didn't just fill 'cache' (because cached=True), or if we want to ensure
        #blending is applied to mixed fresh/cached content, we call reuse_cache.
        #IMPORTANT !!! Original code called reuse_cache only inside the 'else' of positions
        # OR relied on full_prefill returning 'cache'.
        # However, to get the blending benefits (and the breakdown stats), we generally need reuse_cache, I THINK
        if cached:
            with Timer("reuse_cache",stats):
                cache = self.cache.reuse_cache(chunks,self.llm,stats, stub_mode = False, loading_mode = self.loading_mode)

        assert len(cache) == self.model.config.num_hidden_layers
        with Timer("query",stats):
            try:
                return do_query_with_state(self.llm,self.tokenizer,cache, query, output_attentions=not cached,stats=stats), cached,stats
            finally:
                torch.npu.empty_cache()  
                gc.collect()

    def baseline_query(self, prompt: torch.Tensor):
        return self.new_query(prompt, force_full=True)

    def do_query_with_state(
        self, state: DynamicCache, query: torch.Tensor, output_attentions: bool,stats:Dict
    ) -> str:
        # Wrapper reusing global function or self.llm
        output = []
        input_ids = query.to(self.device).unsqueeze(0)

        max_length = 5
        with torch.no_grad():
            for i in range(max_length):
                outputs = self.llm(
                    input_ids=input_ids,
                    past_key_values=state,
                    use_cache=True,
                    output_attentions=output_attentions,
                )
                logits = outputs.logits
                state = outputs.past_key_values

                next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
                decoded_token = self.tokenizer.decode(
                    next_token_id, skip_special_tokens=True
                )
                output.append(decoded_token)

                input_ids = next_token_id.unsqueeze(0)

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    if i == 0:
                        print("EOS at token 0 usually means you have not added BOS")
                    break
        try:
            return "".join(output)
        finally:
            import gc
            gc.collect()
            torch_npu.npu.empty_cache()


def test_add():
    docs = [
        ["A B C", "B C D", "C D E", "D E F"],
        ["B C D", "A B C"],  # this is reuse because N == 1. Positions are changed tho
        ["A B C", "C D E"],  # this is NOT reuse because CDE has a different prefix
        ["A B C", "C D E", "D E F", "B C D"],
        ["A B C", "D E F"],
    ]
    SP = [["SYS PROMPT"] for _ in docs]
    queries = [["wut?"] for _ in docs]

    ccm = CacheCraftCacheManager(4, 1, 0.5)
    print(f"{ccm.sep=}")
    for sp, doc, q in zip(SP, docs, queries):
        print(sp, doc, q)
        answ = ccm.new_query(
            ccm.tokenize(ccm.to_str_prompt(sp + doc + q), use_special_tokens=True)
        )  # adds bos to sys prompt
        print(f"{answ=}\n\n\n\n**********************************************\n\n\n")


A1 = "A B C"
A2 = "B C D"
A3 = "C D E"
A4 = "D E F"
A5 = "E F G"
A6 = "F G H"
A7 = "G H I"
logging.getLogger("cacheblend").setLevel(logging.INFO)


def test_choice():
    docs = [
        [A1, A2, A3, A4],
        [A2, A4, A1],
        [A5, A1, A6],
        [A6, A1, A7],
        [A4, A7, A1],
    ]
    SP = [["SYS PROMPT"] for _ in docs]
    queries = [["wut?"] for _ in docs]
    ccm = CacheCraftCacheManager(100, 1, 0.5)
    print(f"{ccm.sep=}")
    for sp, doc, q in zip(SP, docs, queries):
        for i in range(2):
            print(sp, doc, q)
            answ = ccm.new_query(
                ccm.tokenize(ccm.to_str_prompt(sp + doc + q), use_special_tokens=False)
            )  # adds bos to sys prompt --> not anymroe: we use chat
            print(f"{answ=}\n")
        exit()

    print("******* NOW THE QUERY  ******")


if __name__ == "__main__":
    # test_add()
    test_choice()
    print("TODO")
    print("(1) Remove special tokens in cacheblend (prompt already has beg)")
    print("(2) Remove the # linking. Removing links meeses up the position (maybe)")


class RAGPrompt:
    def __init__(self, docs, query):
        self.docs = docs
        self.query = query


# ---------------------------------------------------------------------------
# CacheCraft V3 — multi-version, β-only (overlap), no attention matrix.
# ---------------------------------------------------------------------------
#
# V3 keeps V1's multi-versioning machinery (up to M variants per cid added by
# `_maybe_add` when an unseen prefix is observed) and the lowest-CFO variant
# pick, but drops everything that requires the attention matrix:
#
#   * No `output_attentions=True` anywhere → fast attention backend used on
#     every model call (no eager fallback, no [heads,T,T] tensor).
#   * No CCI: `cci` is constant 1.0, so CFO collapses to (1 − β').
#   * β is replaced by a pure prefix-overlap fraction: |Sold ∩ Snew| / |Sold|,
#     since attention-weighted β needs `cross_attn_info` we don't have.
#   * No precomputed `recomp_tkns` per variant: token recomputation is
#     decided at blend time by the manager (DiffKV or query-guided).
#
# Two managers wire V3 to the two token strategies:
#
#   * `CacheCraftCacheV3DiffKVManager` — variant pick → CacheBlend DiffKV
#     blend (`generator_build_blent_cache4` with float `recompute_ratio=R`).
#   * `CacheCraftCacheV3QManager` — variant pick → query · final-layer-K
#     selection (FR's `QueryGuidedSelector`) → blend with precomputed
#     indices.
# ---------------------------------------------------------------------------


class CacheCraftCacheV3(CacheCraftCache):
    """V3: β-overlap selection, no attention storage.

    Inherits multi-versioning (`_maybe_add`, M-variant cap, fill-on-add)
    from `CacheCraftCache`. Overrides:
      - `beta`: pure overlap |Sold ∩ Snew| / |Sold| (no attention).
      - `match_lowest_cfo`: causal-prefix scoring (Snew = chunks[:pos]),
        matching the paper's intent that β is over chunks preceding D.
      - `fill_cache_chunkwise`: per-variant prefix-then-chunk forward
        with `output_attentions=False`. Stores KV slice + final-layer K
        (cheap; the latter is consumed by V3-Q).
    """

    def beta(self, ci: CCChunk, S_new):
        # Pure overlap fraction. attn_free CCChunks have no
        # `cross_attn_info`, so the parent's attention-weighted beta
        # would always return 0 — replace with overlap count.
        if ci.is_stub:
            return 0.0
        START = 1 if SKIP_PROMPT_IN_CROSS else 0
        old = [c.cid for c in ci.prefix[START:]]
        new = {c.cid for c in S_new[START:]}
        if not old:
            return 0.0
        common = sum(1 for cid in old if cid in new)
        return common / len(old)

    def match_lowest_cfo(self, chunks: List[Chunk], device):
        """Causal-prefix variant pick: score against `chunks[:pos]`."""
        cachedb_list = []
        recomp_pos = {}
        for pos, chunk in enumerate(chunks):
            matches: list[CCChunk] = self.ccchunk_map.get(chunk.cid, [])
            if not matches:
                continue
            current_prefix = chunks[:pos]
            match = matches[0]
            if len(matches) > 1:
                best = 1e9
                for m in matches:
                    score = self.cfo(m, current_prefix)
                    if score < best:
                        best = score
                        match = m
            cachedb_list.append((
                chunk.cid,
                chunk.tokens,
                match,           # k_source — CCChunk has get_layer
                match,           # v_source
                torch.arange(match.start, match.end),
            ))
            recomp_pos[chunk.cid] = []  # decided at blend time
        return cachedb_list, recomp_pos

    def fill_cache_chunkwise(self, chunks: List[Chunk], positions, llm):
        """Per-variant prefix-then-chunk forward, no attentions.

        Mirrors the reference `fill_cache_chunkwise` (avoids the
        full-prompt attention tensor) but drops `output_attentions=True`
        on the chunk forward — V3 needs no attention matrix at all.
        """
        device = llm.device
        forward_kwargs = {}
        try:
            if "logits_to_keep" in inspect.signature(llm.forward).parameters:
                forward_kwargs["logits_to_keep"] = 1
        except Exception:
            pass

        for pos, cid, prefix, start, end in positions:
            prefix_tokens = [chunks[p].tokens for p in range(pos)]
            past = None
            if prefix_tokens:
                prefix_ids = (
                    torch.cat(prefix_tokens, dim=0).to(device).unsqueeze(0)
                )
                prefix_out = llm(
                    prefix_ids,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    **forward_kwargs,
                )
                past = prefix_out.past_key_values
                del prefix_out, prefix_ids

            chunk_ids = chunks[pos].tokens.to(device).unsqueeze(0)
            out = llm(
                chunk_ids,
                past_key_values=past,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                **forward_kwargs,
            )
            cache = out.past_key_values
            states = []
            num_layers = dynamic_cache_num_layers(cache)
            for i in range(num_layers):
                k_layer, v_layer = dynamic_cache_layer_kv(cache, i)
                k_view = k_layer[0, :, start:end, :]
                v_view = v_layer[0, :, start:end, :]
                k_final = torch.empty(k_view.shape, dtype=k_view.dtype,
                                      device="cpu", pin_memory=True)
                v_final = torch.empty(v_view.shape, dtype=v_view.dtype,
                                      device="cpu", pin_memory=True)
                k_final.copy_(k_view)
                v_final.copy_(v_view)
                states.append((k_final, v_final))

            final_layer_key = states[-1][0]  # CPU pinned, [n_kv, len, dim]
            ccc = CCChunk(
                cid=cid,
                states=states,
                attn=None,
                prefix=prefix,
                start=start,
                end=end,
                recomp_ratio=self.recomp_ratio,
                attn_free=True,
                final_layer_key=final_layer_key,
            )
            self.ccchunk_map.setdefault(cid, []).append(ccc)

            del out, cache, chunk_ids, past, states
            gc.collect()
            if hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
                torch.npu.empty_cache()


# ---------------------------------------------------------------------------
# V3 managers — share `new_query` skeleton, differ in token-selection step.
# ---------------------------------------------------------------------------


class _CacheCraftCacheV3ManagerBase:
    """Common scaffolding for V3 managers — splits prompt, calls maybe_add,
    runs `fill_cache_chunkwise` (no attentions) on positions, then dispatches
    to the subclass's `_blend(...)`.
    """

    def __init__(self, device, cache: CacheCraftCacheV3, llm, tkn,
                 stub_mode: bool = False, loading_mode: str = "generator"):
        assert loading_mode in ("", "sequential", "overlapping", "generator"), \
            f"invalid loading_mode {loading_mode!r}"
        self.device = torch.device(device)
        self.stub_mode = stub_mode
        self.loading_mode = loading_mode
        self.cache = cache
        self.llm = llm
        self.tokenizer = tkn
        self.llm.eval()
        self.model = self.llm.model
        self.sep = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.int64)
        self.ascii_sep = self.tokenizer.sep_token
        self.num_layers = self.llm.config.num_hidden_layers
        self.final_layer_idx = self.num_layers - 1
        self.n_q = 0
        # V3 has no neighbour fusion — these stubs let runners written for
        # the cb/fr on-the-fly managers (which require both) call us
        # uniformly without special-casing.
        self.top_k = 0

    def configure_query_neighbors(self, *_args, **_kwargs):
        # No-op: V3 has no neighbour-fusion step.
        return

    def to_str_prompt(self, lis):
        return to_str_prompt(self.tokenizer, self.ascii_sep, lis)

    def tokenize(self, doc, use_special_tokens: bool = False):
        return self.tokenizer(
            doc, return_tensors="pt", add_special_tokens=use_special_tokens
        )["input_ids"][0].to(torch.int64)

    @torch.no_grad()
    def new_query(self, prompt: torch.Tensor, force_full: bool = False):
        self.n_q += 1
        stats: Dict = {}

        chunks = chunks_from_tokenss(prompt, self.sep)
        chunks, query = chunks[:-1], chunks[-1].tokens

        if force_full:
            if self.stub_mode:
                return "STUB_FULL", False, stats
            cache, _ = full_prefill_cache(chunks, self.llm, output_attentions=False)
            return (
                do_query_with_state(self.llm, self.tokenizer, cache, query,
                                    output_attentions=False, stats=stats),
                False, stats,
            )

        with Timer("maybe_add", stats):
            positions = self.cache.maybe_add(chunks, self.tokenizer, stats=stats)

        if positions:
            stats["fresh_chunks_cids"] = [p[1] for p in positions]
            if self.stub_mode:
                with Timer("fill_cache_stub", stats):
                    self.cache.fill_cache_stub(chunks, positions)
            else:
                with Timer("fill_cache_chunkwise_no_attn", stats):
                    self.cache.fill_cache_chunkwise(chunks, positions, self.llm)

        total_chunks = len(chunks)
        num_fresh = len(positions) if positions else 0
        stats["num_fresh"] = num_fresh
        stats["num_cached"] = total_chunks - num_fresh
        stats["hit_rate"] = (total_chunks - num_fresh) / total_chunks if total_chunks else 0

        if self.stub_mode:
            return "STUB_RESPONSE", False, stats

        with Timer("reuse_cache", stats):
            cache_db_list, _ = self.cache.match(chunks, self.llm.device)
            cache = self._blend(chunks, query, cache_db_list, stats)

        with Timer("query", stats):
            try:
                return (
                    do_query_with_state(self.llm, self.tokenizer, cache, query,
                                        output_attentions=False, stats=stats),
                    bool(positions is None),
                    stats,
                )
            finally:
                if hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
                    torch.npu.empty_cache()
                gc.collect()

    def baseline_query(self, prompt: torch.Tensor):
        return self.new_query(prompt, force_full=True)

    def begin_fresh_query(self):
        # No per-query state to reset — variants persist across queries.
        return

    # --- subclass hook ----------------------------------------------------

    def _blend(self, chunks, query, cache_db_list, stats):
        raise NotImplementedError


class CacheCraftCacheV3DiffKVManager(_CacheCraftCacheV3ManagerBase):
    """V3 + CacheBlend DiffKV blend (layer-1 ΔK token selection)."""

    def _blend(self, chunks, query, cache_db_list, stats):
        doc_ids = (
            torch.cat([c.tokens for c in chunks], dim=0)
            .unsqueeze(0)
            .to(self.llm.device)
        )
        return generator_build_blent_cache4(
            self.llm.model, cache_db_list, doc_ids,
            self.cache.recomp_ratio,
            past_key_value=None, stats=stats,
        )


class CacheCraftCacheV3QManager(_CacheCraftCacheV3ManagerBase):
    """V3 + query-guided token selection (FR-style).

    Order: β-pick variants → query · final-layer-K of selected variants →
    top-R% global indices → blend with precomputed indices.
    """

    def __init__(self, device, cache: CacheCraftCacheV3, llm, tkn,
                 stub_mode: bool = False, loading_mode: str = "generator"):
        super().__init__(device, cache, llm, tkn,
                         stub_mode=stub_mode, loading_mode=loading_mode)
        # Lazy import to avoid circular dep at module load time.
        from cachebend.ncf.fusionrag import QueryGuidedSelector
        from cachebend.llm_docs import (
            positional_encoder, reverse_positional_encoder,
        )
        self._selector = QueryGuidedSelector(recomp_ratio=cache.recomp_ratio)
        self._selector.set_positional_encoders(
            positional_encoder=partial(positional_encoder, self.model),
            reverse_positional_encoder=partial(reverse_positional_encoder, self.model),
        )

    def _compute_query_final_q(self, query_tokens, context_length, stats):
        """Final-layer Q for the query, RoPE-rotated to its position in the
        full prompt. Mirrors `FusionRAGCacheManager._compute_query_final_layer_q`.
        """
        from cachebend.llm_docs import apply_rotary_pos_emb
        with Timer("query_final_q_computation", stats=stats):
            query_ids = query_tokens.unsqueeze(0).to(self.device)
            query_len = query_ids.shape[1]
            position_ids = torch.arange(
                context_length, context_length + query_len,
                device=self.device, dtype=torch.long,
            ).unsqueeze(0)
            attention_mask = torch.ones_like(query_ids, device=self.device)

            outputs = self.model(
                input_ids=query_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            if outputs.hidden_states is None or len(outputs.hidden_states) < 2:
                raise RuntimeError("V3-Q query prefill did not return hidden states.")
            hidden_before_final = outputs.hidden_states[-2]
            final_layer = self.model.layers[self.final_layer_idx]
            normed = final_layer.input_layernorm(hidden_before_final)
            q_proj = final_layer.self_attn.q_proj(normed)

            cfg = self.llm.config
            n_heads = getattr(cfg, "num_attention_heads", None)
            head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_heads)
            q_proj = q_proj.view(1, query_len, n_heads, head_dim).transpose(1, 2)

            cos, sin = self.model.rotary_emb(q_proj, position_ids)
            q_proj, _ = apply_rotary_pos_emb(q_proj, q_proj, cos, sin)
            return q_proj.squeeze(0)  # [n_heads, query_len, head_dim]

    def _blend(self, chunks, query, cache_db_list, stats):
        device = self.llm.device
        chunk_lengths = []
        for entry in cache_db_list:
            _cid, _tokens, k_src, _v_src, _orig_pos = entry
            assert isinstance(k_src, CCChunk), \
                f"V3-Q expected CCChunk source, got {type(k_src)}"
            chunk_lengths.append(k_src.end - k_src.start)

        context_length = sum(chunk_lengths)

        if _is_zero_recompute_ratio(self.cache.recomp_ratio):
            stats["fusionrag_selector_skipped"] = True
            stats["fusionrag_total_tokens"] = context_length
            stats["fusionrag_selected_count"] = 0
            stats["fusionrag_selection_ratio"] = 0.0
            stats["fusionrag_per_chunk_distribution"] = [
                0 for _ in chunk_lengths
            ]
            stats["recompute_indices_count"] = 0
            stats["recompute_ratio_actual"] = 0.0
            recompute_arg = 0.0
        else:
            # 1. Final-layer K from each picked variant (in cache_db_list order,
            #    which == chunks order). final_layer_key was stored at fill time.
            chunk_final_keys = []
            chunk_orig_ranges = []
            for entry in cache_db_list:
                _cid, _tokens, k_src, _v_src, _orig_pos = entry
                assert isinstance(k_src, CCChunk), \
                    f"V3-Q expected CCChunk source, got {type(k_src)}"
                chunk_final_keys.append(k_src.final_layer_key)
                chunk_orig_ranges.append((int(k_src.start), int(k_src.end)))

            # 2. Query Q at the final layer, rotated to position `context_length`.
            query_final_q = self._compute_query_final_q(
                query, context_length, stats
            )

            # 3. FR's selector → flat list of GLOBAL token indices over the
            #    concatenated context (it re-RoPEs each chunk's K to the new
            #    positions internally).
            recompute_indices = self._selector(
                query_final_q=query_final_q,
                chunk_final_keys=chunk_final_keys,
                chunk_lengths=chunk_lengths,
                chunk_original_position_ranges=chunk_orig_ranges,
                device=device,
                stats=stats,
            )

            stats["recompute_indices_count"] = len(recompute_indices)
            stats["recompute_ratio_actual"] = (
                len(recompute_indices) / context_length
                if context_length > 0
                else 0
            )
            recompute_arg = recompute_indices if recompute_indices else 0.0

        # 4. Blend with precomputed indices (gen4 takes a list of global ints).
        doc_ids = (
            torch.cat([c.tokens for c in chunks], dim=0)
            .unsqueeze(0)
            .to(device)
        )
        return generator_build_blent_cache4(
            self.llm.model, cache_db_list, doc_ids,
            recompute_arg,
            past_key_value=None, stats=stats,
        )
