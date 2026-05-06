from time import perf_counter as now
from cachebend.utils import Timer
from torch.nn.utils.rnn import pad_sequence
import torch
from cachebend.cacheblend import CacheBlendImpl, rope32
from cachebend.llm_docs import build_blent_cache
from transformers import (
    AutoTokenizer,
    DynamicCache,
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedTokenizer,
)
import torch_npu
from typing import List, Dict, Tuple
from cachebend.ncf.cutils import Chunk, PosChunk, CachedChunk, to_str_prompt,build_tokenizer,do_query_with_state
from cachebend.ncf.cutils import chash,Chunk,PosChunk,chunks_from_tokenss,full_prefill_cache, save_snapshot
from cachebend.ncf.cutils import dynamic_cache_layer_kv, dynamic_cache_num_layers
import logging, gc, os
from cachebend.llm_docs import build_blent_cache, seq_build_blent_cache, overlap_build_blent_cache, generator_build_blent_cache4, generator_build_blent_cache_unified


model_path = "/data/weights/mistral"
model_path = "/data/weights/llama3.1-8BI"

logging.basicConfig(
    level=getattr(logging, os.getenv("CACHEBEND_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
cache_log = logging.getLogger("cacheblend")
cache_log.setLevel(getattr(logging, os.getenv("CACHEBEND_LOG_LEVEL", "INFO").upper(), logging.INFO))
def isdbg():
    return cache_log.isEnabledFor(logging.DEBUG)


class CacheBlendCache:
    def __init__(self, R):
        self.chunk_to_cchunk :Dict[int,CachedChunk]= dict()
        self.use_pristine = True  # we can also 
        self.recomp_ratio = R
        
        self._seen_cids = set()

    def maybe_add(self,chunks:list[PosChunk],tokenizer, stats = None)->List[PosChunk]:
        #eliminate from the chunks all the chunks that already have one copy
        
        
        
        to_process = [c for c in chunks if c.cid not in self.chunk_to_cchunk]  
        
        positions = [PosChunk(c.tokens,0,len(c.tokens)) for c in to_process]
        
        if stats is not None:
            stats["_internal_added_set"] = [c.cid for c in to_process]
            stats["added_chunk_details"] = [f"CID:{c.cid}|Len{len(c)}" for c in to_process]
            stats["count_added"] = len(to_process)
            
        return positions

    def gen_ids(self,poschunks:list[PosChunk],llm,tokenizer):
        ids = [chunk.tokens_() for chunk in poschunks]
        return ids

    #imprtant modification to make it faster
    #Merged fill_states and fill_cache here to avoid the triple-copy, i think this was the second bottlenck after the interleaving
    
    def populate_cache_one_shot(self, poschunks:list[PosChunk], llm, tokenizer):
        if not poschunks: return

        
        ids = self.gen_ids(poschunks, llm, tokenizer)
        input_ids = pad_sequence(
            ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        input_ids = input_ids.to(device=llm.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

       
        
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
        
        if isdbg(): cache_log.debug(f"Populating cache for {len(poschunks)} chunks directly.")
        
        #Direct Copy NPU
        for pid, chunk in enumerate(poschunks):
            #logic from fill_cache: we treated them as independent, so they start at 0 in the batch
            curr_len = len(chunk.tokens)
            
            c_states = []
            for i in range(num_layers):
                #Slice the NPU tensor (View, no copy)
                #Shape: [Batch, Heads, Seq, Dim]
                #Slice :curr_len to ignore padding from the batch
                k_npu_slice = past_key_values[i][0][pid, :, :curr_len, :]
                v_npu_slice = past_key_values[i][1][pid, :, :curr_len, :]
                
                #Alloc Pinned CPU just once
                k_final = torch.empty_like(k_npu_slice, device="cpu", pin_memory=True)
                v_final = torch.empty_like(v_npu_slice, device="cpu", pin_memory=True)
                
                
                k_final.copy_(k_npu_slice, non_blocking=True)
                v_final.copy_(v_npu_slice, non_blocking=True)
                
                c_states.append((k_final, v_final))
            
            
            cchunk = CachedChunk(chunk.tokens_(), 0, curr_len, c_states)
            
            if cchunk.cid not in self.chunk_to_cchunk:
                self.chunk_to_cchunk[cchunk.cid] = cchunk

        #Wait for NPU copies to finish before 'outputs' variable dies
        # Otherwise Python GC frees the NPU memory mid-copy and crashes
        #but i'm not 100% sure here, this coudl also be a source of slowness
        
        torch.npu.synchronize()
        return

    #Deprecated Old 2-step process. Keeping for reference/internal shame.
    @torch.no_grad()
    def fill_states(self, poschunks:list[PosChunk],llm,tokenizer):
        
        pass
        
    #Deprecated Old fill_cache.
    def fill_cache(self, poschunks:list[PosChunk], cache: DynamicCache, llm):
        
        pass
            
    def recomp_tkns_for_chunks(self,chunks):
        return self.recomp_ratio  #CB just returns the ratio
    
    
    def reuse_cache(self, chunks: list[PosChunk], llm, tokenizer, stats, stub_mode = False,  loading_mode: str = "generator"):
        #temporary deaxctivated to test hopw to make the interleaving laoding work with this
       
        
        if isdbg(): cache_log.debug(f"{chunks=}")
        
        stats["num_used_cids"] = len(chunks)
        stats["used_cids"] = [c.cid for c in chunks]
        
        #prepping Doc IDs
        doc_ids = (
            torch.cat([chunk.tokens_() for chunk in chunks], dim=0)
            .unsqueeze(0)
            .to(llm.device, non_blocking=True)
        )
        
        recom_tkns = self.recomp_tkns_for_chunks(chunks)
        
        # fill the cache database
        cachedb = [] 
        used_log = []
        count_fresh = 0
        count_cached = 0
        added_set = stats.get("_internal_added_set", set()) if stats else set()
        
        with Timer("cachedb_prep", stats=stats):
            for chunk in chunks:
                cws: CachedChunk = self.chunk_to_cchunk[chunk.cid]
                
                is_new = chunk.cid in added_set
                status = "NEW" if is_new else "CACHED"
                if is_new: count_fresh += 1
                else: count_cached += 1
                
                used_log.append(f"CID:{chunk.cid}|{status}")

                #i need to pass the 'cws' object itself
                cachedb.append((
                    chunk.cid,
                    chunk.tokens_(),
                    cws,  # k_
                    cws,  # v_
                    torch.arange(cws.start, cws.end)
                ))
                
        if stats is not None:
            stats["chunk_breakdown"] = used_log
            stats["total_chunks"] = len(chunks)
            stats["num_fresh_chunks"] = count_fresh
            stats["num_cached_chunks"] = count_cached
            hit_tokens = sum(len(c.tokens) for c in chunks if c.cid not in added_set)
            miss_tokens = sum(len(c.tokens) for c in chunks if c.cid in added_set)
            total_tokens = hit_tokens + miss_tokens
            stats["mode"] = "cacheblend"
            stats["token_composition"] = {
                "hit_tokens": int(hit_tokens),
                "miss_tokens": int(miss_tokens),
                "hit_ratio": (float(hit_tokens) / float(total_tokens)) if total_tokens > 0 else 0.0,
                "total_tokens": int(total_tokens),
                "hit_chunks": int(count_cached),
                "miss_chunks": int(count_fresh),
                "total_chunks": int(len(chunks)),
            }
            if "_internal_added_set" in stats: del stats["_internal_added_set"]
            
        if stub_mode:
            return None

        #i think now we might have too many modes, nasty has to be cleanup later
        if loading_mode == "generator":
            # unified generator handles float/dict recompute ratios
            return generator_build_blent_cache_unified(
                llm.model, cachedb, doc_ids, recom_tkns, 
                past_key_value=None, stats=stats, use_piaffe=False
            )
        elif loading_mode == "overlap":
            return overlap_build_blent_cache(
                llm.model, cachedb, doc_ids, recom_tkns, 
                past_key_value=None, stats=stats, use_piaffe=self.use_piaffe
            )
        elif loading_mode == "sequential":
            return seq_build_blent_cache(
                llm.model, cachedb, doc_ids, recom_tkns, 
                past_key_value=None, stats=stats, use_piaffe=self.use_piaffe
            )
        else:
            return build_blent_cache(
                llm.model, cachedb, doc_ids, recom_tkns, 
                past_key_value=None, stats=stats, use_piaffe=self.use_piaffe
            )
            
    def compute_temp_cached_chunks(self, poschunks: list[PosChunk], llm, tokenizer):
        """Compute CachedChunk objects without saving them."""
        if not poschunks:
            return {}
        
        ids = [chunk.tokens_() for chunk in poschunks]
        input_ids = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device=llm.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

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
        cid_to_cchunk = {}

        for pid, chunk in enumerate(poschunks):
            curr_len = len(chunk.tokens)
            c_states = []
            for i in range(num_layers):
                k_npu_slice = past_key_values[i][0][pid, :, :curr_len, :]
                v_npu_slice = past_key_values[i][1][pid, :, :curr_len, :]
                k_final = torch.empty_like(k_npu_slice, device="cpu", pin_memory=True)
                v_final = torch.empty_like(v_npu_slice, device="cpu", pin_memory=True)
                k_final.copy_(k_npu_slice, non_blocking=True)
                v_final.copy_(v_npu_slice, non_blocking=True)
                c_states.append((k_final, v_final))
            
            cchunk = CachedChunk(chunk.tokens_(), 0, curr_len, c_states)
            cid_to_cchunk[cchunk.cid] = cchunk

        torch.npu.synchronize()
        return cid_to_cchunk


class CacheBlendCacheManager:
    def __init__(self,device:str,cache:CacheBlendCache,llm:None,tkn:None, stub_mode = False, loading_mode: str = "generator", no_reuse = False):
        assert loading_mode in ("", "sequential, overlapping", "generator"), "invalid cache loading mode passed"
        self.loading_mode = loading_mode
        self.device = torch.device(device)
        self.stub_mode = stub_mode
        self.no_reuse = no_reuse
        
        if self.stub_mode:
            cache_log.info("CacheBlend running in STUB MODE (Inference Disabled)")
        cache_log.info(f"CacheBlend initialized with cache loading mode: {self.loading_mode}")
        
        if not llm:
            assert not tkn
            config = AutoConfig.from_pretrained(model_path)
            config.sliding_window = None
            config._attn_implementation = "eager"  # to have attns
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path, config=config
            )  
            self.llm.to(self.device)
            self.llm.to(torch.bfloat16)
            self.tokenizer = build_tokenizer()
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
        self.cache.use_piaffe = False
        
    def enable_piaffe(self):
        self.cache.use_piaffe = True
    def tokenize(self, doc: str, use_special_tokens: bool = False) -> torch.Tensor:
        # tokenizer gives me array of array
        return self.tokenizer(
            doc, return_tensors="pt", add_special_tokens=use_special_tokens
        )["input_ids"][0].to(torch.int64)
        
        
    def to_str_prompt(self, lis) -> str:
        return to_str_prompt(self.tokenizer,self.ascii_sep,lis)
        
    @torch.no_grad()
    def new_query(self, prompt:torch.Tensor, force_full=False, snapshot_path = None):
        self.n_q+=1
        stats = {}
        
        if isdbg():cache_log.debug(f"New query {len(prompt)=} {self.tokenizer.decode(prompt[:15])=}")
        chunks = chunks_from_tokenss(prompt, self.sep)
        chunks, query = chunks[:-1], chunks[-1].tokens
        
        if isdbg():cache_log.debug(
            f"\n{chunks=} {query=} {len(query)=} {sum([len(chunk) for chunk in chunks])=} {prompt=}"
        )
        if force_full:
            if self.stub_mode: return "STUB_FULL", False, stats
            cache,_ = full_prefill_cache(chunks,self.llm,output_attentions=False)
            return do_query_with_state(self.llm,self.tokenizer,cache, query, output_attentions=False,stats=stats),False,stats
        else:
            #not force_full
            force_full_caches = False
            if force_full_caches:
                # just to debug
                assert self.cache.recomp_ratio == 0
                with Timer("full_prefill_cache",stats):
                    cache,_= full_prefill_cache(chunks,self.llm,output_attentions=False)
                if not force_full:
                    with Timer("fill_cache",stats):
                        self.cache.fill_full_cache(chunks, cache, self.llm)
            else:
                with Timer("maybe_add",stats):
                    positions = self.cache.maybe_add(chunks, self.tokenizer, stats = stats)
                    
                if positions:
                    if self.stub_mode:
                        # Compute fresh vs cached based on global history
                        query_cids = [c.cid for c in chunks]
                        cached_cids = [cid for cid in query_cids if cid in self.cache._seen_cids]
                        fresh_cids = [cid for cid in query_cids if cid not in self.cache._seen_cids]
                        cid_to_len = {c.cid: len(c.tokens) for c in chunks}
                        
                        # Update history
                        self.cache._seen_cids.update(query_cids)
                        
                        # Mock stats
                        total = len(chunks)
                        stats.update({
                            "mode": "cacheblend",
                            "num_fresh": len(fresh_cids),
                            "num_cached": len(cached_cids),
                            "hit_rate": len(cached_cids) / total if total > 0 else 0,
                            "fresh_chunks_cids": fresh_cids,
                            "chunk_breakdown": [
                                f"CID:{cid}|{'FRESH' if cid in fresh_cids else 'CACHED'}|MatchPLen:0"
                                for cid in query_cids
                            ],
                            "total_chunks": total,
                            "num_fresh_chunks": len(fresh_cids),
                            "num_cached_chunks": len(cached_cids),
                            "token_composition": {
                                "hit_tokens": int(sum(cid_to_len[cid] for cid in cached_cids)),
                                "miss_tokens": int(sum(cid_to_len[cid] for cid in fresh_cids)),
                                "hit_ratio": (
                                    float(sum(cid_to_len[cid] for cid in cached_cids))
                                    / float(sum(cid_to_len[cid] for cid in query_cids))
                                ) if query_cids else 0.0,
                                "total_tokens": int(sum(cid_to_len[cid] for cid in query_cids)),
                                "hit_chunks": int(len(cached_cids)),
                                "miss_chunks": int(len(fresh_cids)),
                                "total_chunks": int(total),
                            },
                        })
                        
                        cache_log.info("Stub mode: returning mock stats (no cache ops).")
                        return "STUB_RESPONSE", True, stats
                    else:
                        if self.no_reuse:
                        
                            positions = [PosChunk(c.tokens, 0, len(c.tokens)) for c in chunks]
                            temp_cchunk_map = self.cache.compute_temp_cached_chunks(positions, self.llm, self.tokenizer)
                            original_entries = {}
                            for cid, cchunk in temp_cchunk_map.items():
                                if cid in self.cache.chunk_to_cchunk:
                                    original_entries[cid] = self.cache.chunk_to_cchunk[cid]
                                self.cache.chunk_to_cchunk[cid] = cchunk
                        else:
                            
                            with Timer("populate_cache", stats):
                                self.cache.populate_cache_one_shot(positions, self.llm, self.tokenizer)
                    

        with Timer("reuse_cache",stats):
            if self.no_reuse:
                cache = self.cache.reuse_cache(chunks, self.llm, self.tokenizer, stats, stub_mode=self.stub_mode, loading_mode=self.loading_mode)
            else:
                
            
                cache = self.cache.reuse_cache(chunks,self.llm,self.tokenizer,stats, stub_mode = self.stub_mode, loading_mode = self.loading_mode)
                
                    
        # Clean up
        if self.no_reuse and positions:
            for cid in temp_cchunk_map:
                if cid in original_entries:
                    self.cache.chunk_to_cchunk[cid] = original_entries[cid]
                else:
                    self.cache.chunk_to_cchunk.pop(cid, None)
                    
        if self.no_reuse:
                stats["num_fresh"] = len(chunks)
                stats["num_cached"] = 0
                stats["hit_rate"] = 0.0
                stats["total_chunks"] = len(chunks)
                stats["num_fresh_chunks"] = len(chunks)
                stats["num_cached_chunks"] = 0
                stats["chunk_breakdown"] = [f"CID:{c.cid}|FRESH" for c in chunks]
                total_tokens = sum(len(c.tokens) for c in chunks)
                stats["mode"] = "cacheblend"
                stats["token_composition"] = {
                    "hit_tokens": 0,
                    "miss_tokens": int(total_tokens),
                    "hit_ratio": 0.0,
                    "total_tokens": int(total_tokens),
                    "hit_chunks": 0,
                    "miss_chunks": int(len(chunks)),
                    "total_chunks": int(len(chunks)),
                }
            
        with Timer("query",stats):
            #this has to be done better
            if cache:
                try:
                    if self.stub_mode:
                        cache_log.info("Stub mode active: returning metrics only.")
                        return "STUB_RESPONSE", True, stats
                    if snapshot_path:
                        try:
                            
                            meta = [{"cid": c.cid, "len": len(c)} for c in chunks]
                            save_snapshot(snapshot_path, "cb", cache, query, meta)
                        except Exception :
                            print("Cache snapshot saving failed.")
                            raise
                    return  do_query_with_state(self.llm,self.tokenizer,cache, query,output_attentions=False,stats=stats),True,stats
                finally:
                    torch.npu.empty_cache() 
                    gc.collect()
            else:
                cache, _ = full_prefill_cache(chunks, self.llm, output_attentions=False)
                if snapshot_path:
                    try:
                        
                        meta = [{"cid": c.cid, "len": len(c)} for c in chunks]
                        save_snapshot(snapshot_path, "cb", cache, query, meta)
                    except Exception :
                        print("Cache snapshot saving failed.")
                        raise
                return do_query_with_state(self.llm,self.tokenizer,cache, query, output_attentions=False,stats=stats),False,stats
        #maybe we need a fallback here
        
    

    def begin_fresh_query(self):
        """
        Start a new query without discarding the accumulated online cache.

        Plain CacheBlend in this benchmark should retain chunks once they have
        been seen in earlier queries of the same run.
        """
        return None
               


class EPICCache(CacheBlendCache):
    def __init__(self,R):
        super().__init__(R)
    def recomp_tkns_for_chunks(self,chunks):
        #the int is the percentage. Now we use x% per chunk to be comparable with CB
        nums = [int(len(c) * self.recomp_ratio)  for c in chunks]
        recom_tkns = {c.cid:[x for x in range(num)] for c,num in zip(chunks,nums)} 
        return recom_tkns  


class Link0Cache(EPICCache):
    def __init__(self,R,B=10):
        super().__init__(R)
        self.bos = B
        
    
    def gen_ids(self, poschunks:list[PosChunk],llm,tokenizer)->List[torch.tensor] :
        bos = torch.tensor([tokenizer.bos_token_d for  _ in range(self.bos)],dtype=torch.int64)
        ids = [chunk.tokens_() for chunk in poschunks]
        is_sysprompt = poschunks[0].start == 0
        start = 1 if is_sysprompt else 0
        # prepend <bos> to each doc
        ids_ = [torch.cat([bos,chunks]) for chunks in ids[start:]]
        if isdbg():cache_log.debug(f"{[i.shape for i in ids_]=}")
        if is_sysprompt:
            # sysprompt untouched
            ids_= [ids[0]]+ids_
        if isdbg():cache_log.debug(f"{[(x.shape,y.shape) for x,y in zip(ids,ids_)]=}") 
        return ids_
    
    def fill_cache(self, poschunks:list[PosChunk], cache: DynamicCache, llm):
        num_layers = dynamic_cache_num_layers(cache)
        if isdbg():cache_log.debug(
            f"Filling cache from {len(poschunks)=} and a cache {num_layers=}"
        )
        for pid,chunk in enumerate(poschunks):
            assert chunk.start == 0
            start = chunk.start
            end = chunk.end
            if start > 0:  #not for sysprompt
                assert False, "the sysprompt does not need to have the BOS but we need a way  to identify that this is the prompt (ALL start from 0)"
                start+=self.bos
                end+=self.bos
            if isdbg():cache_log.debug(f"{pid} {chunk.start=} {chunk.end=}--> {start=} {end=}")
            c_state = [
                (  # layer i, batch_entry pid, all heads, start:end, all dims
                    dynamic_cache_layer_kv(cache, i)[0][pid, :, start:end, :],
                    dynamic_cache_layer_kv(cache, i)[1][pid, :, start:end, :],
                )
                for i in range(num_layers)
            ]
            cchunk = CachedChunk(chunk.tokens_(),start,end,c_state)
            chunk.start = start  # poschunks is going to be used in reuse_cache later. Need to tell that the position now has changed
            chunk.end = end
            assert not cchunk.cid in self.chunk_to_cchunk
            self.chunk_to_cchunk[cchunk.cid]  = cchunk
