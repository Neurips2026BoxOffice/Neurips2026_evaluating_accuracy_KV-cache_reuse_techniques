# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Any,Union
import torch
from transformers.cache_utils import Cache
import os

from transformers.utils import logging
logger = logging.get_logger("blender")

EXTRA_INFO=int(os.environ.get('EXTRA_INFO', 0))
DUMP_K=int(os.environ.get('DUMP_K', 0))
extra_info = {}
ROPE32=False
def isdbg():
    return logger.isEnabledFor(logging.DEBUG)

def rope32(b=True):
    global ROPE32
    ROPE32 = b
    print(f"Enabling ROPE32")
@dataclass
class BlendOutput:
    """The output of the cacheblend module

    :ivar torch.Tensor q: The short Q tensor with selected tokens
    :ivar torch.Tensor k: The long K tensor with the updated values
    :ivar torch.Tensor v: The long V tensor with the updated values
    :ivar torch.Tensor positions: The positions of the selected Q tokens in
        the input sequence
    :ivar torch.Tensor local_indices: The positions of the selected Q tokens in
        fresh q
    :ivar Optional[torch.Tensor] query_start_loc: The modified query_start_loc
        if token selection has happened. Will be None if no selection has
        happened.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    positions: torch.Tensor
    local_indices: torch.Tensor
    query_start_loc: Optional[torch.Tensor]
    xattn_ids:Optional[torch.Tensor]


def mask_to_indices(mask):
    indices = mask.nonzero(as_tuple=True)[0]
    return indices


def indices_to_mask(indices, size):
    mask = torch.zeros(size, dtype=torch.long)
    mask[indices] = 1
    return mask


# @ddi: to slice elegantly. kvcaches are batch,head,token,dim. We want to slice by token (target_dim)
def create_index(ndims, target_dim, index):
    index_obj = [slice(None)] * ndims
    index_obj[target_dim] = index
    return tuple(index_obj)


PositionalEncoder = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]

class CacheBlendImpl:
    def __init__(
        self,
        recompute_ratio: Union[float,list[int]],
        all_reduce_function=None,
    ):
        self.recompute_ratio = recompute_ratio

        # Indexes in the retrieved_kv of the tokens from the fresh_q
        self.indexes_in_kv = torch.tensor([], dtype=torch.long, device="cpu")

        self.positional_encoder: Optional[PositionalEncoder] = None
        self.reverse_positional_encoder: Optional[PositionalEncoder] = None
        self.all_reduce_function = all_reduce_function
        self.xattn_ids = None
        
        # External scores for ensemble selection (diffk + last layer QK)
        self.external_scores: Optional[torch.Tensor] = None

    def set_positional_encoder(self, positional_encoder: PositionalEncoder):
        self.positional_encoder = positional_encoder

    def set_reverse_positional_encoder(
        self, reverse_positional_encoder: PositionalEncoder
    ):
        self.reverse_positional_encoder = reverse_positional_encoder
        
        
    #######################################################################################
    #  Setter for external scores
    def set_external_scores(self, external_scores: Optional[torch.Tensor]):
        """Set external scores (e.g., FusionRAG's QK attention scores) for ensemble selection."""
        self.external_scores = external_scores

    #  Ensemble selection helper
    def _ensemble_select(
        self,
        scores_a: torch.Tensor,  # diffK scores
        scores_b: torch.Tensor,  # external scores (QK attention)
        num_to_select: int,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Alternate between two score rankings to select tokens.
        Returns indices of selected tokens.
        """
        device = scores_a.device
        
        # Mask invalid tokens (set to -inf so they rank last)
        valid_mask_float = valid_mask.to(device).float()
        masked_a = scores_a.clone()
        masked_b = scores_b.clone().to(device)
        
        # Set invalid positions to very negative value
        min_val = torch.finfo(scores_a.dtype).min
        masked_a = masked_a * valid_mask_float + (1 - valid_mask_float) * min_val
        masked_b = masked_b * valid_mask_float + (1 - valid_mask_float) * min_val
        
        # Get rankings (indices sorted by score descending)
        rank_a = torch.argsort(masked_a, descending=True)
        rank_b = torch.argsort(masked_b, descending=True)
        
        selected = set()
        idx_a, idx_b = 0, 0
        
        # Alternate between rankings
        use_a = True
        while len(selected) < num_to_select:
            if use_a:
                # Pick from ranking A (diffK)
                while idx_a < len(rank_a) and rank_a[idx_a].item() in selected:
                    idx_a += 1
                if idx_a < len(rank_a):
                    token_idx = rank_a[idx_a].item()
                    # Only add if valid
                    if valid_mask[token_idx]:
                        selected.add(token_idx)
                    idx_a += 1
            else:
                # Pick from ranking B (external/QK)
                while idx_b < len(rank_b) and rank_b[idx_b].item() in selected:
                    idx_b += 1
                if idx_b < len(rank_b):
                    token_idx = rank_b[idx_b].item()
                    # Only add if valid
                    if valid_mask[token_idx]:
                        selected.add(token_idx)
                    idx_b += 1
            
            use_a = not use_a
            
            # Safety: if both rankings exhausted
            if idx_a >= len(rank_a) and idx_b >= len(rank_b):
                break
        
        return torch.tensor(sorted(selected), dtype=torch.long, device="cpu")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ###################################################################################

    def _select_tokens_single_query(
        self,
        rk: torch.Tensor,
        rv: torch.Tensor,
        valid: torch.Tensor,
        fq: torch.Tensor,
        fk: torch.Tensor,
        fv: torch.Tensor,
        token_dim: int,
    ) -> torch.Tensor:
        """
        Input: retrieved KV, valid_mask, and fresh QKV for a single query
        Output: selected tokens indices
        """
        # We compare the retrieved KVs with the fresh KVs and keep the
        # following tokens:
        #  1. Invalid tokens
        #  2. Token with top difference in the fresh KV, if the token is
        #     valid. Based on previous CacheBlend implementation, we only
        #     use V to compare the difference. The number of tokens to
        #     keep is determined by the `recompute_ratio`
        assert fk.shape == rk.shape
        assert fv.shape == rv.shape
        assert False, "unused"

        # Find the top different tokens
        dims_to_average = [i for i in range(fv.dim()) if i != token_dim]
        diff_per_token = torch.mean((fv - rv) ** 2, dims_to_average)
        diff_per_token = diff_per_token * valid.to(diff_per_token.device)

        num_valid_tokens = valid.sum()
        num_selected_tokens = int(num_valid_tokens * self.recompute_ratio)
        top_indices = torch.topk(diff_per_token, num_selected_tokens).indices
        # logger.debug(f"Indices of the top differences: {top_indices}")

        # Merge the positions with the invalid tokens
        top_mask = indices_to_mask(top_indices, valid.shape[0])
        total_selected_mask = (1 - valid) + top_mask

        local_indices = mask_to_indices(total_selected_mask)
        # logger.debug(f"Local indices of the selected tokens: {local_indices}")
        return local_indices

    def _build_positions(self, query_start_loc: torch.Tensor, device) -> torch.Tensor:
        """Rebuild the positions based on the query start locs"""
        # ret = torch.arange(int(query_start_loc[-1]), device=device)
        ret: torch.Tensor = torch.arange(query_start_loc[-1], device=device)  # type: ignore
        for start, end in zip(query_start_loc[:-1], query_start_loc[1:], strict=False):
            ret[start:end] -= start
        return ret.long()

    # This is for CCRAFT
    def _apply_tokens_all_queries(
        self,
        rk: torch.Tensor,
        rv: torch.Tensor,
        valid: torch.Tensor,
        fq: torch.Tensor,
        fk: torch.Tensor,
        fv: torch.Tensor,
        token_dim: int,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        """
        The selection only happens once.
        Before selection: fk, fv and rk, rv have same dimensionalities. After, not, whence the indexes_in_kv.
        Input: retrieved KV, valid_mask, and fresh QKV for a single query,
        and query_start_loc
        Output: new_query_start_locs
        """
        # Consider TP here.
        # But we cannot couple it with serving engine,
        # so pass a all_reduce_function.

        # We compare the retrieved KVs with the fresh KVs and keep the
        # following tokens:
        #  1. Invalid tokens
        #  2. Token with top difference in the fresh KV, if the token is
        #     valid. Based on previous CacheBlend implementation, we only
        #     use V to compare the difference. The number of tokens to
        #     keep is determined by the `recompute_ratio`
        assert fk.shape == rk.shape
        assert fv.shape == rv.shape
        new_query_start_locs = [0]

        valid = valid.to(dtype=torch.int64)
        assert(len(query_start_loc)==2)
        # NOTE(Sixian): Here I assume valid mask is the same across TPs.
        # As TP runs in lock-step, we should guarantee this in evictor.
        all_indices = [self.indexes_in_kv]
        for qstart, qend in zip(
            query_start_loc[:-1], query_start_loc[1:], strict=False
        ):
            assert qstart == 0, f"{qstart=} is supposed to be 0 with batch size 1"
            # debug: check that the recompute indices are a subset
            mask_indices = valid.nonzero(as_tuple=True)[0].tolist()
            assert  set(self.recompute_ratio).issubset(set(mask_indices)), f"{mask_indices} vs {self.recompute_ratio}"
            
            local_valid = valid[qstart:qend]
            top_indices = self.recompute_ratio
            top_mask = indices_to_mask(top_indices, local_valid.shape[0]).to(local_valid.device)
            #  total_selected_mask = (~local_valid) | top_mask   # should be bette
            total_selected_mask = (1 - local_valid) + top_mask
            local_indices = mask_to_indices(total_selected_mask).to(self.indexes_in_kv.device)
            if isdbg():
                logger.debug(f"{local_valid=}")
                logger.debug(f"{top_indices=}")
                logger.debug(f"{top_mask=}")
                logger.debug(f"{local_indices=}")
            new_query_start_locs.append(new_query_start_locs[-1] + len(local_indices))
            #all_indices.append(local_indices+int(qstart))
            self.indexes_in_kv = torch.cat( (self.indexes_in_kv, local_indices + int(qstart)))  #save locally for later layers
        if isdbg():logger.debug(f"{new_query_start_locs=}")
        #self.indexes_in_kv = torch.cat( all_indices)  #save locally for later layers
        return torch.tensor(
            new_query_start_locs,
            device=query_start_loc.device,
            dtype=query_start_loc.dtype,
        )
    def _select_tokens_all_queries(
        self,
        rk: torch.Tensor,
        rv: torch.Tensor,
        valid: torch.Tensor,
        fq: torch.Tensor,
        fk: torch.Tensor,
        fv: torch.Tensor,
        token_dim: int,
        query_start_loc: torch.Tensor,
        layer:int,
        original_positions:torch.Tensor=None,
        new_positions:torch.Tensor=None
    ) -> torch.Tensor:
        """
        The selection only happens once.
        Before selection: fk, fv and rk, rv have same dimensionalities. After, not, whence the indexes_in_kv.
        Input: retrieved KV, valid_mask, and fresh QKV for a single query,
        and query_start_loc
        Output: new_query_start_locs
        """
        # Consider TP here.
        # But we cannot couple it with serving engine,
        # so pass a all_reduce_function.

        # We compare the retrieved KVs with the fresh KVs and keep the
        # following tokens:
        #  1. Invalid tokens
        #  2. Token with top difference in the fresh KV, if the token is
        #     valid. Based on previous CacheBlend implementation, we only
        #     use V to compare the difference. The number of tokens to
        #     keep is determined by the `recompute_ratio`
        assert fk.shape == rk.shape, f"{layer=} {fk.shape=} {rk.shape=}"
        assert fv.shape == rv.shape, f"{layer=} {fv.shape=} {rv.shape=}"
        new_query_start_locs = [0]
        '''
        fk.shape=torch.Size([8, 6430, 128]) fq.shape=torch.Size([32, 6430, 128])
        '''

        valid = valid.to(dtype=torch.int64)
        diff_k = True
        #k,v have shape [num_kv_heads,L,dim]
        # Find the top different tokens
        dims_to_average = [i for i in range(fv.dim()) if i != token_dim]
        head_dim = 0
        dims_to_average_bis =  [i for i in range(fv.dim()) if i != head_dim]
        if diff_k:
            roped_rk = self.rescale32(rk,original_positions,new_positions)
            diff_per_token = torch.mean((fk - roped_rk) ** 2, dims_to_average)
        else:
            assert False
            roped_rk = rk
            diff_per_token = torch.mean((fv - rv) ** 2, dims_to_average)
        if DUMP_K:
            assert diff_k
            # dims_to_average_bis=[1, 2] dims_to_average=[0, 2] fk.shape=torch.Size([8, 1491, 128]) 
            QID=int(os.environ.get('QID', -1))
            # this is to support fake blending at several layers
            self.indexes_in_kv = torch.tensor([], dtype=torch.long, device="cpu")
            assert QID!=-1
            debug_dir = os.environ.get("CB_DEBUG_DUMP_DIR", "/tmp/cacheblend_maps")
            os.makedirs(debug_dir, exist_ok=True)
            torch.save(diff_per_token.detach().cpu(), f"{debug_dir}/{QID}_{layer}_diff.pt")
            torch.save(fk.detach().cpu(), f"{debug_dir}/{QID}_{layer}_full.pt")
            # different normalization: norm each token separately to unit norm
            # just get avg distance of unit norm, i.e., cosine sim
            nfk = torch.nn.functional.normalize(fk,dim=-1)
            nrk = torch.nn.functional.normalize(roped_rk,dim=-1)
            diff = (nfk-nrk).pow(2)
            diff_per_norm_token = diff.mean(dim=(0,2))
            torch.save(diff_per_norm_token.detach().cpu(), f"{debug_dir}/{QID}_{layer}_ndiff.pt")
            del diff_per_norm_token
        if EXTRA_INFO == 1:
            assert False
            extra_info['value_shape'] = list(fv.shape)
            heads_diviation = torch.mean((fv - rv).abs(), dims_to_average_bis).reshape(-1)
            significance = torch.mean((fv).abs(), dims_to_average_bis).reshape(-1)
            extra_info['normalized_heads_diviation'] = (heads_diviation / significance).tolist()
            extra_info['debug_fv0_mean'] = float(fv[0,:,0].reshape(-1).abs().mean())
            # extra_info['debug_fv1'] = fv[0,:,0].reshape(-1).tolist()
            extra_info['debug_rv0'] = float(rv[0,:,0].reshape(-1).abs().mean())
            # extra_info['debug_rv1'] = rv[0,:,0].reshape(-1).tolist()
        # NOTE(Sixian): Here I assume valid mask is the same across TPs.
        # As TP runs in lock-step, we should guarantee this in evictor.
        diff_per_token = diff_per_token * valid.to(diff_per_token.device)
        if self.all_reduce_function is not None:
            diff_per_token = self.all_reduce_function(diff_per_token)
        for qstart, qend in zip(
            query_start_loc[:-1], query_start_loc[1:], strict=False
        ):
            local_valid = valid[qstart:qend]
            num_valid_tokens = local_valid.sum()
            num_selected_tokens = int(num_valid_tokens * self.recompute_ratio)
            
            ##################################################################3
            #addition
            
            if self.external_scores is not None:
                top_indices = self._ensemble_select(
                    diff_per_token[qstart:qend],
                    self.external_scores[qstart:qend],
                    num_selected_tokens,
                    local_valid
                )
            else:
                #the original behavior
                top_indices = torch.topk(
                    diff_per_token[qstart:qend], num_selected_tokens
                ).indices
            
            #####################################################
            
            
            if self.recompute_ratio == 1.0:
                assert len(top_indices) == num_valid_tokens
            
            top_mask = indices_to_mask(top_indices, local_valid.shape[0]).to(local_valid.device)
            total_selected_mask = (1 - local_valid) + top_mask
            local_indices = mask_to_indices(total_selected_mask).to(self.indexes_in_kv.device)
            new_query_start_locs.append(new_query_start_locs[-1] + len(local_indices))
            self.indexes_in_kv = torch.cat(
                (self.indexes_in_kv, local_indices + int(qstart))
            )
            
        del diff_per_token
        torch.npu.empty_cache()
        if self.use_anti_piaffe:
            xattn = self.aggregate_attention_topk(fq,fk,top_indices,K=len(top_indices))
            if xattn is not None:print(f"{xattn.shape=}")
        else:
            xattn = None
        return torch.tensor(
            new_query_start_locs,
            device=query_start_loc.device,
            dtype=query_start_loc.dtype,
        ),  xattn
        

    def aggregate_attention_topk(self,fq, fk, L,  K=10):
        """
        fq: [n_q_heads, n_tokens, d]
        fk: [n_k_heads, n_tokens, d]
        L: list[int] tokens whose outgoing attention we track
        boundaries: list of (start, end) tuples for intra-block masking
        K: number of top receivers to return
        """
        
        # TODO: are fq,fk in correct order? Where's the mask:
        # Compute mean head representations
        q_mean = fq.mean(0)   # [n_tokens, d]
        k_mean = fk.mean(0)   # [n_tokens, d]
        boundaries = self.chunk_boundaries
        dev = q_mean.device
        sink = torch.arange(0,boundaries[0][1]).to(dev)
        print("NOTE WE ARE FORCING K == ALL TO CHECK SIMILAR RESULT")
        print("NOTE: >> query for sure attends strongly to last tokens of last doc. should we pay stronger attentipn to it")
        # Note: tokens in the prefix are exact so we will never recomp them
        # ALso tokens in the last chunk cannot be recomp
        last_tkns = [x for x in range(boundaries[-1][0],boundaries[-1][1]) if x not in L]
        total = boundaries[-1][1]
        K = total-len(L)-len(last_tkns)-len(sink)   # this forces ALL
        if K <=0:return None

        # Compute aggregate attention matrix
        # scores[i,j] = attention from token i (query) to token j (key)
        attn = torch.matmul(q_mean, k_mean.T) / fq.shape[-1]**0.5   # [n_tokens, n_tokens]

        # causal mask
        np.set_printoptions(threshold=np.inf, linewidth=200)
        mask = torch.tril(torch.ones_like(attn, dtype=torch.bool))
        # PROBLEM: if you completely zero out a line the softmax for that line is ill-defined
        # this happens for sure to the sink but can also happen to other blocks if it happens
        # that chunk starts with masked out
        # solution is to compute the normal attention matrix without masking anything
        # THEN we mask to remove the contributes we do nto care about
        
        min_dtype = torch.finfo(attn.dtype).min
        attn = attn.masked_fill(~mask, min_dtype)
        attn = attn - attn.max(dim=-1, keepdim=True).values
        attn = torch.softmax(attn, dim=-1)  # normalize over keys
        
        # Force initial tokens to zero  
        attn[:,0:boundaries[0][1]] = 0
        # Force attention to L tokens to 0
        attn[:,L] = False
        # Mask out intra-block attention
        for start, end in boundaries:
            attn[start:end, start:end] = False

        # Aggregate attention sent by tokens in L
        # Sum attention weights emitted by those tokens
        attn_from_L = attn[L, :]          # [len(L), n_tokens]
        received_scores = attn_from_L.sum(0)  # [n_tokens]

        # Exclude self-tokens in L if desired
        # received_scores[L] = 0.0

        # Top-K receivers
        topk_vals, topk_idx = torch.topk(received_scores, K)
        row_sums = attn.sum(dim=1)
        assert not torch.any(torch.isin(topk_idx, sink)), f"Tensors are not disjoint!\n{topk_idx=}\n{topk_vals}\n{sink=}\n{K=} {len(L)=} {boundaries[0][1]=} {mask.shape=}\n{[(i,r.item()) for i,r in enumerate(row_sums)]}"
        assert not torch.any(torch.isin(sink, topk_idx)), "Tensors are not disjoint!"

        # add the sink
        return torch.cat([sink,topk_idx])


    #@ ddi this rescale only rescales the K. 
    def rescale(self,
                retrieved_k: torch.Tensor,
                original_positions: torch.Tensor,
                positions: torch.Tensor,
                ):
        dumb_q = torch.zeros_like(retrieved_k)
        dumb_q, rk_no_position = self.reverse_positional_encoder(
            original_positions.to(device=retrieved_k.device, dtype=torch.long),
            dumb_q,
            retrieved_k,
        )
        dumb_q, rk_with_position = self.positional_encoder(
            positions.to(device=retrieved_k.device, dtype=torch.long), dumb_q, rk_no_position
        )
        if False:
            ko,kn=retrieved_k.squeeze(0),rk_with_position.squeeze(0)
            print(f"Rescaling {original_positions=} to {positions=} {ko.shape=}")
            for h,h1 in zip(ko,kn):
                for t,t1 in zip(h,h1):
                    err=torch.norm(t-t1,p=2).item()
                    print(err,end=" ")
                print("\n\n")
            assert torch.allclose(retrieved_k,rk_with_position,rtol=1e-5,atol=1e-8) 
        return rk_with_position
    
    def rescale32(self,
                retrieved_k: torch.Tensor,
                original_positions: torch.Tensor,
                positions: torch.Tensor,
                ):
        rk32 = retrieved_k.to(torch.float32)
        dumb_q = torch.zeros_like(rk32)
        dumb_q, rk_no_position = self.reverse_positional_encoder(
            original_positions.to(device=retrieved_k.device, dtype=torch.long),
            dumb_q, rk32
        )
        dumb_q, rk_with_position = self.positional_encoder(
            positions.to(device=retrieved_k.device, dtype=torch.long), dumb_q, rk_no_position
        )
        ko,kn=retrieved_k.squeeze(0),rk_with_position.squeeze(0)
        if False:
            print(f"Rescaling {original_positions=} to {positions=} {ko.shape=}")
            for h,h1 in zip(ko,kn):
                for t,t1 in zip(h,h1):
                    err=torch.norm(t-t1,p=2).item()
                    print(err,end=" ")
                print("\n\n")
            assert torch.allclose(rk32,rk_with_position,rtol=1e-5,atol=1e-8) 
        return rk_with_position.to(torch.bfloat16)

    def blend(
        self,
        layer_id: int,
        retrieved_k: torch.Tensor,
        retrieved_v: torch.Tensor,
        valid_mask: torch.Tensor,
        original_positions: torch.Tensor,
        fresh_q: torch.Tensor,
        fresh_k: torch.Tensor,
        fresh_v: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        token_dim: int,
    ) -> BlendOutput:
        """This function blends the retrieved KV with fresh KVs, and
        returns the short Q + long KV (blended) + positions of the tokens in Q

        :param int layer_id: The layer id
        :param torch.Tensor retrieved_k: The retrieved K layer, in shape
            [num_tokens, hidden_dims]
        :param torch.Tensor retrieved_v: The retrieved V layer, in shape
            [num_tokens, hidden_dims]
        :param torch.Tensor valid_mask: A CPU tensor returned from the
            retriever indicating whether the KV is valid.
        :param torch.Tensor original_positions: The original positions of the
            tokens in the retrieved KV
        :param torch.Tensor fresh_q: The fresh Q tensor from QKV split,
            in shape [num_tokens, hidden_dims]
        :param torch.Tensor fresh_k: The fresh K tensor from QKV split,
            in shape [num_tokens, hidden_dims]
        :param torch.Tensor fresh_v: The fresh V tensoy from QKV split,
            in shape [num_tokens, hidden_dims]
        :param torch.Tensor positions: The positions in the input of the
            tokens in the fresh_q
        :param torch.Tensor query_start_loc: The start location of the query if
            input_tokens has multiple requests in a batch. The length should be
            the number of requests in the batch + 1. Note this will NOT be
            changed after token selection. (+1 because it starts by 0)
        :param int token_dim: The token dimension

        :return: The blended Q, K, V, and positions
        """
        # We should convert the shape of KV to [num_elems, hidden_dimensions]
        # assert valid_mask.is_cpu, "valid_mask should be on CPU"
        if DUMP_K == 1:
            do_blend = True
            assert self.recompute_ratio == 1.0
        else:
            do_blend = (layer_id == 1)
        if layer_id == 0:
            return BlendOutput(
                fresh_q,
                fresh_k,
                fresh_v,
                positions,
                torch.arange(fresh_q.shape[token_dim], device="cpu", dtype=torch.long),
                query_start_loc=None,
                xattn_ids=self.xattn_ids
            )

        elif do_blend:
            if isinstance(self.recompute_ratio,list):
                logger.debug(f"@ddi blending l1 at fixed positions {self.recompute_ratio=}")
                query_start_locs_tensor = self._apply_tokens_all_queries(
                    retrieved_k,
                    retrieved_v,
                    valid_mask,
                    fresh_q,
                    fresh_k,
                    fresh_v,
                    token_dim,
                    query_start_loc,
                )
            else:
                logger.debug("@ddi blending l1 with tx cb")
                query_start_locs_tensor,xattn_ids = self._select_tokens_all_queries(
                    retrieved_k,
                    retrieved_v,
                    valid_mask,
                    fresh_q,
                    fresh_k,
                    fresh_v,
                    token_dim,
                    query_start_loc,
                    layer_id,
                    original_positions,
                    positions
                )
                self.xattn_ids = xattn_ids
            if isdbg():logger.debug(f"{self.indexes_in_kv=}")
            index_obj = create_index(fresh_k.dim(), token_dim, self.indexes_in_kv)
            new_q = fresh_q[index_obj]
            new_positions = positions[self.indexes_in_kv]
            if isdbg():
                logger.debug(
                    f"Selected {len(self.indexes_in_kv)} tokens out of "
                    f"{retrieved_k.shape} tokens to blend. "
                    f"{new_positions=}"
                )
            return BlendOutput(
                new_q,
                fresh_k,
                fresh_v,
                new_positions,
                self.indexes_in_kv,
                query_start_locs_tensor,
                self.xattn_ids
            )

        else:
            assert len(self.indexes_in_kv) == fresh_k.shape[token_dim]
            index_obj = create_index(fresh_k.dim(), token_dim, self.indexes_in_kv) # [None..., indexes, ...None]
            if ( self.positional_encoder is not None and self.reverse_positional_encoder is not None):
                logger.debug(f"{layer_id=} Rescaling  {original_positions=} -> {positions=}. {retrieved_k.shape}")
                if ROPE32:
                    rk_with_position = self.rescale32(retrieved_k, original_positions, positions)
                else:
                    rk_with_position = self.rescale(retrieved_k, original_positions, positions)
            else:
                logger.warning(
                    "Positional encoder and reverse positional "
                    "encoder is not set. This may lead to "
                    "incorrect results."
                )
                rk_with_position = retrieved_k

            rk_with_position[index_obj] = fresh_k
            retrieved_v[index_obj] = fresh_v

            return BlendOutput(
                fresh_q,
                rk_with_position,
                retrieved_v,
                positions,
                self.indexes_in_kv,
                None,
                xattn_ids=self.xattn_ids
            )
