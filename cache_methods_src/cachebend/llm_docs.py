import torch
import numpy as np
import os
from torch import nn
import torch_npu
from typing import Optional, TYPE_CHECKING,Union, List
from cachebend.cacheblend import BlendOutput, CacheBlendImpl
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from functools import partial
from cachebend.utils import Timer
from transformers.utils import logging
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2Attention, Qwen2Model, Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Attention
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention, LlamaModel, LlamaForCausalLM
from transformers import PreTrainedTokenizer, AutoModelForCausalLM
import gc
from cachebend.utils import get_mem,report_npu_tensors2



########################################################################

USE_NPU_FUSED_ATTENTION = os.getenv("CB_USE_NPU_FUSED_ATTENTION", "1").strip().lower() in {
    "1", "true", "yes", "y", "on"
}

from transformers import DynamicCache
import time
from typing import Union, Optional, List, Dict, Tuple



######################################################################

logger = logging.get_logger("cachebend")
logger.setLevel(logging.INFO)


def _cache_num_layers(cache: DynamicCache) -> int:
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return len(cache)


def _cache_layer_kv(cache: DynamicCache, layer_idx: int):
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        if hasattr(layer, "keys") and hasattr(layer, "values"):
            return layer.keys, layer.values
    layer = cache[layer_idx]
    if isinstance(layer, (tuple, list)) and len(layer) >= 2:
        return layer[0], layer[1]
    raise TypeError(f"Unsupported cache layer format at index {layer_idx}: {type(layer)!r}")

def proj(self_attn,normalized_hidden_states,hidden_shape):
    fresh_q = self_attn.q_proj(normalized_hidden_states).view(hidden_shape)
    fresh_k = self_attn.k_proj(normalized_hidden_states).view(hidden_shape)
    fresh_v = self_attn.v_proj(normalized_hidden_states).view(hidden_shape)
    if isinstance(self_attn,Qwen3Attention):
        fresh_q = self_attn.q_norm(fresh_q)
        fresh_k = self_attn.k_norm(fresh_k)
    fresh_q = fresh_q.transpose(1, 2)
    fresh_k = fresh_k.transpose(1, 2)
    fresh_v = fresh_v.transpose(1, 2)
    return fresh_q,fresh_k,fresh_v

def isdbg():
    return logger.isEnabledFor(logging.DEBUG)
inplace_rope=True

if TYPE_CHECKING:
    # CausalLM = Qwen2ForCausalLM
    CausalLM = LlamaForCausalLM
    # Model = Qwen2Model
    Model = LlamaModel
    # DecoderLayer = Qwen2DecoderLayer
    DecoderLayer = LlamaDecoderLayer
    # Attention = Qwen2Attention
    Attention = LlamaAttention

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_partial(x, cos, sin, flag=1):
    """Apply RoPE to x, handling partial rotary embeddings.

    When rotary_dim < head_dim (e.g. Phi-4 with partial_rotary_factor=0.75),
    only the first rotary_dim dimensions are rotated; the rest pass through.
    When rotary_dim == head_dim (Llama, Mistral, Qwen), this is equivalent
    to the standard rotate_half path.
    """
    rotary_dim = cos.shape[-1]
    head_dim = x.shape[-1]
    if rotary_dim < head_dim:
        # Split into rotary and non-rotary parts
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_rot_out = (x_rot * cos) + flag * (rotate_half(x_rot) * sin)
        return torch.cat((x_rot_out, x_pass), dim=-1)
    else:
        return (x * cos) + flag * (rotate_half(x) * sin)


def apply_rotary_pos_emb_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim=1,
    flag: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RoPE function for Ascend NPU. Supports partial rotary embeddings.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = _apply_rope_partial(q, cos, sin, flag)
    k_embed = _apply_rope_partial(k, cos, sin, flag)

    return q_embed, k_embed


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim=1,
    flag: int = 1,
):
    """Applies Rotary Position Embedding to the query and key tensors.

    Supports partial rotary embeddings (e.g. Phi-4 with partial_rotary_factor < 1.0).
    When cos/sin have fewer dimensions than q/k, only the first rotary_dim dimensions
    are rotated and the rest pass through unchanged.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = _apply_rope_partial(q, cos, sin, flag)
    k_embed = _apply_rope_partial(k, cos, sin, flag)
    return q_embed, k_embed


def positional_encoder(model: "Model", position_ids: torch.LongTensor, q: torch.Tensor, k: torch.Tensor):
    cos, sin = model.rotary_emb(q, position_ids.unsqueeze(0))
    q_shape = q.shape
    k_shape = k.shape
    if inplace_rope:
        q, k = apply_rotary_pos_emb_inplace(q, k, cos, sin)
    else:
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q.reshape(q_shape), k.reshape(k_shape)

def reverse_positional_encoder(model: "Model", position_ids: torch.LongTensor, q: torch.Tensor, k: torch.Tensor):
    cos, sin = model.rotary_emb(q, position_ids.unsqueeze(0))
    q_shape = q.shape
    k_shape = k.shape
    if inplace_rope:
        q,k = apply_rotary_pos_emb_inplace(q,k,cos,sin,flag=-1)
    else:
        q, k = apply_rotary_pos_emb(q, k, cos, sin, flag=-1)
    return q.reshape(q_shape), k.reshape(k_shape)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
def eager_attention_forward_fused(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        **kwargs,
        ):
    # Repeat key and value states for Grouped-Query Attention (GQA)
    key_states = repeat_kv(key, module.num_key_value_groups) # repeat on the level of heads
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_output = _sdpa_attention_maybe_chunked(
        query,
        key_states,
        value_states,
        attention_mask,
        is_causal=False,
    )

    # The output is already in the correct shape, but we transpose and contiguous
    # for compatibility with a typical attention block's output.
    attn_output = attn_output.transpose(1, 2).contiguous()
    
    return attn_output


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _explicit_mask_attention_chunk_size(query: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> int:
    """Return query-axis chunk size for large explicit-mask attention calls.

    CacheBlend partial recompute can select thousands of query positions while
    attending over a long cached KV sequence. A single explicit-mask SDPA call
    may allocate heads * selected_tokens * total_tokens workspace. Chunking on
    the query axis preserves the exact attention result and bounds peak memory.
    """
    if attention_mask is None:
        return 0
    if not _env_flag("CB_ENABLE_CHUNKED_SDPA", default=False):
        return 0

    raw = os.getenv("CB_ATTN_CHUNK_SIZE")
    if raw is not None:
        chunk_size = int(raw)
        return max(0, chunk_size)

    return int(os.getenv("CB_ATTN_AUTO_CHUNK_SIZE", "1024"))


def _raise_with_chunked_sdpa_hint(exc: RuntimeError, query: torch.Tensor, key: torch.Tensor) -> None:
    msg = str(exc).lower()
    if "out of memory" not in msg and "oom" not in msg:
        raise exc

    if _env_flag("CB_ENABLE_CHUNKED_SDPA", default=False):
        print(
            "CacheBlend explicit-mask SDPA ran out of memory even with "
            "CB_ENABLE_CHUNKED_SDPA=1. Try a smaller CB_ATTN_CHUNK_SIZE, "
            "for example CB_ATTN_CHUNK_SIZE=512 or 256.",
            flush=True,
        )
    else:
        print(
            "CacheBlend explicit-mask SDPA ran out of memory. This can happen "
            f"when recomputing {query.shape[-2]} selected tokens over "
            f"{key.shape[-2]} cached tokens. Re-run with "
            "CB_ENABLE_CHUNKED_SDPA=1, optionally with CB_ATTN_CHUNK_SIZE=512.",
            flush=True,
        )
    raise exc


def _sdpa_attention_maybe_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    is_causal: bool,
) -> torch.Tensor:
    chunk_size = _explicit_mask_attention_chunk_size(query, attention_mask)
    q_len = query.shape[-2]
    if chunk_size <= 0 or q_len <= chunk_size:
        try:
            return torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )
        except RuntimeError as exc:
            _raise_with_chunked_sdpa_hint(exc, query, key)

    outs = []
    for start in range(0, q_len, chunk_size):
        end = min(start + chunk_size, q_len)
        mask_slice = attention_mask[..., start:end, :] if attention_mask is not None else None
        query_slice = query[..., start:end, :]
        try:
            outs.append(
                torch.nn.functional.scaled_dot_product_attention(
                    query_slice,
                    key,
                    value,
                    attn_mask=mask_slice,
                    dropout_p=0.0,
                    is_causal=is_causal,
                )
            )
        except RuntimeError as exc:
            _raise_with_chunked_sdpa_hint(exc, query_slice, key)
    return torch.cat(outs, dim=-2)


def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        **kwargs,
        ):
    key_states = repeat_kv(key, module.num_key_value_groups) # repeat on the level of heads
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # NOTE: we could do the matmul in fp32 and do the  "to" cast just at the nd
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


def npu_fused_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        num_heads: int,
        num_kv_heads: int,
        scaling: float,
) -> torch.Tensor:
    """Memory-efficient attention using Ascend NPU FlashAttention kernel.

    Inputs in BNSD format: (batch, num_heads, seq_len, head_dim).
    The NPU kernel handles GQA natively (no need for repeat_kv).
    """
    # Convert mask: the NPU kernel expects a bool mask where True = MASKED (blocked),
    # while PyTorch SDPA uses additive masks with -inf for masked positions.
    npu_mask = None
    if attention_mask is not None:
        # attention_mask has large negative values for masked positions
        npu_mask = (attention_mask < -1.0).squeeze(0).squeeze(0)  # (Q_S, KV_S) bool
        if npu_mask.ndim == 2:
            npu_mask = npu_mask.unsqueeze(0)  # (1, Q_S, KV_S)
        npu_mask = npu_mask.contiguous()

    # This function is only called during blending, which always provides a mask.
    # sparse_mode=0 means "use the explicit atten_mask".
    sparse_mode = 0

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    common_kwargs = dict(
        num_heads=num_heads,
        input_layout="BNSD",
        num_key_value_heads=num_kv_heads,
        atten_mask=npu_mask,
        sparse_mode=sparse_mode,
    )
    from torch_npu import npu_fused_infer_attention_score as _npu_fa
    result = _npu_fa(
        query=query,
        key=key,
        value=value,
        scale=scaling,
        softmax_lse_flag=False,
        **common_kwargs,
    )
    attn_output = result[0]
    # Output is BNSD, transpose to BSND for downstream
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


def attention_forward(
        attn: "Attention",
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        query_states: Optional[torch.Tensor] = None,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
        ) -> torch.Tensor:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    # this is done only the first time: after this, the values are prepared by  preceding layer
    if query_states is None or key_states is None or value_states is None:
        
        query_states,key_states,value_states=proj(attn,hidden_states,hidden_shape)
            
        cos, sin = position_embeddings
        if not inplace_rope:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            query_states, key_states = apply_rotary_pos_emb_inplace(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, attn.layer_idx, cache_kwargs)

    num_q_heads = kwargs.get("num_q_heads", getattr(attn, "num_heads", query_states.shape[1]))
    num_kv_heads = kwargs.get("num_kv_heads", getattr(attn, "num_key_value_heads", key_states.shape[1]))

    # Prefer NPU fused FlashAttention kernel (memory-efficient, handles GQA natively)
    # NPU fused attention is only used during blending (always has an explicit mask)
    can_try_npu_fused = (
        USE_NPU_FUSED_ATTENTION
        and attention_mask is not None
    )
    if can_try_npu_fused:
        try:
            attn_output = npu_fused_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                num_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                scaling=attn.scaling,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "oom" in msg:
                raise
            logger.warning(f"NPU fused attention failed ({exc}), falling back to SDPA")
            attn_output = None
        except Exception as exc:
            logger.warning(f"NPU fused attention failed ({exc}), falling back to SDPA")
            attn_output = None

        # If NPU fused succeeded, skip to reshape+o_proj below
        if attn_output is not None:
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn.o_proj(attn_output)
            return attn_output

    # Fallback: SDPA or eager
    use_sdpa = os.getenv("CB_USE_SDPA", "1") == "1"
    if use_sdpa:
        try:
            if attention_mask is None:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                )
            else:
                # Use explicit mask; SDPA requires is_causal=False when attn_mask is provided.
                attn_output = _sdpa_attention_maybe_chunked(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    is_causal=False,
                )
            attn_output = attn_output.transpose(1, 2).contiguous()
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "oom" in msg:
                raise
            # Fall back to eager path for non-OOM runtime issues.
            attn_output = eager_attention_forward_fused(
                attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                scaling=attn.scaling,
                **kwargs,
            )
        except Exception:
            # Fall back to eager path (will be slower/more memory)
            attn_output = eager_attention_forward_fused(
                attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                scaling=attn.scaling,
                **kwargs,
            )
    else:
        attn_output = eager_attention_forward_fused(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=attn.scaling,
            **kwargs,
        )
    #print("post_attn ",get_mem(7))
    #report_npu_tensors2()

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn.o_proj(attn_output)
    return attn_output


def decoder_forward(decoder_layer: "DecoderLayer",
        hidden_states: torch.Tensor,
        query_states: Optional[torch.Tensor] = None,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs):
        residual = hidden_states
        if query_states is None or key_states is None or value_states is None:
            hidden_states = decoder_layer.input_layernorm(hidden_states)

        # print(f"{decoder_layer.self_attn.num_key_value_groups=}")
        # Self Attention
        attn_out = attention_forward(decoder_layer.self_attn,
                hidden_states=hidden_states,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
                )
        #hidden_states = residual + hidden_states
        hidden_states=attn_out.add_(residual)

        # Fully Connected
        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        mlp_out = decoder_layer.mlp(hidden_states)
        #hidden_states = residual + hidden_states
        hidden_states=mlp_out.add_(residual)
        return hidden_states

def update_cache_database(
        model: "Model",
        cache_database: dict[int, tuple[torch.LongTensor, torch.Tensor, torch.Tensor]],
        doc_ids: torch.LongTensor
        ):
    past_key_value = DynamicCache()
    model.forward(input_ids=doc_ids, past_key_values=past_key_value, use_cache=True)
    num_layers = _cache_num_layers(past_key_value)
    for idx in range(doc_ids.shape[0]):
        ids = doc_ids[idx]
        h = hash(tuple(ids.tolist()))
        key_layers = []
        value_layers = []
        for layer_idx in range(num_layers):
            k_layer, v_layer = _cache_layer_kv(past_key_value, layer_idx)
            key_layers.append(k_layer[idx])
            value_layers.append(v_layer[idx])
        cache_database[h] = (ids, key_layers, value_layers)
        print(f"{h=} --> {len(ids)=}")
        
        
##########################################################################################
# Helper to fetch a specific layer from a source (Tensor or an object with get_layer method)
def _fetch_layer_tensor(source, layer_idx: int, kv_idx: int, device: torch.device):
    """
    Fetch a single layer's tensor.
    source: either a List[Tensor], a Tensor (layers, ...), or an object that has get_layer().
    Legend: 0 for Keys, 1 for Values (onlyused if source is an object).
    """
    if hasattr(source, "get_layer"):
        # LbL loading from ChunkWithState
        return source.get_layer(layer_idx, kv_idx, device)
    
    
    if isinstance(source, list):
        t = source[layer_idx]
    else:
        t = source[layer_idx]
    
    return t.to(device, non_blocking=t.is_pinned())


#i think this version is better, safer
def seq_build_blent_cache(
        model: "Model",
        cache_database: list, # a list of (h, tokens, k_source, v_source, pos)
        doc_ids: torch.LongTensor,
        recompute_ratio: Union[float, dict[int, list[int]]],
        past_key_value: Optional[DynamicCache] = None,
        stats = None,
        **kwargs,
) -> BaseModelOutputWithPast:
    
    """
    loads layer one by one
    """
    
    if stats is None: stats = {}
    
    #setup
    assert doc_ids.shape[0] == 1, "Only batch size = 1 is supported"
    doc_embeds = model.embed_tokens(doc_ids)

    if past_key_value is None:
        past_key_value = DynamicCache()

    past_seen_tokens = past_key_value.get_seq_length()
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + doc_embeds.shape[1], 
        device=doc_ids.device, dtype=torch.long
    )
    
    hidden_states = doc_embeds
    position_embeddings = model.rotary_emb(hidden_states, cache_position.unsqueeze(0))

    #Initialize blender, metadata ---
    blender = CacheBlendImpl(recompute_ratio=recompute_ratio)
    blender.set_positional_encoder(partial(positional_encoder, model))
    blender.set_reverse_positional_encoder(partial(reverse_positional_encoder, model))

    # metadata tensors
    valid_mask = torch.zeros_like(doc_ids, dtype=torch.bool, device="cpu")
    original_positions = torch.zeros_like(doc_ids, dtype=torch.int64)
    positions = torch.tensor([list(range(doc_ids.shape[-1])) for _ in range(doc_ids.shape[0])], device=doc_ids.device)
    query_start_loc = torch.tensor([0, doc_ids.shape[-1]], device=doc_ids.device)

    tkn_to_chunk = []
    chunk_boundaries = []
    
    # We store metadata, are then used to fetch tensors later in the loop
    # Structure is:  list of (batch_idx, start_col, end_col, k_source, v_source)
    layer_fetch_plan = [] 

    #Metadata Pass 
    with Timer("scan cache metadata", stats=stats):
        doc_idx = 0
        batch_idx = 0
        cid = 0
        
        for (h, cached_doc_ids, k_source, v_source, orig_pos) in cache_database:
            cached_n = cached_doc_ids.shape[0]
            
            # Update masks
            assert not valid_mask[batch_idx, doc_idx:doc_idx+cached_n].any()
            valid_mask[batch_idx, doc_idx:doc_idx+cached_n] = True
            
            ##update positions
            if orig_pos is None:
                original_positions[batch_idx, doc_idx:doc_idx+cached_n] = torch.arange(cached_n)
            else:
                original_positions[batch_idx, doc_idx:doc_idx+cached_n] = orig_pos.unsqueeze(0)

           
            tkn_to_chunk += [cid] * cached_n
            chunk_boundaries.append((doc_idx, doc_idx+cached_n))
            
            if isinstance(recompute_ratio, dict) and h in recompute_ratio:
                rcmp = [x + doc_idx for x in recompute_ratio[h]]
                recompute_ratio[h] = rcmp
            
            # Add to plan for layer-wise fetching
            layer_fetch_plan.append({
                "batch_idx": batch_idx,
                "start": doc_idx,
                "end": doc_idx + cached_n,
                "k_source": k_source,
                "v_source": v_source
            })
            
            doc_idx += cached_n
            cid += 1

    # finnalize metadata
    if isinstance(recompute_ratio, dict):
        blender.recompute_ratio = [x for k in sorted(recompute_ratio.keys()) for x in recompute_ratio[k]]
    
    valid_mask = valid_mask.to(doc_ids.device, non_blocking=True)
    
    #Decoding Loop (layer by layer loading) ---
    
   
    fresh_q, fresh_k, fresh_v = None, None, None
    local_indices = None
    selected_idx = cache_position
    deselected_idx = None
    xattn_ids = None
    
    # Config setup
    use_anti_piaffe = kwargs.get("use_piaffe", False)
    blender.use_anti_piaffe = use_anti_piaffe
    if use_anti_piaffe:
        blender.chunk_boundaries = chunk_boundaries

    config = model.config
    n_q_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_q_heads)
    head_dim = config.hidden_size // n_q_heads
    kwargs['num_q_heads'] = n_q_heads
    kwargs['num_kv_heads'] = n_kv_heads
    kwargs['head_dim'] = head_dim

    
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    def _make_causal_mask(input_tensor, *args, **kwargs):
        # Quick stub if not imported from original script, assuming standard HF behavior
        # In real integration, ensure this import exists or is passed
        return model.model._update_causal_mask(kwargs.get('attention_mask', None), input_tensor, kwargs.get('cache_position'), kwargs.get('past_key_value'), False)

    # Iterate over  layers
    for layer_id, decoder_layer in enumerate(model.layers[: config.num_hidden_layers]):
        
        # A. Standard Forward Pass (Computation)
        selected_pos_embeds = position_embeddings if local_indices is None else (position_embeddings[0][:, local_indices], position_embeddings[1][:, local_indices])
        selected_doc_embeds = doc_embeds if local_indices is None else doc_embeds[:, local_indices]
        
        with Timer(f"Decoding_L{layer_id}", stats=stats):
            
            # Using a simplified call assuming Llama 
            causal_mask = model._update_causal_mask(
                None, selected_doc_embeds, selected_idx, past_key_value, False 
            )
            
            layer_output = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=selected_idx.unsqueeze(0),
                past_key_value=past_key_value,
                position_embeddings=selected_pos_embeds,
                use_cache=True,
                **kwargs,
            )
            
            # Extract output. HF returns tuple (hidden_states, past_kv)
            hidden_states = layer_output[0]

        # Projection & Blending (next layer)
        if layer_id + 1 < config.num_hidden_layers:
            with Timer(f"projections_L{layer_id}", stats=stats):
                next_decoder_layer = model.layers[layer_id + 1]
                normalized_hidden_states = next_decoder_layer.input_layernorm(hidden_states)
                
                # Manual QKV projection to perform Rope manually before next layer
                #this specific to Llama/Mistral architecture, don't know about Qwen
                #nedd to ask tianx about this
                q_proj = next_decoder_layer.self_attn.q_proj(normalized_hidden_states)
                k_proj = next_decoder_layer.self_attn.k_proj(normalized_hidden_states)
                v_proj = next_decoder_layer.self_attn.v_proj(normalized_hidden_states)
                
                bsz, q_len, _ = hidden_states.shape
                fresh_q = q_proj.view(bsz, q_len, n_q_heads, head_dim).transpose(1, 2)
                fresh_k = k_proj.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
                fresh_v = v_proj.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
                
                cos, sin = selected_pos_embeds
                fresh_q, fresh_k = apply_rotary_pos_emb(fresh_q, fresh_k, cos, sin)

            # Fetch cached ttensors for this layer
            with Timer(f"retrieve_L{layer_id+1}", stats=stats):
                # Allocate memory only for kust one layer
                # shapes are based on first  fetch
                #  construct the composite tensor for the current layer from parts
                
               
                ref_k = _fetch_layer_tensor(layer_fetch_plan[0]["k_source"], layer_id + 1, 0, doc_ids.device)
                
                current_layer_retrieved_k = torch.empty(
                    (doc_ids.shape[0], ref_k.shape[-3], doc_ids.shape[1], ref_k.shape[-1]),
                    device=doc_ids.device, dtype=model.dtype
                )
                current_layer_retrieved_v = torch.empty_like(current_layer_retrieved_k)
                
                # Fill the buffer
                for item in layer_fetch_plan:
                    k_chunk = _fetch_layer_tensor(item["k_source"], layer_id + 1, 0, doc_ids.device)
                    v_chunk = _fetch_layer_tensor(item["v_source"], layer_id + 1, 1, doc_ids.device)
                    
                    current_layer_retrieved_k[:, :, item["start"]:item["end"], :] = k_chunk
                    current_layer_retrieved_v[:, :, item["start"]:item["end"], :] = v_chunk

            # Blending
            with Timer(f"Blending_L{layer_id}", stats=stats):
                ret: BlendOutput = blender.blend(
                    layer_id=layer_id + 1,
                    retrieved_k=current_layer_retrieved_k.squeeze(0),
                    retrieved_v=current_layer_retrieved_v.squeeze(0),
                    valid_mask=valid_mask.squeeze(0),
                    original_positions=original_positions.squeeze(0),
                    fresh_q=fresh_q.squeeze(0),
                    fresh_k=fresh_k.squeeze(0),
                    fresh_v=fresh_v.squeeze(0),
                    positions=positions.squeeze(0),
                    query_start_loc=query_start_loc,
                    token_dim=1,
                )
                stats["recomp_ids"] = len(blender.indexes_in_kv)

            # pdate indices  for next itreation
            local_indices = ret.local_indices
            
            if ret.query_start_loc is not None:
                hidden_states = hidden_states[:, local_indices]
                # Logic to update Deselected Indices for cache update
                selected_idx = cache_position[local_indices]
                valid_mask_tmp = valid_mask.clone().detach().reshape(-1)
                valid_mask_tmp[selected_idx] = False
                deselected_idx = torch.where(valid_mask_tmp)[0]
                
                # Handle Chunk ID Mapping updates for Piaffe if needed
                tkn_to_chunk_prime = [x for x in tkn_to_chunk if x in selected_idx]
                tkn_to_chunk = [x for x in tkn_to_chunk if x not in selected_idx]
                tkn_to_chunk += tkn_to_chunk_prime

            #update Fresh QKV with results
            fresh_q = ret.q.unsqueeze(0)
            
            if deselected_idx is not None:
                #Mixed Reuse
                fresh_k = ret.k[:, ret.local_indices, :].unsqueeze(0)
                fresh_v = ret.v[:, ret.local_indices, :].unsqueeze(0)
                to_cache_k = ret.k[:, deselected_idx, :].unsqueeze(0)
                to_cache_v = ret.v[:, deselected_idx, :].unsqueeze(0)
                past_key_value.update(to_cache_k, to_cache_v, layer_id + 1, {})
            else:
                #No Reuse (or Full Recompute logic)
                fresh_k = ret.k.unsqueeze(0)
                fresh_v = ret.v.unsqueeze(0)

            #clenup memory
            del current_layer_retrieved_k
            del current_layer_retrieved_v
            
            #Handle default case where local_indices is empty, that is 100% reuse
            if local_indices.numel() == 0:
                # Fast-forward filling cache for remaining layers
                for layer_id_t in range(layer_id + 2, config.num_hidden_layers):
                    # Fetch 
                    k_next = torch.empty((doc_ids.shape[0], ref_k.shape[-3], doc_ids.shape[1], ref_k.shape[-1]), device=doc_ids.device, dtype=model.dtype)
                    v_next = torch.empty_like(k_next)
                    
                    for item in layer_fetch_plan:
                        k_chunk = _fetch_layer_tensor(item["k_source"], layer_id_t, 0, doc_ids.device)
                        v_chunk = _fetch_layer_tensor(item["v_source"], layer_id_t, 1, doc_ids.device)
                        k_next[:, :, item["start"]:item["end"], :] = k_chunk
                        v_next[:, :, item["start"]:item["end"], :] = v_chunk

                    retrieved_k_to_cache = blender.rescale(k_next.squeeze(0), original_positions.squeeze(0), positions.squeeze(0)).unsqueeze(0)
                    retrieved_v_to_cache = v_next
                    past_key_value.update(retrieved_k_to_cache, retrieved_v_to_cache, layer_id_t, {})
                    
                    del k_next, v_next
                return past_key_value

    return past_key_value





def _overlap_build_blent_cache(
        model: "Model",
        cache_database: list, # a list of (h, tokens, k_source, v_source, pos)
        doc_ids: torch.LongTensor,
        recompute_ratio: Union[float, dict[int, List[int]]],
        past_key_value: Optional[DynamicCache] = None,
        stats=None,
        **kwargs,
) -> BaseModelOutputWithPast:
    
    """
    loads one layer after the other,
    but now it laods while inference, 
    so faster (???)
    Yes, it is, about halt the time as sequential, 
    and obv no performance drop
    Just speed this up
    """

    if stats is None:
        stats = {}

    assert doc_ids.shape[0] == 1, "Only batch size = 1 is supported"
    doc_embeds = model.embed_tokens(doc_ids)

    if past_key_value is None:
        past_key_value = DynamicCache()

    past_seen_tokens = past_key_value.get_seq_length()
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + doc_embeds.shape[1],
        device=doc_ids.device, dtype=torch.long
    )

    hidden_states = doc_embeds
    position_embeddings = model.rotary_emb(hidden_states, cache_position.unsqueeze(0))

    blender = CacheBlendImpl(recompute_ratio=recompute_ratio)
    blender.set_positional_encoder(partial(positional_encoder, model))
    blender.set_reverse_positional_encoder(partial(reverse_positional_encoder, model))

    valid_mask = torch.zeros_like(doc_ids, dtype=torch.bool, device="cpu")
    original_positions = torch.zeros_like(doc_ids, dtype=torch.int64)
    positions = torch.tensor([list(range(doc_ids.shape[-1])) for _ in range(doc_ids.shape[0])], device=doc_ids.device)
    query_start_loc = torch.tensor([0, doc_ids.shape[-1]], device=doc_ids.device)

    tkn_to_chunk = []
    chunk_boundaries = []

    layer_fetch_plan = []

    with Timer("scan cache metadata", stats=stats):
        doc_idx = 0
        batch_idx = 0
        cid = 0

        for (h, cached_doc_ids, k_source, v_source, orig_pos) in cache_database:
            cached_n = cached_doc_ids.shape[0]

            assert not valid_mask[batch_idx, doc_idx:doc_idx+cached_n].any()
            valid_mask[batch_idx, doc_idx:doc_idx+cached_n] = True

            if orig_pos is None:
                original_positions[batch_idx, doc_idx:doc_idx+cached_n] = torch.arange(cached_n)
            else:
                original_positions[batch_idx, doc_idx:doc_idx+cached_n] = orig_pos.unsqueeze(0)

            tkn_to_chunk += [cid] * cached_n
            chunk_boundaries.append((doc_idx, doc_idx+cached_n))

            if isinstance(recompute_ratio, dict) and h in recompute_ratio:
                rcmp = [x + doc_idx for x in recompute_ratio[h]]
                recompute_ratio[h] = rcmp

            layer_fetch_plan.append({
                "batch_idx": batch_idx,
                "start": doc_idx,
                "end": doc_idx + cached_n,
                "k_source": k_source,
                "v_source": v_source
            })

            doc_idx += cached_n
            cid += 1

    if isinstance(recompute_ratio, dict):
        blender.recompute_ratio = [x for k in sorted(recompute_ratio.keys()) for x in recompute_ratio[k]]

    valid_mask = valid_mask.to(doc_ids.device, non_blocking=True)

    fresh_q, fresh_k, fresh_v = None, None, None
    local_indices = None
    selected_idx = cache_position
    deselected_idx = None

    use_anti_piaffe = kwargs.get("use_piaffe", False)
    blender.use_anti_piaffe = use_anti_piaffe
    if use_anti_piaffe:
        blender.chunk_boundaries = chunk_boundaries

    config = model.config
    n_q_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_q_heads)
    head_dim = config.hidden_size // n_q_heads
    kwargs['num_q_heads'] = n_q_heads
    kwargs['num_kv_heads'] = n_kv_heads
    kwargs['head_dim'] = head_dim

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    next_layer_retrieved_k = None
    next_layer_retrieved_v = None
    next_layer_prefetched = False

    for layer_id, decoder_layer in enumerate(model.layers[: config.num_hidden_layers]):

        selected_pos_embeds = position_embeddings if local_indices is None else (position_embeddings[0][:, local_indices], position_embeddings[1][:, local_indices])
        causal_mask = model._update_causal_mask(
                None, hidden_states, selected_idx, past_key_value, False
        )

        with Timer(f"Decoding_L{layer_id}", stats=stats):
            layer_output = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=selected_idx.unsqueeze(0),
                past_key_value=past_key_value,
                position_embeddings=selected_pos_embeds,
                use_cache=True,
                **kwargs,
            )
            hidden_states = layer_output[0]

        if layer_id + 1 < config.num_hidden_layers:
            
            with Timer(f"projections_L{layer_id}", stats=stats):
                next_decoder_layer = model.layers[layer_id + 1]
                normalized_hidden_states = next_decoder_layer.input_layernorm(hidden_states)

                q_proj = next_decoder_layer.self_attn.q_proj(normalized_hidden_states)
                k_proj = next_decoder_layer.self_attn.k_proj(normalized_hidden_states)
                v_proj = next_decoder_layer.self_attn.v_proj(normalized_hidden_states)

                bsz, q_len, _ = hidden_states.shape
                fresh_q = q_proj.view(bsz, q_len, n_q_heads, head_dim).transpose(1, 2)
                fresh_k = k_proj.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
                fresh_v = v_proj.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

                cos, sin = selected_pos_embeds if local_indices is None else (position_embeddings[0][:, local_indices], position_embeddings[1][:, local_indices])
                fresh_q, fresh_k = apply_rotary_pos_emb(fresh_q, fresh_k, cos, sin)

            if next_layer_prefetched:
                 retrieved_k_to_blend = next_layer_retrieved_k
                 retrieved_v_to_blend = next_layer_retrieved_v
            else:
                 print(f"First blending for L{layer_id+1}, fetching synchronously.")
                 with Timer(f"retrieve_L{layer_id+1}_sync", stats=stats):
                     ref_k = _fetch_layer_tensor(layer_fetch_plan[0]["k_source"], layer_id + 1, 0, doc_ids.device)
                     current_layer_retrieved_k_sync = torch.empty(
                         (doc_ids.shape[0], ref_k.shape[-3], doc_ids.shape[1], ref_k.shape[-1]),
                         device=doc_ids.device, dtype=model.dtype
                     )
                     current_layer_retrieved_v_sync = torch.empty_like(current_layer_retrieved_k_sync)

                     for item in layer_fetch_plan:
                         k_chunk = _fetch_layer_tensor(item["k_source"], layer_id + 1, 0, doc_ids.device)
                         v_chunk = _fetch_layer_tensor(item["v_source"], layer_id + 1, 1, doc_ids.device)
                         current_layer_retrieved_k_sync[:, :, item["start"]:item["end"], :] = k_chunk
                         current_layer_retrieved_v_sync[:, :, item["start"]:item["end"], :] = v_chunk

                     retrieved_k_to_blend = current_layer_retrieved_k_sync
                     retrieved_v_to_blend = current_layer_retrieved_v_sync

            with Timer(f"Blending_L{layer_id}", stats=stats):
                ret: BlendOutput = blender.blend(
                    layer_id=layer_id + 1,
                    retrieved_k=retrieved_k_to_blend.squeeze(0),
                    retrieved_v=retrieved_v_to_blend.squeeze(0),
                    valid_mask=valid_mask.squeeze(0),
                    original_positions=original_positions.squeeze(0),
                    fresh_q=fresh_q.squeeze(0),
                    fresh_k=fresh_k.squeeze(0),
                    fresh_v=fresh_v.squeeze(0),
                    positions=positions.squeeze(0),
                    query_start_loc=query_start_loc,
                    token_dim=1,
                )
                stats["recomp_ids"] = len(blender.indexes_in_kv)

            local_indices = ret.local_indices

            if ret.query_start_loc is not None:
                if local_indices is not None and local_indices.numel() > 0:
                     hidden_states = hidden_states[:, local_indices]
                     selected_idx = selected_idx[local_indices]
                else:
                     print(f"All tokens reused after blending L{layer_id+1}, fast-forwarding cache.")
                     for layer_id_t in range(layer_id + 2, config.num_hidden_layers):
                         #for the love of god this is sp nasty
                         #i need to make this more verbose somehow
                         k_next_full = torch.empty((doc_ids.shape[0], next_layer_retrieved_k.shape[-3], doc_ids.shape[1], next_layer_retrieved_k.shape[-1]), device=doc_ids.device, dtype=model.dtype) if next_layer_retrieved_k is not None else torch.empty((doc_ids.shape[0], n_kv_heads, doc_ids.shape[1], head_dim), device=doc_ids.device, dtype=model.dtype)
                         v_next_full = torch.empty_like(k_next_full)

                         for item in layer_fetch_plan:
                             k_chunk = _fetch_layer_tensor(item["k_source"], layer_id_t, 0, doc_ids.device)
                             v_chunk = _fetch_layer_tensor(item["v_source"], layer_id_t, 1, doc_ids.device)
                             k_next_full[:, :, item["start"]:item["end"], :] = k_chunk
                             v_next_full[:, :, item["start"]:item["end"], :] = v_chunk

                         past_key_value.update(k_next_full, v_next_full, layer_id_t, {})

                         del k_next_full, v_next_full
                     return past_key_value

            else:
                if local_indices is not None and local_indices.numel() > 0:
                    hidden_states = hidden_states[:, local_indices]
                    selected_idx = selected_idx[local_indices]

        next_layer_prefetched = False
        if layer_id + 2 < config.num_hidden_layers:
            with Timer(f"retrieve_L{layer_id+2}", stats=stats):
                ref_k = _fetch_layer_tensor(layer_fetch_plan[0]["k_source"], layer_id + 2, 0, doc_ids.device)

                current_layer_retrieved_k_prefetch = torch.empty(
                    (doc_ids.shape[0], ref_k.shape[-3], doc_ids.shape[1], ref_k.shape[-1]),
                    device=doc_ids.device, dtype=model.dtype
                )
                current_layer_retrieved_v_prefetch = torch.empty_like(current_layer_retrieved_k_prefetch)

                for item in layer_fetch_plan:
                    k_chunk = _fetch_layer_tensor(item["k_source"], layer_id + 2, 0, doc_ids.device)
                    v_chunk = _fetch_layer_tensor(item["v_source"], layer_id + 2, 1, doc_ids.device)
                    current_layer_retrieved_k_prefetch[:, :, item["start"]:item["end"], :] = k_chunk
                    current_layer_retrieved_v_prefetch[:, :, item["start"]:item["end"], :] = v_chunk

                next_layer_retrieved_k = current_layer_retrieved_k_prefetch
                next_layer_retrieved_v = current_layer_retrieved_v_prefetch
                next_layer_prefetched = True

    return past_key_value

def overlap_build_blent_cache(
        model: "Model",
        cache_database: list, # a list of (h, tokens, k_source, v_source, pos)
        doc_ids: torch.LongTensor,
        recompute_ratio: Union[float, dict[int, List[int]]],
        past_key_value: Optional[DynamicCache] = None,
        stats=None,
        **kwargs,
) -> BaseModelOutputWithPast:
    
    """
    like overlap_build_blent_cache, but better in theory, we will have to see.
    Fixed one potential edge case problem when doing full recomp.
    Loads one layer after the other, but prefetches Layer N+1 
    while Layer N is being computed by the NPU.
    Faster but uses slightly more memory (2 layers at once).
    """

    if stats is None:
        stats = {}

    assert doc_ids.shape[0] == 1, "Only batch size = 1 is supported"
    doc_embeds = model.embed_tokens(doc_ids)

    if past_key_value is None:
        past_key_value = DynamicCache()

    past_seen_tokens = past_key_value.get_seq_length()
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + doc_embeds.shape[1],
        device=doc_ids.device, dtype=torch.long
    )

    hidden_states = doc_embeds
    position_embeddings = model.rotary_emb(hidden_states, cache_position.unsqueeze(0))

    blender = CacheBlendImpl(recompute_ratio=recompute_ratio)
    blender.set_positional_encoder(partial(positional_encoder, model))
    blender.set_reverse_positional_encoder(partial(reverse_positional_encoder, model))

    valid_mask = torch.zeros_like(doc_ids, dtype=torch.bool, device="cpu")
    original_positions = torch.zeros_like(doc_ids, dtype=torch.int64)
    positions = torch.tensor([list(range(doc_ids.shape[-1])) for _ in range(doc_ids.shape[0])], device=doc_ids.device)
    query_start_loc = torch.tensor([0, doc_ids.shape[-1]], device=doc_ids.device)

    tkn_to_chunk = []
    chunk_boundaries = []
    layer_fetch_plan = []

    with Timer("scan cache metadata", stats=stats):
        doc_idx = 0
        batch_idx = 0
        cid = 0

        for (h, cached_doc_ids, k_source, v_source, orig_pos) in cache_database:
            cached_n = cached_doc_ids.shape[0]

            assert not valid_mask[batch_idx, doc_idx:doc_idx+cached_n].any()
            valid_mask[batch_idx, doc_idx:doc_idx+cached_n] = True

            if orig_pos is None:
                original_positions[batch_idx, doc_idx:doc_idx+cached_n] = torch.arange(cached_n)
            else:
                original_positions[batch_idx, doc_idx:doc_idx+cached_n] = orig_pos.unsqueeze(0)

            tkn_to_chunk += [cid] * cached_n
            chunk_boundaries.append((doc_idx, doc_idx+cached_n))

            if isinstance(recompute_ratio, dict) and h in recompute_ratio:
                rcmp = [x + doc_idx for x in recompute_ratio[h]]
                recompute_ratio[h] = rcmp

            layer_fetch_plan.append({
                "batch_idx": batch_idx,
                "start": doc_idx,
                "end": doc_idx + cached_n,
                "k_source": k_source,
                "v_source": v_source
            })

            doc_idx += cached_n
            cid += 1

    if isinstance(recompute_ratio, dict):
        blender.recompute_ratio = [x for k in sorted(recompute_ratio.keys()) for x in recompute_ratio[k]]

    valid_mask = valid_mask.to(doc_ids.device, non_blocking=True)

    fresh_q, fresh_k, fresh_v = None, None, None
    local_indices = None
    selected_idx = cache_position
    deselected_idx = None

    use_anti_piaffe = kwargs.get("use_piaffe", False)
    blender.use_anti_piaffe = use_anti_piaffe
    if use_anti_piaffe:
        blender.chunk_boundaries = chunk_boundaries

    config = model.config
    n_q_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_q_heads)
    head_dim = config.hidden_size // n_q_heads
    kwargs['num_q_heads'] = n_q_heads
    kwargs['num_kv_heads'] = n_kv_heads
    kwargs['head_dim'] = head_dim

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    #Prefetch state variables
    next_layer_retrieved_k = None
    next_layer_retrieved_v = None
    next_layer_prefetched = False

    for layer_id, decoder_layer in enumerate(model.layers[: config.num_hidden_layers]):

        selected_pos_embeds = position_embeddings if local_indices is None else (position_embeddings[0][:, local_indices], position_embeddings[1][:, local_indices])
        
        causal_mask = model._update_causal_mask(
                None, hidden_states, selected_idx, past_key_value, False
        )

        with Timer(f"Decoding_L{layer_id}", stats=stats):
            layer_output = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=selected_idx.unsqueeze(0),
                past_key_value=past_key_value,
                position_embeddings=selected_pos_embeds,
                use_cache=True,
                **kwargs,
            )
            hidden_states = layer_output[0]

        if layer_id + 1 < config.num_hidden_layers:
            
            with Timer(f"projections_L{layer_id}", stats=stats):
                next_decoder_layer = model.layers[layer_id + 1]
                normalized_hidden_states = next_decoder_layer.input_layernorm(hidden_states)

                q_proj = next_decoder_layer.self_attn.q_proj(normalized_hidden_states)
                k_proj = next_decoder_layer.self_attn.k_proj(normalized_hidden_states)
                v_proj = next_decoder_layer.self_attn.v_proj(normalized_hidden_states)

                bsz, q_len, _ = hidden_states.shape
                fresh_q = q_proj.view(bsz, q_len, n_q_heads, head_dim).transpose(1, 2)
                fresh_k = k_proj.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
                fresh_v = v_proj.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

                cos, sin = selected_pos_embeds if local_indices is None else (position_embeddings[0][:, local_indices], position_embeddings[1][:, local_indices])
                fresh_q, fresh_k = apply_rotary_pos_emb(fresh_q, fresh_k, cos, sin)

            if next_layer_prefetched:
                 retrieved_k_to_blend = next_layer_retrieved_k
                 retrieved_v_to_blend = next_layer_retrieved_v
            else:
                 # Sync Fetch
                 # print(f"First blending for L{layer_id+1}, fetching synchronously.")
                 with Timer(f"retrieve_L{layer_id+1}_sync", stats=stats):
                     ref_k = _fetch_layer_tensor(layer_fetch_plan[0]["k_source"], layer_id + 1, 0, doc_ids.device)
                     current_layer_retrieved_k_sync = torch.empty(
                         (doc_ids.shape[0], ref_k.shape[-3], doc_ids.shape[1], ref_k.shape[-1]),
                         device=doc_ids.device, dtype=model.dtype
                     )
                     current_layer_retrieved_v_sync = torch.empty_like(current_layer_retrieved_k_sync)

                     for item in layer_fetch_plan:
                         k_chunk = _fetch_layer_tensor(item["k_source"], layer_id + 1, 0, doc_ids.device)
                         v_chunk = _fetch_layer_tensor(item["v_source"], layer_id + 1, 1, doc_ids.device)
                         current_layer_retrieved_k_sync[:, :, item["start"]:item["end"], :] = k_chunk
                         current_layer_retrieved_v_sync[:, :, item["start"]:item["end"], :] = v_chunk

                     retrieved_k_to_blend = current_layer_retrieved_k_sync
                     retrieved_v_to_blend = current_layer_retrieved_v_sync

            with Timer(f"Blending_L{layer_id}", stats=stats):
                ret: BlendOutput = blender.blend(
                    layer_id=layer_id + 1,
                    retrieved_k=retrieved_k_to_blend.squeeze(0),
                    retrieved_v=retrieved_v_to_blend.squeeze(0),
                    valid_mask=valid_mask.squeeze(0),
                    original_positions=original_positions.squeeze(0),
                    fresh_q=fresh_q.squeeze(0),
                    fresh_k=fresh_k.squeeze(0),
                    fresh_v=fresh_v.squeeze(0),
                    positions=positions.squeeze(0),
                    query_start_loc=query_start_loc,
                    token_dim=1,
                )
                stats["recomp_ids"] = len(blender.indexes_in_kv)

            # Cleanup fetched layers
            del retrieved_k_to_blend
            del retrieved_v_to_blend
            # Clear prefetch references
            next_layer_retrieved_k = None
            next_layer_retrieved_v = None

            local_indices = ret.local_indices

            if ret.query_start_loc is not None:
                if local_indices is not None and local_indices.numel() > 0:
                     hidden_states = hidden_states[:, local_indices]
                     selected_idx = selected_idx[local_indices]
                else:
                     # 100% reuse case
                     # print(f"All tokens reused after blending L{layer_id+1}, fast-forwarding cache.")
                     
                     # FIX: Check reference shape from source, not from next_layer_retrieved_k (which is None)
                     ref_k_ff = _fetch_layer_tensor(layer_fetch_plan[0]["k_source"], layer_id + 2, 0, doc_ids.device)
                     
                     for layer_id_t in range(layer_id + 2, config.num_hidden_layers):
                         k_next_full = torch.empty((doc_ids.shape[0], ref_k_ff.shape[-3], doc_ids.shape[1], ref_k_ff.shape[-1]), device=doc_ids.device, dtype=model.dtype)
                         v_next_full = torch.empty_like(k_next_full)

                         for item in layer_fetch_plan:
                             k_chunk = _fetch_layer_tensor(item["k_source"], layer_id_t, 0, doc_ids.device)
                             v_chunk = _fetch_layer_tensor(item["v_source"], layer_id_t, 1, doc_ids.device)
                             k_next_full[:, :, item["start"]:item["end"], :] = k_chunk
                             v_next_full[:, :, item["start"]:item["end"], :] = v_chunk

                         past_key_value.update(k_next_full, v_next_full, layer_id_t, {})

                         del k_next_full, v_next_full
                     return past_key_value

            else:
                if local_indices is not None and local_indices.numel() > 0:
                    hidden_states = hidden_states[:, local_indices]
                    selected_idx = selected_idx[local_indices]

        # Prefetch Logic for L+2
        next_layer_prefetched = False
        if layer_id + 2 < config.num_hidden_layers:
            with Timer(f"retrieve_L{layer_id+2}", stats=stats):
                ref_k = _fetch_layer_tensor(layer_fetch_plan[0]["k_source"], layer_id + 2, 0, doc_ids.device)

                current_layer_retrieved_k_prefetch = torch.empty(
                    (doc_ids.shape[0], ref_k.shape[-3], doc_ids.shape[1], ref_k.shape[-1]),
                    device=doc_ids.device, dtype=model.dtype
                )
                current_layer_retrieved_v_prefetch = torch.empty_like(current_layer_retrieved_k_prefetch)

                for item in layer_fetch_plan:
                    k_chunk = _fetch_layer_tensor(item["k_source"], layer_id + 2, 0, doc_ids.device)
                    v_chunk = _fetch_layer_tensor(item["v_source"], layer_id + 2, 1, doc_ids.device)
                    current_layer_retrieved_k_prefetch[:, :, item["start"]:item["end"], :] = k_chunk
                    current_layer_retrieved_v_prefetch[:, :, item["start"]:item["end"], :] = v_chunk

                next_layer_retrieved_k = current_layer_retrieved_k_prefetch
                next_layer_retrieved_v = current_layer_retrieved_v_prefetch
                next_layer_prefetched = True

    return past_key_value


def _token_runs(mask: torch.Tensor):
    if mask.numel() == 0:
        return []
    indices = mask.nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        return []
    runs = []
    start = int(indices[0])
    prev = start
    for idx in indices[1:].tolist():
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            runs.append((start, prev + 1))
            start = prev = idx
    runs.append((start, prev + 1))
    return runs


def _slice_chunk_tokens(chunk: torch.Tensor, start: int, end: int):
    index = [slice(None)] * chunk.dim()
    index[-2] = slice(start, end)
    return chunk[tuple(index)]


## this is a duplicate, remember to clean this up later
def _fill_buffer_async(fetch_plan, layer_idx, k_buffer, v_buffer, copy_token_mask=None):
    """
    Fill preallocated buffers for a given layer.
    Uses non_blocking=True to allow overlap with NPU compute.
    """
    device = k_buffer.device
    
    
    t0 = time.perf_counter()
    
    
    for item in fetch_plan:
        token_runs = None
        if copy_token_mask is not None:
            token_runs = _token_runs(copy_token_mask[item["start"]:item["end"]])
            if not token_runs:
                continue

        # fetches initiate the transfer
        k_chunk = _fetch_layer_tensor(item["k_source"], layer_idx, 0, device)
        v_chunk = _fetch_layer_tensor(item["v_source"], layer_idx, 1, device)

        try:
            if copy_token_mask is None:
                k_slice = k_buffer[:, :, item["start"]:item["end"], :]
                v_slice = v_buffer[:, :, item["start"]:item["end"], :]
                k_slice.copy_(k_chunk, non_blocking=k_chunk.is_pinned())
                v_slice.copy_(v_chunk, non_blocking=v_chunk.is_pinned())
            else:
                for local_start, local_end in token_runs:
                    global_start = item["start"] + local_start
                    global_end = item["start"] + local_end
                    k_slice = k_buffer[:, :, global_start:global_end, :]
                    v_slice = v_buffer[:, :, global_start:global_end, :]
                    k_src = _slice_chunk_tokens(k_chunk, local_start, local_end)
                    v_src = _slice_chunk_tokens(v_chunk, local_start, local_end)
                    k_slice.copy_(k_src, non_blocking=k_chunk.is_pinned())
                    v_slice.copy_(v_src, non_blocking=v_chunk.is_pinned())
        except RuntimeError as e:
            print(f"Porcapaletta! Copy mismatch at Layer {layer_idx}. "
                  f"Buffer Slice: {k_slice.shape}, Source: {k_chunk.shape}")
            raise e
      
    # DEBUG: timing for async copy scheduling
    t1 = time.perf_counter()
    dt_ms = (t1 - t0) * 1000
    
    # If dt_ms is tiny (< 0.2ms) CPU didn't wait -> PIPELINING ACTIVE
    # If dt_ms is large (> 2.0ms) BLOCKED / SERIAL
    PRINT_COPY_DURATION = False
    if PRINT_COPY_DURATION and layer_idx < 5: 
        print(f"DEBUG PIPELINE: Layer {layer_idx} Copy Command Duration: {dt_ms:.4f} ms")
    # DEBUG END ---
        
        
def _estimate_buffer_bytes(buffer_shape, dtype, buffers=1):
    return int(np.prod(buffer_shape)) * torch.tensor([], dtype=dtype).element_size() * buffers


def _get_device_free_bytes(device: torch.device):
    try:
        if device.type == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "mem_get_info"):
            free_bytes, _ = torch.npu.mem_get_info()
            return int(free_bytes)
    except Exception:
        return None
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
            return int(free_bytes)
    except Exception:
        return None
    return None


def _should_use_low_mem_prefetch(buffer_shape, dtype, device, stats=None):
    # Ping-pong uses 2 buffers for K and 2 for V.
    pingpong_bytes = _estimate_buffer_bytes(buffer_shape, dtype, buffers=4)
    free_bytes = _get_device_free_bytes(device)

    max_gb = float(os.getenv("CB_PREFETCH_MAX_GB", "4.0"))
    max_bytes = max_gb * (1024**3)

    use_low_mem = False
    reason = ""

    if free_bytes is not None and pingpong_bytes > free_bytes * 0.5:
        use_low_mem = True
        reason = "pingpong_gt_half_free"
    if pingpong_bytes > max_bytes:
        use_low_mem = True
        if not reason:
            reason = "pingpong_gt_max_gb"

    if stats is not None:
        stats["prefetch_pingpong_bytes"] = pingpong_bytes
        if free_bytes is not None:
            stats["prefetch_free_bytes"] = free_bytes
        stats["prefetch_use_low_mem"] = use_low_mem
        if reason:
            stats["prefetch_low_mem_reason"] = reason

    return use_low_mem


def _prefetch_buffer_shape(fetch_plan, doc_ids):
    # Determine shape from the first source without copying a layer to device.
    ref_source = fetch_plan[0]["k_source"]
    if hasattr(ref_source, "states"):
        ref_shape = ref_source.states[0][0].shape
    else:
        ref_shape = ref_source[0].shape
    return (doc_ids.shape[0], ref_shape[-3], doc_ids.shape[1], ref_shape[-1])


def _make_npu_prefetch_stream(device: torch.device):
    if not _env_flag("CB_ENABLE_ASYNC_PREFETCH", default=False):
        return None
    if device.type != "npu":
        return None
    npu = getattr(torch, "npu", None)
    if npu is None or not all(hasattr(npu, name) for name in ("Stream", "Event", "stream", "current_stream")):
        return None
    try:
        return npu.Stream(device=device)
    except Exception:
        pass
    if hasattr(npu, "device"):
        try:
            with npu.device(device):
                return npu.Stream()
        except Exception:
            pass
    try:
        return npu.Stream()
    except Exception:
        return None


def _make_npu_prefetch_event():
    try:
        return torch.npu.Event()
    except Exception:
        return None


def _stream_wait_event(stream, event) -> bool:
    try:
        if hasattr(stream, "wait_event"):
            stream.wait_event(event)
            return True
    except Exception:
        pass
    try:
        if hasattr(event, "wait"):
            event.wait(stream)
            return True
    except Exception:
        pass
    return False


def _record_stream_event(event, stream=None) -> bool:
    try:
        if stream is None:
            event.record()
        else:
            event.record(stream)
        return True
    except TypeError:
        try:
            event.record()
            return True
        except Exception:
            return False
    except Exception:
        return False


def _layer_prefetch_generator_sync(fetch_plan, doc_ids, dtype, num_layers, stats, start_layer: int, copy_token_mask=None):
    buffer_shape = _prefetch_buffer_shape(fetch_plan, doc_ids)
    device = doc_ids.device

    k_bufs = [
        torch.zeros(buffer_shape, device=device, dtype=dtype),
        torch.zeros(buffer_shape, device=device, dtype=dtype),
    ]
    v_bufs = [
        torch.zeros(buffer_shape, device=device, dtype=dtype),
        torch.zeros(buffer_shape, device=device, dtype=dtype),
    ]

    ping = 0
    if start_layer < num_layers:
        with Timer(f"prefetch_L{start_layer}_init", stats=stats):
            _fill_buffer_async(fetch_plan, start_layer, k_bufs[ping], v_bufs[ping], copy_token_mask)
            if device.type == "npu" and hasattr(torch, "npu"):
                torch.npu.synchronize()

    for layer_id in range(start_layer, num_layers):
        curr_k = k_bufs[ping]
        curr_v = v_bufs[ping]

        if isdbg() and layer_id == start_layer:
            if curr_k.abs().sum() == 0:
                print(f"[CRITICAL FAIL]Ao! Generator yielding EMPTY BUFFER for Layer {layer_id}")

        next_layer = layer_id + 1
        pong = 1 - ping
        if next_layer < num_layers:
            _fill_buffer_async(fetch_plan, next_layer, k_bufs[pong], v_bufs[pong], copy_token_mask)

        yield curr_k, curr_v
        ping = pong


def _layer_prefetch_generator_low_mem(fetch_plan, doc_ids, dtype, num_layers, stats, start_layer: int, copy_token_mask=None):
    buffer_shape = _prefetch_buffer_shape(fetch_plan, doc_ids)
    device = doc_ids.device

    k_buf = torch.empty(buffer_shape, device=device, dtype=dtype)
    v_buf = torch.empty_like(k_buf)
    for layer_id in range(start_layer, num_layers):
        with Timer(f"prefetch_L{layer_id}_seq", stats=stats):
            _fill_buffer_async(fetch_plan, layer_id, k_buf, v_buf, copy_token_mask)
            if device.type == "npu" and hasattr(torch, "npu"):
                torch.npu.synchronize()
        yield k_buf, v_buf


class _AsyncLayerPrefetchIterator:
    def __init__(self, fetch_plan, doc_ids, dtype, num_layers, stats, start_layer: int, copy_stream, copy_token_mask=None):
        self.fetch_plan = fetch_plan
        self.doc_ids = doc_ids
        self.dtype = dtype
        self.num_layers = num_layers
        self.stats = stats
        self.start_layer = start_layer
        self.layer_id = start_layer
        self.ping = 0
        self.copy_stream = copy_stream
        self.copy_token_mask = copy_token_mask
        self.buffer_shape = _prefetch_buffer_shape(fetch_plan, doc_ids)
        self.device = doc_ids.device
        self.k_bufs = [
            torch.zeros(self.buffer_shape, device=self.device, dtype=dtype),
            torch.zeros(self.buffer_shape, device=self.device, dtype=dtype),
        ]
        self.v_bufs = [
            torch.zeros(self.buffer_shape, device=self.device, dtype=dtype),
            torch.zeros(self.buffer_shape, device=self.device, dtype=dtype),
        ]
        self.copy_events = [None, None]
        self.buffer_was_yielded = [False, False]
        if stats is not None:
            stats["prefetch_async_enabled"] = True
        if self.layer_id < self.num_layers:
            self._start_prefetch(self.layer_id, self.ping, init=True)

    def __iter__(self):
        return self

    def _record_current_stream_event(self):
        event = _make_npu_prefetch_event()
        if event is None:
            return None
        if _record_stream_event(event, torch.npu.current_stream()):
            return event
        return None

    def _start_prefetch(self, layer_id: int, buffer_idx: int, init: bool = False):
        done_event = _make_npu_prefetch_event()
        if done_event is None:
            raise RuntimeError("NPU event creation failed after async prefetch was enabled")

        needs_reuse_wait = self.buffer_was_yielded[buffer_idx]
        reuse_event = None
        if needs_reuse_wait:
            reuse_event = self._record_current_stream_event()

        timer_name = f"prefetch_L{layer_id}_{'init' if init else 'schedule'}"
        with Timer(timer_name, stats=self.stats):
            with torch.npu.stream(self.copy_stream):
                if needs_reuse_wait and (reuse_event is None or not _stream_wait_event(self.copy_stream, reuse_event)):
                    torch.npu.synchronize()
                _fill_buffer_async(
                    self.fetch_plan,
                    layer_id,
                    self.k_bufs[buffer_idx],
                    self.v_bufs[buffer_idx],
                    self.copy_token_mask,
                )
                if not _record_stream_event(done_event, self.copy_stream):
                    raise RuntimeError("NPU event record failed after async prefetch was enabled")

        self.copy_events[buffer_idx] = done_event

    def _wait_for_prefetch(self, layer_id: int, buffer_idx: int):
        event = self.copy_events[buffer_idx]
        if event is None:
            return
        with Timer(f"prefetch_L{layer_id}_wait", stats=self.stats):
            current_stream = torch.npu.current_stream()
            if not _stream_wait_event(current_stream, event):
                event.synchronize()
        self.copy_events[buffer_idx] = None

    def __next__(self):
        if self.layer_id >= self.num_layers:
            raise StopIteration

        layer_id = self.layer_id
        buffer_idx = self.ping
        self._wait_for_prefetch(layer_id, buffer_idx)

        curr_k = self.k_bufs[buffer_idx]
        curr_v = self.v_bufs[buffer_idx]

        if isdbg() and layer_id == self.start_layer:
            if curr_k.abs().sum() == 0:
                print(f"[CRITICAL FAIL]Ao! Generator yielding EMPTY BUFFER for Layer {layer_id}")

        next_layer = layer_id + 1
        next_buffer_idx = 1 - buffer_idx
        if next_layer < self.num_layers:
            self._start_prefetch(next_layer, next_buffer_idx)

        self.buffer_was_yielded[buffer_idx] = True
        self.layer_id = next_layer
        self.ping = next_buffer_idx
        return curr_k, curr_v


def _build_copy_token_mask(num_tokens: int, skip_token_indices):
    if skip_token_indices is None:
        return None
    copy_token_mask = torch.ones(num_tokens, dtype=torch.bool, device="cpu")
    skip = torch.as_tensor(skip_token_indices, dtype=torch.long, device="cpu")
    skip = skip[(skip >= 0) & (skip < num_tokens)]
    if skip.numel() > 0:
        copy_token_mask[skip] = False
    return copy_token_mask


def _build_layer_prefetch_iterator(
        fetch_plan,
        doc_ids,
        dtype,
        num_layers,
        stats,
        start_layer: int,
        skip_token_indices=None,
):
    buffer_shape = _prefetch_buffer_shape(fetch_plan, doc_ids)
    device = doc_ids.device
    copy_token_mask = _build_copy_token_mask(doc_ids.shape[1], skip_token_indices)
    if stats is not None and copy_token_mask is not None:
        stats["prefetch_copy_tokens"] = int(copy_token_mask.sum().item())
        stats["prefetch_skip_tokens"] = int(copy_token_mask.numel() - copy_token_mask.sum().item())

    if _should_use_low_mem_prefetch(buffer_shape, dtype, device, stats):
        if stats is not None:
            stats["prefetch_async_enabled"] = False
        return _layer_prefetch_generator_low_mem(
            fetch_plan, doc_ids, dtype, num_layers, stats, start_layer, copy_token_mask
        )

    copy_stream = _make_npu_prefetch_stream(device)
    event = _make_npu_prefetch_event() if copy_stream is not None else None
    if copy_stream is None or event is None:
        if stats is not None:
            stats["prefetch_async_enabled"] = False
        return _layer_prefetch_generator_sync(
            fetch_plan, doc_ids, dtype, num_layers, stats, start_layer, copy_token_mask
        )

    return _AsyncLayerPrefetchIterator(
        fetch_plan, doc_ids, dtype, num_layers, stats, start_layer, copy_stream, copy_token_mask
    )


def layer_prefetch_generator2(fetch_plan, doc_ids, dtype, num_layers, stats, start_layer: int = 1, skip_token_indices=None):
    """
    Double-buffered prefetch iterator for CacheBlend.
    Layer 0 is computed normally, so cached retrieval starts at layer 1.
    """
    return _build_layer_prefetch_iterator(
        fetch_plan,
        doc_ids,
        dtype,
        num_layers,
        stats,
        start_layer=start_layer,
        skip_token_indices=skip_token_indices,
    )

def generator_build_blent_cache4(
        model: "Model",
        cache_database: list, 
        doc_ids: torch.LongTensor,
        recompute_ratio: Union[float, dict[int, List[int]]],
        past_key_value: Optional[DynamicCache] = None,
        stats = None,
        external_scores: Optional[torch.Tensor] = None, #new, please god i hope i don't break this
        **kwargs,
) -> BaseModelOutputWithPast:
    """
    So far the most correct one, uses the generator properly
    and has a correct full recomp and partial reocmp behavior
    Works both with zcf and cacheblend code and cc
    """
    
    
    if stats is None: stats = {}
    assert doc_ids.shape[0] == 1
    doc_embeds = model.embed_tokens(doc_ids)
    if past_key_value is None: past_key_value = DynamicCache()
    
    past_seen_tokens = past_key_value.get_seq_length()
    cache_position = torch.arange(past_seen_tokens, past_seen_tokens + doc_embeds.shape[1], device=doc_ids.device, dtype=torch.long)
    hidden_states = doc_embeds
    position_embeddings = model.rotary_emb(hidden_states, cache_position.unsqueeze(0))
    
    blender = CacheBlendImpl(recompute_ratio=recompute_ratio)
    blender.set_positional_encoder(partial(positional_encoder, model))
    blender.set_reverse_positional_encoder(partial(reverse_positional_encoder, model))
    
    #################################################################
    #new
    if external_scores is not None:
        blender.set_external_scores(external_scores)
    ###################################################################
    
    
    valid_mask = torch.zeros_like(doc_ids, dtype=torch.bool, device="cpu")
    original_positions = torch.zeros_like(doc_ids, dtype=torch.int64)
    positions = torch.tensor([list(range(doc_ids.shape[-1])) for _ in range(doc_ids.shape[0])], device=doc_ids.device)
    query_start_loc = torch.tensor([0, doc_ids.shape[-1]], device=doc_ids.device)
    
    tkn_to_chunk, chunk_boundaries = [], []
    layer_fetch_plan = []
    
    
    #addition
    #fixed a potential big bug for the case
    #in which a prompt contains the same doc twice and we do some partial recomputation
    #this never showed up in our dataset by nature of how rag works
    #but still, better safe then sorry i guess
    #for the older version, in case thios new creates problem, refer to the commit of 4-12-2025 on wip-mltchunk, made before lunch
    
    
    all_recompute_indices = []

    with Timer("scan cache metadata", stats=stats):
        doc_idx = 0
        batch_idx = 0
        cid = 0
        for (h, cached_doc_ids, k_source, v_source, orig_pos) in cache_database:
            cached_n = cached_doc_ids.shape[0]
            valid_mask[batch_idx, doc_idx:doc_idx+cached_n] = True
            if orig_pos is None:
                original_positions[batch_idx, doc_idx:doc_idx+cached_n] = torch.arange(cached_n)
            else:
                original_positions[batch_idx, doc_idx:doc_idx+cached_n] = orig_pos.unsqueeze(0)
            
            tkn_to_chunk += [cid] * cached_n
            chunk_boundaries.append((doc_idx, doc_idx+cached_n))
            if isinstance(recompute_ratio, dict) and h in recompute_ratio:
                raw_indices = recompute_ratio[h]
                shifted_indices = [x + doc_idx for x in raw_indices]
                all_recompute_indices.extend(shifted_indices)
                #recompute_ratio[h] = [x + doc_idx for x in recompute_ratio[h]]
            
            layer_fetch_plan.append({
                "batch_idx": batch_idx, "start": doc_idx, "end": doc_idx + cached_n,
                "k_source": k_source, "v_source": v_source
            })
            doc_idx += cached_n
            cid += 1

    if isinstance(recompute_ratio, dict):
        blender.recompute_ratio = sorted(all_recompute_indices)
    valid_mask = valid_mask.to(doc_ids.device, non_blocking=True)

    config = model.config
    n_q_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_q_heads)
    head_dim = config.hidden_size // n_q_heads
    kwargs.update({'num_q_heads': n_q_heads, 'num_kv_heads': n_kv_heads, 'head_dim': head_dim})

    blender.use_anti_piaffe = kwargs.get("use_piaffe", False)
    if blender.use_anti_piaffe:
        blender.chunk_boundaries = chunk_boundaries

    fixed_recompute_indices = recompute_ratio if isinstance(recompute_ratio, list) else None
    prefetch_start_layer = 2 if fixed_recompute_indices is not None else 1
    cache_generator = layer_prefetch_generator2(
        layer_fetch_plan,
        doc_ids,
        model.dtype,
        config.num_hidden_layers,
        stats,
        start_layer=prefetch_start_layer,
        skip_token_indices=fixed_recompute_indices,
    )
    
    #########################################33333addition#####################
    #replicate the og build blend cache behaviour
    if isinstance(recompute_ratio, float) and recompute_ratio == 0.0:
        #cache_log.debug("rr=0 fast path: rescaling all layers without blending")
        
        # Determine tensor shapes
        sample_cws = cache_database[0][2]  # k_source from first entry
        n_layers = len(sample_cws.states)
        total_seq_len = doc_ids.shape[1]
        n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = config.hidden_size // config.num_attention_heads

        # Pre-allocate dense tensors on device
        retrieved_k = torch.empty(
            (n_layers, 1, n_kv_heads, total_seq_len, head_dim),
            device=doc_ids.device, dtype=model.dtype
        )
        retrieved_v = torch.empty(
            (n_layers, 1, n_kv_heads, total_seq_len, head_dim),
            device=doc_ids.device, dtype=model.dtype
        )

        # Fill tensors from cache_database
        doc_idx = 0
        for (_, _, k_source, v_source, orig_pos) in cache_database:
            # k_source and v_source are CachedChunk objects
            cached_n = len(k_source)
    
            k_layers = []
            v_layers = []
            
           
            for layer_k, layer_v in k_source.states:
                kk = layer_k.to(doc_ids.device)
                vv = layer_v.to(doc_ids.device)
                
                
                if kk.ndim == 4: 
                    kk = kk.squeeze(0)
                if vv.ndim == 4:
                    vv = vv.squeeze(0)
                    
                k_layers.append(kk)
                v_layers.append(vv)
            

            
            k_block = torch.stack(k_layers, dim=0).unsqueeze(1)  
            v_block = torch.stack(v_layers, dim=0).unsqueeze(1) 
            
            retrieved_k[:, :, :, doc_idx:doc_idx+cached_n, :] = k_block
            retrieved_v[:, :, :, doc_idx:doc_idx+cached_n, :] = v_block
            doc_idx += cached_n
            

        # Rescale all layers and build cache
        past_key_value = DynamicCache()
        for layer_id in range(n_layers):
            k_rescaled = blender.rescale(
                retrieved_k[layer_id].squeeze(0),
                original_positions.squeeze(0),
                positions.squeeze(0)
            ).unsqueeze(0)
            v_final = retrieved_v[layer_id]  # V doesn't need rotation
            past_key_value.update(k_rescaled, v_final, layer_id, {})
        
        return past_key_value
    
    #############################addition######################################
    

    fresh_q, fresh_k, fresh_v = None, None, None
    local_indices = None
    selected_idx = cache_position
    deselected_idx = None
    xattn_ids = None
    

    for layer_id, decoder_layer in enumerate(model.layers[: config.num_hidden_layers]):
        
        # Forward Pass (Compute Layer N)
        
        selected_pos_embeds = position_embeddings if local_indices is None else (position_embeddings[0][:, local_indices], position_embeddings[1][:, local_indices])
        selected_doc_embeds = doc_embeds if local_indices is None else doc_embeds[:, local_indices]
        
        with Timer(f"Decoding_L{layer_id}", stats=stats):
            if (not blender.use_anti_piaffe) or (xattn_ids is None):
                causal_mask = _make_causal_mask(model,
                        layer_id=layer_id,
                        input_tensor=selected_doc_embeds,
                        cache_position=selected_idx,
                        past_key_value=past_key_value,
                        last_positions_in_cache=deselected_idx
                        )
            else:
                if layer_id == 1:
                    causal_mask = build_chunked_causal_mask(
                        model=model,
                        layer_id=layer_id,
                        input_tensor=selected_doc_embeds,
                        cache_position=selected_idx,
                        past_key_value=past_key_value,
                        last_positions_in_cache=deselected_idx,
                        additional_ids=xattn_ids,
                        chunk_ids = torch.tensor(tkn_to_chunk)
                    )
            
            layer_output = decoder_forward(decoder_layer,
                    hidden_states,
                    query_states=fresh_q,
                    key_states=fresh_k,
                    value_states=fresh_v,
                    attention_mask=causal_mask,
                    position_ids=selected_idx.unsqueeze(0),
                    past_key_value=past_key_value,
                    cache_position=selected_idx,
                    position_embeddings=selected_pos_embeds,
                    **kwargs,
                    )

        #prepare n ext Layer (
        if layer_id + 1 < config.num_hidden_layers:
            
            
            with Timer(f"projections_L{layer_id}", stats=stats):
                next_decoder_layer = model.layers[layer_id+1]
                hidden_states = layer_output
                normalized_hidden_states = next_decoder_layer.input_layernorm(hidden_states)
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, next_decoder_layer.self_attn.head_dim)
                fresh_q, fresh_k, fresh_v = proj(next_decoder_layer.self_attn, normalized_hidden_states, hidden_shape)
                cos, sin = selected_pos_embeds
                if not inplace_rope:
                    fresh_q, fresh_k = apply_rotary_pos_emb(fresh_q, fresh_k, cos, sin)
                else:
                    fresh_q, fresh_k = apply_rotary_pos_emb_inplace(fresh_q, fresh_k, cos, sin)
            
            # Retrieve Cache for Layer N+1
            # next method is called here. This returns the buffer for N+1
            # and triggers the prefetch for N+2.
            with Timer(f"retrieve_L{layer_id+1}", stats=stats):
                if fixed_recompute_indices is not None and layer_id == 0:
                    # Exact-token mode only uses retrieved KV at layer 1 for
                    # shape checks; the selected token set is already known.
                    retrieved_k_layer, retrieved_v_layer = fresh_k, fresh_v
                else:
                    retrieved_k_layer, retrieved_v_layer = next(cache_generator)

            #Blend Layer N+1
            with Timer(f"Blending", stats=stats):
                ret: BlendOutput = blender.blend(
                        layer_id=layer_id+1,
                        retrieved_k=retrieved_k_layer.squeeze(0), # Use the single layer yielded
                        retrieved_v=retrieved_v_layer.squeeze(0),
                        valid_mask=valid_mask.squeeze(0),
                        original_positions=original_positions.squeeze(0),
                        fresh_q=fresh_q.squeeze(0),
                        fresh_k=fresh_k.squeeze(0),
                        fresh_v=fresh_v.squeeze(0),
                        positions=positions.squeeze(0),
                        query_start_loc=query_start_loc,
                        token_dim=1,
                        )
                stats["recomp_ids"] = len(blender.indexes_in_kv)

            local_indices = ret.local_indices
            
            
            if ret.query_start_loc is not None:
                hidden_states = hidden_states[:, local_indices]
                query_start_loc = query_start_loc
                selected_idx = cache_position[local_indices]
                valid_mask_tmp = valid_mask.clone().detach().reshape(-1)
                valid_mask_tmp[selected_idx] = False
                deselected_idx = torch.where(valid_mask_tmp)[0]
                xattn_ids = ret.xattn_ids
                tkn_to_chunk_prime = [x for x in tkn_to_chunk if x in selected_idx]
                tkn_to_chunk = [x for x in tkn_to_chunk if x not in selected_idx]
                tkn_to_chunk += tkn_to_chunk_prime
                
            fresh_q = ret.q.unsqueeze(0)
            if deselected_idx is not None:
                fresh_k = ret.k[:, ret.local_indices, :].unsqueeze(0)
                fresh_v = ret.v[:, ret.local_indices, :].unsqueeze(0)
                to_cache_k = ret.k[:, deselected_idx, :].unsqueeze(0)
                to_cache_v = ret.v[:, deselected_idx, :].unsqueeze(0)
                past_key_value.update(to_cache_k, to_cache_v, layer_id+1, {})
            else:
                fresh_k = ret.k.unsqueeze(0)
                fresh_v = ret.v.unsqueeze(0)
                
          
            if local_indices.numel() == 0: 
                # we must fill the remaining layers from the cache.
                # since the generator manages the double buffers, we continue draining it.
                with Timer("fast_forward_fill", stats=stats):
                    for layer_id_t in range(layer_id + 2, config.num_hidden_layers):
                        # This fetches L(t) and triggers L(t+1)
                        k_next, v_next = next(cache_generator)
                        
                        retrieved_k_to_cache = blender.rescale(
                            k_next.squeeze(0), 
                            original_positions.squeeze(0), 
                            positions.squeeze(0)
                        ).unsqueeze(0)
                        
                        past_key_value.update(retrieved_k_to_cache, v_next, layer_id_t, {})
                return past_key_value

    return past_key_value







#wake up babe new generator_build_blent_cache just dropped


def layer_prefetch_generator5(fetch_plan, doc_ids, dtype, num_layers, stats):
    """
    Generator for Unified Blending (V5).
    Yields Layer 0 History first, then Layer 1, etc.
    This prevents the Sequence Length mismatch between L0 and L1.
    """
    return _build_layer_prefetch_iterator(fetch_plan, doc_ids, dtype, num_layers, stats, start_layer=0)


def generator_build_blent_cache5(
        model: "Model",
        cache_database: list, 
        doc_ids: torch.LongTensor,
        recompute_ratio: Union[float, Dict[int, float]],
        past_key_value: Optional[DynamicCache] = None,
        stats=None,
        **kwargs,
) -> DynamicCache:
    """
    Unified blending for LMCache Online (V5 - Fixed for Miss Corruption).
    Strategy: Compute Fresh first (to get valid Misses), then Overwrite Hits.
    """
    if stats is None: stats = {}
    assert doc_ids.shape[0] == 1
    doc_embeds = model.embed_tokens(doc_ids)
    
    if past_key_value is None:
        past_key_value = DynamicCache()
    
    past_seen_tokens = past_key_value.get_seq_length()
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + doc_embeds.shape[1],
        device=doc_ids.device, dtype=torch.long
    )
    
    hidden_states = doc_embeds
    position_embeddings = model.rotary_emb(hidden_states, cache_position.unsqueeze(0))
    
    is_lmcache_online = isinstance(recompute_ratio, dict)
    
    blender = CacheBlendImpl(recompute_ratio=0.0)
    blender.set_positional_encoder(partial(positional_encoder, model))
    blender.set_reverse_positional_encoder(partial(reverse_positional_encoder, model))
    blender.use_anti_piaffe = kwargs.get("use_piaffe", False)
    
    # Metadata setup
    valid_mask = torch.ones(doc_ids.shape, dtype=torch.bool, device="cpu")
    original_positions = torch.zeros(doc_ids.shape, dtype=torch.int64, device="cpu")
    positions = torch.arange(doc_ids.shape[-1], device=doc_ids.device).unsqueeze(0)
    
    chunk_boundaries = []
    layer_fetch_plan = []
    total_tokens = doc_ids.shape[1]
    
    with Timer("scan_cache_metadata", stats=stats):
        doc_idx = 0
        for (cid, cached_doc_ids, k_source, v_source, orig_pos) in cache_database:
            cached_n = cached_doc_ids.shape[0]
            chunk_ratio = recompute_ratio.get(cid, 0.0) if is_lmcache_online else recompute_ratio
            
            original_positions[0, doc_idx:doc_idx+cached_n] = (
                torch.arange(cached_n) if orig_pos is None else orig_pos
            )
            
            chunk_boundaries.append((doc_idx, doc_idx + cached_n, cid, chunk_ratio))
            layer_fetch_plan.append({
                "batch_idx": 0, "start": doc_idx, "end": doc_idx + cached_n,
                "k_source": k_source, "v_source": v_source
            })
            doc_idx += cached_n

    valid_mask = valid_mask.to(doc_ids.device, non_blocking=True)
    original_positions = original_positions.to(doc_ids.device, non_blocking=True)

    config = model.config
    num_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    
    kwargs.update({'num_q_heads': config.num_attention_heads, 'num_kv_heads': n_kv_heads, 'head_dim': head_dim})

    # Start the generator
    cache_generator = layer_prefetch_generator5(layer_fetch_plan, doc_ids, model.dtype, num_layers, stats)

    # Prefetch L0 immediately. We hold it in 'retrieved_k/v' but DO NOT inject it yet.
    retrieved_k_layer, retrieved_v_layer = next(cache_generator)

    selected_mask = None
    
    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------
    for layer_id, decoder_layer in enumerate(model.layers[:num_layers]):
        
        # 1. CLEAN FORWARD PASS
        # We run the layer without injecting cache first.
        # This ensures that "Miss" tokens attend to valid features, not zeros.
        with Timer(f"Decoding_L{layer_id}", stats=stats):
            causal_mask = _make_causal_mask(
                model, layer_id, doc_embeds, cache_position,
                past_key_value, None
            )
            
            layer_output = decoder_forward(
                decoder_layer,
                hidden_states,
                attention_mask=causal_mask,
                position_ids=cache_position.unsqueeze(0),
                past_key_value=past_key_value, # Appends Fresh KV here
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_output

        # 2. OVERWRITE HITS (The "Blend")
        # Now past_key_value contains [Old History + Fresh Current Chunk].
        # We want to replace the "Fresh" part with "Cached" for Hits.
        
        with Timer(f"Overwrite_L{layer_id}", stats=stats):
            # A. Extract the Fresh KV we just computed
            fresh_k_all, fresh_v_all = _cache_layer_kv(past_key_value, layer_id)
            
            # Identify the slice corresponding to the current document
            seq_len = doc_ids.shape[1]
            fresh_k_slice = fresh_k_all[:, :, -seq_len:, :] # [1, H, Seq, D]
            fresh_v_slice = fresh_v_all[:, :, -seq_len:, :]
            
            # B. Prepare the Cached KV (Un-rotate -> Re-rotate)
            rescaled_cached_k = blender.rescale(
                retrieved_k_layer.squeeze(0),
                original_positions.squeeze(0),
                positions.squeeze(0)
            ).unsqueeze(0)
            
            cached_v = retrieved_v_layer # No rotation needed for V
            
            # C. Construct the Final KV
            if layer_id == 0:
                
                final_k = fresh_k_slice.clone()
                final_v = fresh_v_slice.clone()
            else:
                # Layer 1+: Start with Cached (Default)
                final_k = rescaled_cached_k.clone()
                final_v = cached_v.clone()
                
                # Overwrite based on selection
                if layer_id == 1:
                    
                    
                    # Logic Check:
                    # 1. Loop L0 -> Forward L0 -> Blend L0 (Keep Fresh) -> Project L0->L1 -> Select L1 Tokens.
                    # 2. Loop L1 -> Forward L1 (using Fresh L0) -> Blend L1 (using Mask from prev step).
                    
                   
                    pass 

                if selected_mask is not None:
                    # selected_mask is [Seq]. Expand to [1, H, Seq, D]
                    # Overwrite selected (recompute) indices with Fresh data
                    final_k[:, :, selected_mask, :] = fresh_k_slice[:, :, selected_mask, :]
                    final_v[:, :, selected_mask, :] = fresh_v_slice[:, :, selected_mask, :]
                
                

            # D. Commit back to DynamicCache (In-place overwrite)
            fresh_k_all[:, :, -seq_len:, :] = final_k
            fresh_v_all[:, :, -seq_len:, :] = final_v

        # 3. PREPARE NEXT LAYER (Fetch & Select)
        if layer_id + 1 < num_layers:
            
            # A. Fetch L(i+1) Cached
            with Timer(f"retrieve_L{layer_id+1}", stats=stats):
                # Prefetch next layer for the next iteration
                retrieved_k_layer, retrieved_v_layer = next(cache_generator)

            # B. If we are at Layer 1, perform SELECTION logic for future layers
            if layer_id + 1 == 1:
                # We need "Fresh Key" for L1 to compare against "Cached Key L1".
                # We calculate it via projection from current hidden_states.
                with Timer(f"projections_L{layer_id}", stats=stats):
                    next_layer = model.layers[layer_id + 1]
                    normalized = next_layer.input_layernorm(hidden_states)
                    hidden_shape = (*hidden_states.shape[:-1], -1, next_layer.self_attn.head_dim)
                    fresh_q_next, fresh_k_next, fresh_v_next = proj(next_layer.self_attn, normalized, hidden_shape)
                    
                    cos, sin = position_embeddings
                    if not inplace_rope:
                        fresh_q_next, fresh_k_next = apply_rotary_pos_emb(fresh_q_next, fresh_k_next, cos, sin)
                    else:
                        fresh_q_next, fresh_k_next = apply_rotary_pos_emb_inplace(fresh_q_next, fresh_k_next, cos, sin)

                # Now select
                with Timer("select_recompute_tokens", stats=stats):
                    recompute_indices, _, _ = _g5_select_recompute_tokens(
                        fresh_k=fresh_k_next.squeeze(0),
                        retrieved_k=retrieved_k_layer.squeeze(0), # This is L1 Cached
                        valid_mask=valid_mask.squeeze(0),
                        original_positions=original_positions.squeeze(0),
                        positions=positions.squeeze(0),
                        chunk_boundaries=chunk_boundaries,
                        recompute_ratio=recompute_ratio,
                        blender=blender,
                        stats=stats
                    )
                
                selected_mask = torch.zeros(total_tokens, dtype=torch.bool, device=doc_ids.device)
                selected_mask[recompute_indices] = True
                if stats: stats["recomp_ids"] = len(recompute_indices)

    return past_key_value


def generator_build_blent_cache_unified(
        model: "Model",
        cache_database: list,
        doc_ids: torch.LongTensor,
        recompute_ratio: Union[float, Dict[int, float]],
        past_key_value: Optional[DynamicCache] = None,
        stats=None,
        **kwargs,
) -> DynamicCache:
    """
    Unified entrypoint for CacheBlend/ZCF and LMCache Online.
    - float recompute_ratio -> CacheBlend/ZCF behavior (gen4)
    - dict recompute_ratio  -> LMCache Online behavior (gen5)
    Keeps legacy generators intact for safety.
    """
    if isinstance(recompute_ratio, dict):
        return generator_build_blent_cache5(
            model,
            cache_database,
            doc_ids,
            recompute_ratio,
            past_key_value=past_key_value,
            stats=stats,
            **kwargs,
        )
    return generator_build_blent_cache4(
        model,
        cache_database,
        doc_ids,
        recompute_ratio,
        past_key_value=past_key_value,
        stats=stats,
        **kwargs,
    )


def _record_selection_diagnostics(
    stats: dict,
    selected_indices: List[int],
    diff_per_token: torch.Tensor,
) -> None:
    """Record overlap/diagnostic surrogates directly from selector internals.

    This is an internal proxy for token-change overlap when full baseline tensors
    are not serialized. It compares selected indices against oracle top-|S|
    indices by delta score and stores JSON-safe scalars/lists.
    """
    if stats is None:
        return
    try:
        sel = sorted({int(x) for x in selected_indices})
        stats["recompute_selected_indices"] = sel
        if len(sel) == 0:
            stats["oracle_changed_indices"] = []
            stats["heuristic_overlap_score"] = 0.0
            stats["per_layer_kl_divergence"] = [0.0]
            return

        k = min(len(sel), int(diff_per_token.shape[0]))
        oracle = torch.topk(diff_per_token.float(), k).indices.tolist()
        oracle = sorted({int(x) for x in oracle})
        stats["oracle_changed_indices"] = oracle

        sset = set(sel)
        oset = set(oracle)
        stats["heuristic_overlap_score"] = float(len(sset & oset) / max(1, len(oset)))

        # Single-value KL proxy between selected-mask and oracle-mask distributions.
        eps = 1e-8
        n = int(diff_per_token.shape[0])
        p = torch.full((n,), eps, dtype=torch.float32, device=diff_per_token.device)
        q = torch.full((n,), eps, dtype=torch.float32, device=diff_per_token.device)
        if oracle:
            p[oracle] = 1.0
        if sel:
            q[sel] = 1.0
        p = p / p.sum()
        q = q / q.sum()
        kl = torch.sum(p * torch.log((p + eps) / (q + eps))).item()
        stats["per_layer_kl_divergence"] = [float(kl)]
    except Exception:
        # Never fail core inference due to diagnostics.
        return


def _g5_select_recompute_tokens(
    fresh_k: torch.Tensor,
    retrieved_k: torch.Tensor,
    valid_mask: torch.Tensor,
    original_positions: torch.Tensor,
    positions: torch.Tensor,
    chunk_boundaries: list,
    recompute_ratio: Union[float, Dict[int, float]],
    blender: CacheBlendImpl,
    fresh_q: torch.Tensor = None,
    stats: dict = None
) -> Tuple[List[int], Optional[torch.Tensor], bool]:
    """
    Compute ΔK and select tokens to recompute.
    
    For LMCache Online:
    - valid_mask is always True (we have dummy KV for misses)
    - chunk_boundaries contains ratio info to identify misses
    """
    
    is_lmcache_online = isinstance(recompute_ratio, dict)
    device = fresh_k.device
    total_tokens = fresh_k.shape[1]
    
    # Early exit: "all misses" case for LMCache Online
    if is_lmcache_online:
        all_miss = all(ratio >= 1.0 for ratio in recompute_ratio.values())
        if all_miss:
            all_indices = list(range(total_tokens))
            if stats:
                stats["mode"] = "lmcache_online_all_miss"
                stats["num_miss_tokens"] = total_tokens
                stats["num_hit_tokens"] = 0
                stats["hit_budget"] = 0
                stats["num_selected"] = total_tokens
                stats["recompute_selected_indices"] = [int(x) for x in all_indices]
            return all_indices, None, True
    
    # Compute ΔK
    rescaled_k = blender.rescale32(retrieved_k, original_positions, positions)
    diff_per_token = torch.mean((fresh_k - rescaled_k) ** 2, dim=(0, 2))
    
    xattn_ids = None
    
    if not is_lmcache_online:
        
        # CACHEBLEND MODE
        
        num_valid = valid_mask.sum().item()
        budget = int(num_valid * recompute_ratio)
        
        if budget > 0:
            diff_per_token_masked = diff_per_token * valid_mask.to(device)
            top_indices = torch.topk(diff_per_token_masked, budget).indices
            top_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            top_mask[top_indices] = True
        else:
            top_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            top_indices = torch.tensor([], device=device, dtype=torch.long)
        
        selected_mask = (~valid_mask) | top_mask
        selected_indices = torch.where(selected_mask)[0].tolist()
        
        if blender.use_anti_piaffe and fresh_q is not None:
            xattn_ids = blender.aggregate_attention_topk(fresh_q, fresh_k, top_indices.tolist())
        
        if stats:
            stats["mode"] = "cacheblend"
            stats["global_budget"] = budget
            stats["num_selected"] = len(selected_indices)
            _record_selection_diagnostics(stats, selected_indices, diff_per_token_masked)
        
        return selected_indices, xattn_ids, False
    
    else:
        
        # LMCACHE ONLINE MODE (with at least some hits)
        
        miss_indices = []
        hit_indices = []
        hit_deltas = []
        hit_ratio = None
        
        for (start, end, cid, ratio) in chunk_boundaries:
            if ratio >= 1.0:
                # MISS: 100% recompute
                miss_indices.extend(range(start, end))
            else:
                # HIT: collect for R% selection
                hit_ratio = ratio  # All hits share same R
                for idx in range(start, end):
                    hit_indices.append(idx)
                    hit_deltas.append((idx, diff_per_token[idx].item()))
        
        # Select top R% of hits by ΔK
        selected_hit_indices = []
        if hit_deltas and hit_ratio is not None:
            hit_budget = int(len(hit_deltas) * hit_ratio)
            
            if hit_budget > 0:
                hit_deltas.sort(key=lambda x: x[1], reverse=True)
                selected_hit_indices = [x[0] for x in hit_deltas[:hit_budget]]
                top_indices = torch.tensor(selected_hit_indices, device=device)
            else:
                top_indices = torch.tensor([], device=device, dtype=torch.long)
        else:
            hit_budget = 0
            top_indices = torch.tensor([], device=device, dtype=torch.long)
        
        # Combine miss indices + selected hit indices
        selected_indices = sorted(set(miss_indices + selected_hit_indices))
        
        if blender.use_anti_piaffe and fresh_q is not None:
            xattn_ids = blender.aggregate_attention_topk(fresh_q, fresh_k, top_indices.tolist())
        
        if stats:
            stats["mode"] = "lmcache_online"
            stats["num_miss_tokens"] = len(miss_indices)
            stats["num_hit_tokens"] = len(hit_indices)
            stats["hit_budget"] = hit_budget
            stats["num_selected"] = len(selected_indices)
            _record_selection_diagnostics(stats, selected_indices, diff_per_token)
        
        return selected_indices, xattn_ids, False


def _g5_fast_path_no_recompute(model, cache_database, doc_ids, original_positions, positions, blender, config, stats):
    """Fast path when R=0 for all chunks."""
    
    sample_cws = cache_database[0][2]
    n_layers = len(sample_cws.states)
    total_seq_len = doc_ids.shape[1]
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads

    retrieved_k = torch.empty((n_layers, 1, n_kv_heads, total_seq_len, head_dim), device=doc_ids.device, dtype=model.dtype)
    retrieved_v = torch.empty((n_layers, 1, n_kv_heads, total_seq_len, head_dim), device=doc_ids.device, dtype=model.dtype)

    doc_idx = 0
    for (_, _, k_source, v_source, _) in cache_database:
        cached_n = len(k_source)
        k_layers, v_layers = [], []
        
        for layer_k, layer_v in k_source.states:
            kk = layer_k.to(doc_ids.device)
            vv = layer_v.to(doc_ids.device)
            if kk.ndim == 4:
                kk = kk.squeeze(0)
            if vv.ndim == 4:
                vv = vv.squeeze(0)
            k_layers.append(kk)
            v_layers.append(vv)

        k_block = torch.stack(k_layers, dim=0).unsqueeze(1)
        v_block = torch.stack(v_layers, dim=0).unsqueeze(1)
        
        retrieved_k[:, :, :, doc_idx:doc_idx+cached_n, :] = k_block
        retrieved_v[:, :, :, doc_idx:doc_idx+cached_n, :] = v_block
        doc_idx += cached_n

    past_key_value = DynamicCache()
    for layer_id in range(n_layers):
        k_rescaled = blender.rescale(
            retrieved_k[layer_id].squeeze(0),
            original_positions.squeeze(0),
            positions.squeeze(0)
        ).unsqueeze(0)
        past_key_value.update(k_rescaled, retrieved_v[layer_id], layer_id, {})

    return past_key_value





def _g5_select_recompute_tokens(
    fresh_k: torch.Tensor,
    retrieved_k: torch.Tensor,
    valid_mask: torch.Tensor,
    original_positions: torch.Tensor,
    positions: torch.Tensor,
    chunk_boundaries: list,
    recompute_ratio: Union[float, Dict[int, float]],
    blender: CacheBlendImpl,
    fresh_q: torch.Tensor = None,
    stats: dict = None
) -> Tuple[List[int], Optional[torch.Tensor], bool]:
    """
    Compute ΔK and select tokens to recompute.
    
    CacheBlend (float): Top R% globally.
    LMCache Online (dict): Per-chunk ratios, 1.0 = 100% for misses.
    
    Returns: (selected_indices, xattn_ids, all_recompute)
        - all_recompute: True if ALL tokens need recompute (all misses case)
    """
    
    is_lmcache_online = isinstance(recompute_ratio, dict)
    device = fresh_k.device
    total_tokens = fresh_k.shape[1]
    
    
    # Early exit: "all misses" case for LMCache Online
   
    if is_lmcache_online:
        all_miss = all(ratio >= 1.0 for ratio in recompute_ratio.values())
        if all_miss:
            all_indices = list(range(total_tokens))
            if stats:
                stats["mode"] = "lmcache_online_all_miss"
                stats["num_miss_tokens"] = total_tokens
                stats["num_hit_tokens"] = 0
                stats["hit_budget"] = 0
                stats["num_selected"] = total_tokens
                stats["recompute_selected_indices"] = [int(x) for x in all_indices]
            return all_indices, None, True
    
   
    # Normal path: compute ΔK and select
    
    rescaled_k = blender.rescale32(retrieved_k, original_positions, positions)
    
    diff_per_token = torch.mean((fresh_k - rescaled_k) ** 2, dim=(0, 2))
    diff_per_token_masked = diff_per_token * valid_mask.to(device)
    
    xattn_ids = None
    
    if not is_lmcache_online:
       
        # CACHEBLEND MODE
       
        num_valid = valid_mask.sum().item()
        budget = int(num_valid * recompute_ratio)
        
        if budget > 0:
            top_indices = torch.topk(diff_per_token_masked, budget).indices
            top_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            top_mask[top_indices] = True
        else:
            top_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            top_indices = torch.tensor([], device=device, dtype=torch.long)
        
        selected_mask = (~valid_mask) | top_mask
        selected_indices = torch.where(selected_mask)[0].tolist()
        
        if blender.use_anti_piaffe and fresh_q is not None:
            xattn_ids = blender.aggregate_attention_topk(fresh_q, fresh_k, top_indices.tolist())
        
        if stats:
            stats["mode"] = "cacheblend"
            stats["global_budget"] = budget
            stats["num_selected"] = len(selected_indices)
            _record_selection_diagnostics(stats, selected_indices, diff_per_token_masked)
        
        return selected_indices, xattn_ids, False
    
    else:
        
        # LMCACHE ONLINE MODE (with at least some hits)
        
        selected_indices = []
        hit_deltas = []
        
        for (start, end, cid, ratio) in chunk_boundaries:
            if ratio >= 1.0:
                selected_indices.extend(range(start, end))
            else:
                for idx in range(start, end):
                    if valid_mask[idx]:
                        hit_deltas.append((idx, diff_per_token[idx].item(), ratio))
        
        hit_budget = 0
        if hit_deltas:
            hit_ratio = hit_deltas[0][2]
            hit_budget = int(len(hit_deltas) * hit_ratio)
            
            if hit_budget > 0:
                hit_deltas.sort(key=lambda x: x[1], reverse=True)
                selected_indices.extend([x[0] for x in hit_deltas[:hit_budget]])
                top_indices = torch.tensor([x[0] for x in hit_deltas[:hit_budget]], device=device)
            else:
                top_indices = torch.tensor([], device=device, dtype=torch.long)
        else:
            top_indices = torch.tensor([], device=device, dtype=torch.long)
        
        selected_indices = sorted(set(selected_indices))
        
        if blender.use_anti_piaffe and fresh_q is not None:
            xattn_ids = blender.aggregate_attention_topk(fresh_q, fresh_k, top_indices.tolist())
        
        if stats:
            stats["mode"] = "lmcache_online"
            stats["num_miss_tokens"] = sum(end - start for start, end, _, r in chunk_boundaries if r >= 1.0)
            stats["num_hit_tokens"] = len(hit_deltas)
            stats["hit_budget"] = hit_budget
            stats["num_selected"] = len(selected_indices)
            _record_selection_diagnostics(stats, selected_indices, diff_per_token)
        
        return selected_indices, xattn_ids, False


def _g5_fast_path_no_recompute(model, cache_database, doc_ids, original_positions, positions, blender, config, stats):
    """Fast path when R=0 for all chunks."""
    
    sample_cws = cache_database[0][2]
    n_layers = len(sample_cws.states)
    total_seq_len = doc_ids.shape[1]
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads

    retrieved_k = torch.empty((n_layers, 1, n_kv_heads, total_seq_len, head_dim), device=doc_ids.device, dtype=model.dtype)
    retrieved_v = torch.empty((n_layers, 1, n_kv_heads, total_seq_len, head_dim), device=doc_ids.device, dtype=model.dtype)

    doc_idx = 0
    for (_, _, k_source, v_source, _) in cache_database:
        cached_n = len(k_source)
        k_layers, v_layers = [], []
        
        for layer_k, layer_v in k_source.states:
            kk = layer_k.to(doc_ids.device)
            vv = layer_v.to(doc_ids.device)
            if kk.ndim == 4:
                kk = kk.squeeze(0)
            if vv.ndim == 4:
                vv = vv.squeeze(0)
            k_layers.append(kk)
            v_layers.append(vv)

        k_block = torch.stack(k_layers, dim=0).unsqueeze(1)
        v_block = torch.stack(v_layers, dim=0).unsqueeze(1)
        
        retrieved_k[:, :, :, doc_idx:doc_idx+cached_n, :] = k_block
        retrieved_v[:, :, :, doc_idx:doc_idx+cached_n, :] = v_block
        doc_idx += cached_n

    past_key_value = DynamicCache()
    for layer_id in range(n_layers):
        k_rescaled = blender.rescale(
            retrieved_k[layer_id].squeeze(0),
            original_positions.squeeze(0),
            positions.squeeze(0)
        ).unsqueeze(0)
        past_key_value.update(k_rescaled, retrieved_v[layer_id], layer_id, {})

    return past_key_value



###############################################################################


def build_blent_cache(
        model: "Model",
        # note this is now either a dict or a list
        cache_database: dict[int, tuple[torch.LongTensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        doc_ids: torch.LongTensor,
        recompute_ratio: Union[float,dict[int,list[int]]],
        past_key_value: Optional[Cache] = None,
        stats = None,
        **kwargs,
        ) -> BaseModelOutputWithPast:
    assert doc_ids.shape[0] == 1, "Only batch size = 1 is supported"
    assert not model.gradient_checkpointing
    doc_embeds = model.embed_tokens(doc_ids)

    if past_key_value is None:  #not none in general since these are the caches of docs
        past_key_value = DynamicCache()

    past_seen_tokens = past_key_value.get_seq_length() if past_key_value is not None else 0
    cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + doc_embeds.shape[1], device=doc_ids.device, dtype=torch.long
            )
    if isdbg():logger.debug(f"Building blent cache {len(cache_database)=} {doc_ids.shape=} {past_seen_tokens=}, {cache_position=}")
    assert past_seen_tokens == 0, f"{past_seen_tokens=}"

    hidden_states = doc_embeds

    # create position embeddings to be shared across the decoder layers
    if isdbg():logger.debug(f"{hidden_states.shape=}, {cache_position.unsqueeze(0).shape=}")
    position_embeddings = model.rotary_emb(hidden_states, cache_position.unsqueeze(0))

    
    blender = CacheBlendImpl(recompute_ratio=recompute_ratio)
    blender.set_positional_encoder(partial(positional_encoder, model))
    blender.set_reverse_positional_encoder(partial(reverse_positional_encoder, model))
    valid_mask: torch.Tensor = torch.zeros_like(doc_ids, dtype=torch.bool,device="cpu")
    original_positions: torch.LongTensor = torch.zeros_like(doc_ids, dtype=torch.int64)
    positions: torch.LongTensor = torch.tensor([list(range(doc_ids.shape[-1])) for _ in range(doc_ids.shape[0])], device=doc_ids.device)
    query_start_loc: torch.LongTensor = torch.tensor([0, doc_ids.shape[-1]], device=doc_ids.device)
    
    if isdbg():logger.debug(f"@ddi blending. {doc_ids.shape=} {doc_ids=} {positions=} {query_start_loc=}")

    retrieved_k = None
    retrieved_v = None
    
    if stats is None: stats = {}
    
    #@ddi hack for now:
    if isinstance(cache_database,dict):
        assert False
            

    # @ddi this seems to to the following: for each element in the cache, check if there is a match (and do not stop after finding one?)
    if isdbg():logger.debug(f"Beginning the blending. {doc_ids.shape=} {valid_mask=}")

    '''
    ISSUE: this code assumes that a doc is unique, and does not even appear as sub-doc of a doc
    in my test case I can have  G H I A B C F G H I  J K
    THen GHI appears twice and it is used twice
    In general we need a more robust approach to this as in: we need to match not with a sliding window as done now
    but trying to match chunk by chunk
    FOr now, we can force docs to be unique
    '''
    tkn_to_chunk = []
    chunk_boundaries = []
    with Timer("scan cache",stats=stats):
        doc_idx = 0
        batch_idx = 0
        verb = False
        cid = 0
        for (h,cached_doc_ids, original_k, original_v,orig_pos) in cache_database:
            cached_n = cached_doc_ids.shape[0]
            #if isdbg():logger.debug(f"{doc_idx=} {cached_n=}")
            with Timer(f"valid_{h}",verbose=verb):
                assert not valid_mask[batch_idx,doc_idx:doc_idx+cached_n].any()
                valid_mask[batch_idx, doc_idx:doc_idx+cached_n] = True
            with Timer(f"pos_{h}",verbose=verb):
                if  orig_pos is None:
                    original_positions[batch_idx, doc_idx:doc_idx+cached_n] = torch.arange(cached_n)
                else:
                    #if isdbg():logger.debug(f"{orig_pos[:10]=} {orig_pos[-10:]=}")
                    original_positions[batch_idx, doc_idx:doc_idx+cached_n] = orig_pos.unsqueeze(0)  #need to be proper 2d tensor not ,x
                tkn_to_chunk+=[cid]*(cached_n)
                chunk_boundaries.append((doc_idx,doc_idx+cached_n))
                if isinstance(recompute_ratio,dict) and h in recompute_ratio:
                    rcmp= [x+doc_idx for x in recompute_ratio[h]]
                    recompute_ratio[h]=rcmp
            # the recomp-tokens are RELATIVE to the chunk
            with Timer(f"retrieve",verbose=verb,stats=stats):
                # preallocate once
                if retrieved_k is None:
                    n_layers = len(original_k)
                    retrieved_k = torch.empty(
                        (n_layers, doc_ids.shape[0], original_k[0].shape[-3], doc_ids.shape[1], original_k[0].shape[-1]),
                        device=doc_ids.device, dtype=model.dtype
                    )
                    retrieved_v = torch.empty(
                        (n_layers, doc_ids.shape[0], original_v[0].shape[-3], doc_ids.shape[1], original_v[0].shape[-1]),
                        device=doc_ids.device, dtype=model.dtype
                    )
                k_block = original_k
                v_block = original_v
                # stack into a single block
                # this was used when cachedb was populated with a list of caches, one entry per layer
                #k_block = torch.stack(original_k, dim=0)
                #v_block = torch.stack(original_v, dim=0)

                # assign in one go
                """
                print("#"*100)
                print(type(k_block))
                print(type(k_block[0]))
                print(len(k_block))
                print(k_block[0].shape)
                print("#"*100)
                """
                #a nasty patch to make the current zcf_v1 code work asap
                #please change the upstream code to use tensors consistently
                if isinstance(k_block, list):
                    k_block_tensor = torch.stack(k_block, dim=0)
                else:
                    k_block_tensor = k_block
                    
                if isinstance(v_block, list):
                    v_block_tensor = torch.stack(v_block, dim=0)
                else:
                    v_block_tensor = v_block
                
                retrieved_k[:, batch_idx, :, doc_idx:doc_idx + cached_n, :] = k_block_tensor
                retrieved_v[:, batch_idx, :, doc_idx:doc_idx + cached_n, :] = v_block_tensor
                #retrieved_k[:, batch_idx, :, doc_idx:doc_idx + cached_n, :] = k_block
                #retrieved_v[:, batch_idx, :, doc_idx:doc_idx + cached_n, :] = v_block
            doc_idx += cached_doc_ids.shape[0]
            cid+=1
            

    if isinstance(recompute_ratio,dict): # TODO: can this be wrong?
        if isdbg():logger.debug(f"@ddi final list of recomp tkns {recompute_ratio}")
        blender.recompute_ratio=[x for k in sorted(recompute_ratio.keys()) for x in recompute_ratio[k]] # values in dict to list
        if isdbg():logger.debug(f"@ddi final list of recomp tkns in blender {blender.recompute_ratio}")
    else:
        if isdbg():logger.debug(f"@ddi final  recomp ratio in blender {blender.recompute_ratio}")
    
    # Now valid mask has TRUE where we re-use and FALSE where we do not reuse
    assert valid_mask.all(), valid_mask
    valid_mask = valid_mask.to(doc_ids,non_blocking=True)
    #return
    fresh_q = None
    fresh_k = None
    fresh_v = None
    local_indices = None
    selected_idx = cache_position
    deselected_idx = None
    
    xattn_ids = None
    # for anti-piaffe
    if "use_piaffe" in kwargs:
        use_anti_piaffe = kwargs["use_piaffe"]
    else:
        use_anti_piaffe = False
    blender.use_anti_piaffe = use_anti_piaffe
    if use_anti_piaffe:
        blender.chunk_boundaries = chunk_boundaries

    config=model.config
    n_q_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_q_heads)
    head_dim = config.hidden_size // n_q_heads
    kwargs['num_q_heads']=n_q_heads
    kwargs['num_kv_heads']=n_kv_heads
    kwargs['head_dim']=head_dim

    for layer_id, decoder_layer in enumerate(model.layers[: model.config.num_hidden_layers]):
        selected_pos_embeds = position_embeddings if local_indices is None else (position_embeddings[0][:, local_indices], position_embeddings[1][:, local_indices])
        selected_doc_embeds = doc_embeds if local_indices is None else doc_embeds[:, local_indices]
        with Timer(f"Decoding",stats = stats):
            if (not use_anti_piaffe) or (xattn_ids is None):
                causal_mask = _make_causal_mask(model,
                        layer_id=layer_id,
                        input_tensor=selected_doc_embeds,
                        cache_position=selected_idx,
                        past_key_value=past_key_value,
                        last_positions_in_cache=deselected_idx
                        )
            else:
                if layer_id == 1:
                    causal_mask = build_chunked_causal_mask(
                        model=model,
                        layer_id=layer_id,
                        input_tensor=selected_doc_embeds,
                        cache_position=selected_idx,
                        past_key_value=past_key_value,
                        last_positions_in_cache=deselected_idx,
                        additional_ids=xattn_ids,
                        chunk_ids = torch.tensor(tkn_to_chunk)
                    )
            # the "fresh" are NONE at the beginning, so everything is computred from scratch
            layer_output = decoder_forward(decoder_layer,
                    hidden_states,
                    query_states=fresh_q,
                    key_states=fresh_k,
                    value_states=fresh_v,
                    attention_mask=causal_mask,
                    position_ids=selected_idx.unsqueeze(0),
                    past_key_value=past_key_value,
                    cache_position=selected_idx,
                    position_embeddings=selected_pos_embeds,
                    **kwargs,
                    )
        # this sleep speeds up execution probably by forcing npu not to batch
        # more requests (maybe tries to coalesce computing layer_output and using it)
        #time.sleep(0.001)
        if retrieved_k is not None and retrieved_v is not None and layer_id + 1 < model.config.num_hidden_layers:
            with Timer(f"projections+rope",stats=stats):
                next_decoder_layer = model.layers[layer_id+1]
                hidden_states = layer_output
                normalized_hidden_states = next_decoder_layer.input_layernorm(hidden_states)
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, next_decoder_layer.self_attn.head_dim)
                fresh_q,fresh_k,fresh_v=proj(next_decoder_layer.self_attn,normalized_hidden_states,hidden_shape)
                cos, sin = selected_pos_embeds
                # @ddi this shifts the freshq and fresh_k
                if not inplace_rope:
                    fresh_q, fresh_k = apply_rotary_pos_emb(fresh_q, fresh_k, cos, sin)
                else:
                    fresh_q, fresh_k = apply_rotary_pos_emb_inplace(fresh_q, fresh_k, cos, sin)
            
            '''
            FIXME
            THis is uncanny. If I print before blending, performance
            goes up 50%. Prolly CANN tries to coalesce too much
            '''
            #time.sleep(0.001)
            # @ddi: blend also shifts the re-used K
            with Timer(f"Blending",stats=stats):
                ret: BlendOutput = blender.blend(
                        layer_id=layer_id+1,
                        retrieved_k=retrieved_k[layer_id+1].squeeze(0),
                        retrieved_v=retrieved_v[layer_id+1].squeeze(0),
                        valid_mask=valid_mask.squeeze(0),
                        original_positions=original_positions.squeeze(0),
                        fresh_q=fresh_q.squeeze(0),
                        fresh_k=fresh_k.squeeze(0),
                        fresh_v=fresh_v.squeeze(0),
                        positions=positions.squeeze(0),
                        query_start_loc=query_start_loc,
                        token_dim=1,
                        )
                #stats["indices"] = blender.indexes_in_kv.tolist()
                stats["recomp_ids"] = len(blender.indexes_in_kv)

            local_indices = ret.local_indices
            if ret.query_start_loc is not None:
                hidden_states = hidden_states[:, local_indices]
                query_start_loc = query_start_loc
                selected_idx = cache_position[local_indices]
                valid_mask_tmp = valid_mask.clone().detach().reshape(-1)
                valid_mask_tmp[selected_idx] = False
                deselected_idx = torch.where(valid_mask_tmp)[0]
                # grab the cross_att ids
                xattn_ids = ret.xattn_ids
                # remove the local indices from the tkn_to_chunk  and re-add them at the end
                # so first identify the chunks corresponding to selected tkns
                # remove them from the original
                # add them back at the end
                tkn_to_chunk_prime = [x for x in tkn_to_chunk if x in selected_idx]
                tkn_to_chunk = [x for x in tkn_to_chunk if x not in selected_idx]
                tkn_to_chunk+=tkn_to_chunk_prime
                
                

            fresh_q = ret.q.unsqueeze(0)
            if deselected_idx is not None:
                fresh_k = ret.k[:, ret.local_indices, :].unsqueeze(0)
                fresh_v = ret.v[:, ret.local_indices, :].unsqueeze(0)
                to_cache_k = ret.k[:, deselected_idx, :].unsqueeze(0)
                to_cache_v = ret.v[:, deselected_idx, :].unsqueeze(0)
                if isdbg() and layer_id==1:logger.debug(f"{layer_id=} {fresh_k.shape=} {to_cache_k.shape=}")
                past_key_value.update(to_cache_k, to_cache_v, layer_id+1, {})
            else:  #@ddi recomp ratio is 1
                fresh_k = ret.k.unsqueeze(0)
                fresh_v = ret.v.unsqueeze(0)
            if local_indices.numel() == 0:  #@ddi recomp ratio is 0: just rescale K
                for layer_id_t in range(layer_id+2, model.config.num_hidden_layers):
                    retrieved_k_to_cache = blender.rescale(retrieved_k[layer_id_t].squeeze(0), original_positions.squeeze(0), positions.squeeze(0)).unsqueeze(0)
                    retrieved_v_to_cache = retrieved_v[layer_id_t]
                    past_key_value.update(retrieved_k_to_cache, retrieved_v_to_cache, layer_id_t, {})
                return past_key_value

    #hidden_states = model.norm(hidden_states)  #@ddi.left here but we don't need this
    return past_key_value

def build_chunked_causal_mask(
    input_tensor,
    model,
    cache_position,
    chunk_ids,            # 1D tensor of same length as all context. THis is in normal order.
    additional_ids=None,  # 1D tensor/list of allowed position ids for cross-chunk attention
    past_key_value=None,
    last_positions_in_cache=None,
    layer_id=0,
):
    if model.config._attn_implementation == "flash_attention_2":
        raise NotImplementedError
    if model.config._attn_implementation == "flex_attention":
        raise NotImplementedError
    using_static_cache = isinstance(past_key_value, StaticCache)
    using_sliding_window_cache = isinstance(past_key_value, SlidingWindowCache)
    if using_static_cache or using_sliding_window_cache:
        raise NotImplementedError# Chunk restriction: only allow within-chunk, or if key in additional_ids

    dev = input_tensor.device
    past_seen_tokens = past_key_value.get_seq_length(layer_id) if past_key_value is not None else 0
    dtype = input_tensor.dtype
    min_dtype =  torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = past_seen_tokens + sequence_length
    dev = input_tensor.device
    batch_size = input_tensor.shape[0]
    causal_mask = torch.full(
        (sequence_length, target_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=dev
    )
    if last_positions_in_cache is not None:
        positions_in_cache = torch.cat(
            (
                torch.arange(
                    past_seen_tokens - last_positions_in_cache.shape[0],
                    device=dev,
                ),
                last_positions_in_cache,
            ),
            dim=0,
        )
    else:
        positions_in_cache = torch.arange(
            past_seen_tokens, device=dev
        )
    # positions_in_cache are the NON recomputed ones
    # cache_position are the recomputed ones
    # positions are both. the NEW tokens are at the bottom
    # total_len = total number of tokens seen so far (cached + new)

    # 1️⃣ Build causal mask for new tokens (queries) against all tokens (keys)
    # positions_in_cache + cache_position are chronological, so this works
    if layer_id==100:print(f"{cache_position=} {cache_position.shape=}")
    if layer_id==100:print(f"{positions_in_cache=} {positions_in_cache.shape=}")
    tkn_ids = torch.cat((positions_in_cache, cache_position), dim=0)
    np.set_printoptions(threshold=np.inf, linewidth=200)
    if layer_id==100:print(f"{tkn_ids=} {tkn_ids.shape=}")
    N = len(chunk_ids)
    assert N == tkn_ids.numel()
    assert N == target_length
    R = len(cache_position)
    chunk_ids = chunk_ids.to(dev)
    causal = tkn_ids.unsqueeze(0)<=tkn_ids.unsqueeze(1) #N x N
    #print(np.array(causal.to(torch.int).cpu()))
    # NOTE: chunk ids is in natural order. it's not "scrambled" in cached and recomp
    chunk_ids_ = chunk_ids[tkn_ids]
    same_chunk = chunk_ids_.unsqueeze(0) ==  chunk_ids_.unsqueeze(1) # NxN
    if layer_id==100:print(np.array(chunk_ids_.cpu()))
    #if layer_id==1:print(f"{np.array(same_chunk.to(torch.int).cpu())=}")
    assert causal.shape[0] == N and causal.shape == same_chunk.shape
    final_mask = causal & same_chunk
    #if layer_id==1:print(f"{final_mask[-R:,22:].to(torch.int)=}")
    extra_mask = tkn_ids.unsqueeze(1) >= additional_ids.unsqueeze(0) # N x M
    if layer_id==100:print(np.array(additional_ids.cpu()))
    # find the indices that correspond to the extra
    extra_indices = (tkn_ids.unsqueeze(0) == additional_ids.unsqueeze(1)).nonzero(as_tuple=False)[:,1]
    final_mask[:,extra_indices]|= extra_mask
    final_mask = final_mask[-R:]  #take the part only for the recomp  #TODO: do this earlier
    mask = torch.full((R,N),min_dtype,device=dev, dtype=input_tensor.dtype)
    mask[final_mask] =0.0
    mask = mask[None,None,:,:].expand(batch_size,1,-1,-1)
    return mask

# the goal is to have a causal mask that only attends to a subset of the cross-chunk tokens
# this is
def __build_chunked_causal_mask(
    input_tensor,
    model,
    cache_position,
    chunk_ids,            # 1D tensor of same length as all context. THis is in normal order.
    additional_ids=None,  # 1D tensor/list of allowed position ids for cross-chunk attention
    past_key_value=None,
    last_positions_in_cache=None,
    layer_id=0,
):
    if model.config._attn_implementation == "flash_attention_2":
        raise NotImplementedError
    if model.config._attn_implementation == "flex_attention":
        raise NotImplementedError
    using_static_cache = isinstance(past_key_value, StaticCache)
    using_sliding_window_cache = isinstance(past_key_value, SlidingWindowCache)
    if using_static_cache or using_sliding_window_cache:
        raise NotImplementedError# Chunk restriction: only allow within-chunk, or if key in additional_ids
    
    past_seen_tokens = past_key_value.get_seq_length(layer_id) if past_key_value is not None else 0
    dtype = input_tensor.dtype
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = past_seen_tokens + sequence_length
    batch_size = input_tensor.shape[0]
    causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=input_tensor.device
            )
    if last_positions_in_cache is not None:
        positions_in_cache = torch.cat((torch.arange(past_seen_tokens-last_positions_in_cache.shape[0], device=cache_position.device), last_positions_in_cache), dim=0)
    else:
        positions_in_cache = torch.arange(past_seen_tokens, device=cache_position.device)
    #positions_in_cache are the NON recomputed ones
    #cache_position are the recomputed ones
    #positions are both. the NEW tokens are at the bottom
    # total_len = total number of tokens seen so far (cached + new)
    total_len = len(chunk_ids)
    recomp_len = len(cache_position)
    dev = input_tensor.device

    # 1️⃣ Build causal mask for new tokens (queries) against all tokens (keys)
    # positions_in_cache + cache_position are chronological, so this works
    positions = torch.cat((positions_in_cache, cache_position), dim=0)
    diagonal_attend_mask = positions > cache_position.reshape(-1, 1)
    # shape: [recomp_len, total_len]

    # 2️⃣ Build allowed mask based on chunk_ids and additional_ids
    key_chunks = chunk_ids.unsqueeze(0)    # [1, total_len]
    query_chunks = chunk_ids.unsqueeze(1)  # [total_len, 1]
    same_chunk = key_chunks == query_chunks  # [total_len, total_len]

    # By default, only same-chunk allowed
    allowed_mask_full = same_chunk.clone().to(dev)

    # Enable special cross-chunk if key ∈ additional_ids
    if additional_ids is not None:
        additional_ids = torch.tensor(additional_ids, device=dev)
        cross_chunk_allow = torch.isin(positions.squeeze(0), additional_ids)
        # Broadcast: allow these keys globally
        allowed_mask_full |= cross_chunk_allow.unsqueeze(0)

    # 3️⃣ Extract the part relevant for recomputed queries
    # We want only the rows corresponding to cache_position (the recomputed/new tokens)
    query_rows = cache_position
    allowed_mask = allowed_mask_full[query_rows]   # shape [recomp_len, total_len]

    # 4️⃣ Combine causal + allowed
    full_mask = diagonal_attend_mask & allowed_mask

    # 5️⃣ Convert to additive mask
    causal_mask = torch.full(
        (recomp_len, total_len), fill_value=min_dtype, dtype=dtype, device=dev
    )
    causal_mask = torch.where(full_mask, 0.0, causal_mask)

    # 6️⃣ Expand for batch
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    return causal_mask

def _build_chunked_causal_mask(
    input_tensor,
    model,
    cache_position,
    chunk_ids,            # 1D tensor of same length as all context. THis is in normal order.
    additional_ids=None,  # 1D tensor/list of allowed position ids for cross-chunk attention
    past_key_value=None,
    last_positions_in_cache=None,
    layer_id=0,
):
    if model.config._attn_implementation == "flash_attention_2":
        raise NotImplementedError
    if model.config._attn_implementation == "flex_attention":
        raise NotImplementedError
    using_static_cache = isinstance(past_key_value, StaticCache)
    using_sliding_window_cache = isinstance(past_key_value, SlidingWindowCache)
    if using_static_cache or using_sliding_window_cache:
        raise NotImplementedError# Chunk restriction: only allow within-chunk, or if key in additional_ids
    
    past_seen_tokens = past_key_value.get_seq_length(layer_id) if past_key_value is not None else 0
    dtype = input_tensor.dtype
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = past_seen_tokens + sequence_length
    batch_size = input_tensor.shape[0]
    causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=input_tensor.device
            )
    if last_positions_in_cache is not None:
        positions_in_cache = torch.cat((torch.arange(past_seen_tokens-last_positions_in_cache.shape[0], device=cache_position.device), last_positions_in_cache), dim=0)
    else:
        positions_in_cache = torch.arange(past_seen_tokens, device=cache_position.device)
    #positions_in_cache are the NON recomputed ones
    #cache_position are the recomputed ones
    #positions are both. the NEW tokens are at the bottom
    
    positions = torch.cat((positions_in_cache, cache_position), dim=0).reshape(1, -1)
    if isdbg():logger.debug(f"{layer_id=} {past_seen_tokens=}\n{last_positions_in_cache=} {len(last_positions_in_cache) if last_positions_in_cache is not None else 0=}\n{cache_position=} {len(cache_position) if cache_position is not None else  0=}\n{positions=} {len(positions) if positions is not None else 0=}\n{positions_in_cache=} {len(positions_in_cache) if positions_in_cache is not None else 0=}\n{sequence_length=} {target_length=} {additional_ids=} {chunk_ids.shape=}")
    
    # 1. compute the causal mask of the recomp tokens
    # 2. compute the FULL causal mask for ALL tokens
    # 3. compute the FU
    
    # causal mask for recomp tokens: [recomp_tokens x total_tokens]
    diagonal_attend_mask = positions > cache_position.reshape(-1, 1)

    dev = cache_position.device
    if additional_ids is not None:
        additional_ids = torch.tensor(additional_ids, device=dev)
        # For each key, mark if it's in the allowed set
        allowed_cross_chunk = torch.isin(positions.squeeze(0), additional_ids)
    else:  #unused for now
        assert additional_ids is not None
        #allow all when allowed is None
        allowed_cross_chunk = torch.ones_like(positions.squeeze(0), dtype=torch.bool)
        
       # Build per-token chunk indices
    key_chunks = chunk_ids.unsqueeze(0)   # [1, total_len]
    query_chunks = chunk_ids.unsqueeze(1) # [total_len, 1]

    # Allow if same chunk OR if key is in allowed list
    same_chunk = key_chunks == query_chunks
    allowed_mask = same_chunk.to(dev) | allowed_cross_chunk.unsqueeze(0).to(dev)
    # from the allowed mask, take the entries corresponding to the recomp tokens
    allowed_mask = allowed_mask[cache_position.to(dev)]
     # Combine with causal mask
    if isdbg():logger.debug(f'{sequence_length=}\n{additional_ids.shape=}\n{key_chunks.shape=}\n{query_chunks.shape=}\n{allowed_mask.shape=}\n{allowed_cross_chunk.shape=}\n{diagonal_attend_mask.shape=}\n{allowed_mask.shape=}' )
    full_mask = torch.logical_and(diagonal_attend_mask.to(dev),allowed_mask.to(dev))
    causal_mask = torch.where(full_mask, torch.zeros_like(causal_mask), causal_mask)

    # Final broadcast shape
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    return causal_mask

def _make_causal_mask(
        model: "Model",
        layer_id: int,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_value: Cache,
        last_positions_in_cache: Optional[torch.Tensor],
        ):
    if model.config._attn_implementation == "flash_attention_2":
        raise NotImplementedError
    if model.config._attn_implementation == "flex_attention":
        raise NotImplementedError
    using_static_cache = isinstance(past_key_value, StaticCache)
    using_sliding_window_cache = isinstance(past_key_value, SlidingWindowCache)
    if using_static_cache or using_sliding_window_cache:
        raise NotImplementedError

    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.
    past_seen_tokens = past_key_value.get_seq_length(layer_id) if past_key_value is not None else 0
    dtype = input_tensor.dtype
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = past_seen_tokens + sequence_length
    batch_size = input_tensor.shape[0]

    # Fast path: standard causal mask for prefill. Let SDPA handle causality.
    if os.getenv("CB_USE_SDPA", "1") == "1":
        if last_positions_in_cache is None and past_seen_tokens == 0:
            # cache_position should be contiguous 0..seq-1 in this case
            if cache_position is not None and cache_position.numel() == sequence_length:
                if cache_position[0].item() == 0 and cache_position[-1].item() == sequence_length - 1:
                    return None
    causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=input_tensor.device
            )
    if last_positions_in_cache is not None:
        positions_in_cache = torch.cat((torch.arange(past_seen_tokens-last_positions_in_cache.shape[0], device=cache_position.device), last_positions_in_cache), dim=0)
    else:
        positions_in_cache = torch.arange(past_seen_tokens, device=cache_position.device)
    '''
    last_positions_in_cache=None len(last_positions_in_cache) if last_positions_in_cache is not None else 0=0                                               | 1/3 [00:01<00:03,  1.76s/it]
    cache_position=tensor([   0,    1,    2,  ..., 6427, 6428, 6429], device='npu:0') len(cache_position) if cache_position is not None else  0=6430
    positions=tensor([[   0,    1,    2,  ..., 6427, 6428, 6429]], device='npu:0') len(positions) if positions is not None else 0=1
    positions_in_cache=tensor([], device='npu:0', dtype=torch.int64) len(positions_in_cache) if positions_in_cache is not None else 0=0
    sequence_length=6430 target_length=6430 additional_ids=None
    
    # so past seen tokens are the cached ones with positions marked in last_positions_in_cache
    # cache_position are the positions of the recomp tokens
    # positions are _all_ the position indices of the K/V matrices 
    # position in cache: at layer 0 they are EMPTY (past_seen_tokens is always 0). Then position_in_cache is same as last_positions_in_cache becasue past seen == 0 always for us
    layer_id=1 past_seen_tokens=5144
    last_positions_in_cache=tensor([   0,    1,    2,  ..., 6427, 6428, 6429], device='npu:0') len(last_positions_in_cache) if last_positions_in_cache is not None else 0=5144
    cache_position=tensor([  42,   43,   44,  ..., 6384, 6388, 6412], device='npu:0') len(cache_position) if cache_position is not None else  0=1286
    positions=tensor([[   0,    1,    2,  ..., 6384, 6388, 6412]], device='npu:0') len(positions) if positions is not None else 0=1
    positions_in_cache=tensor([   0,    1,    2,  ..., 6427, 6428, 6429], device='npu:0') len(positions_in_cache) if positions_in_cache is not None else 0=5144
    sequence_length=1286 target_length=6430 additional_ids=None
    
    layer_id=2 past_seen_tokens=5144
    last_positions_in_cache=tensor([   0,    1,    2,  ..., 6427, 6428, 6429], device='npu:0') len(last_positions_in_cache) if last_positions_in_cache is not None else 0=5144
    cache_position=tensor([  42,   43,   44,  ..., 6384, 6388, 6412], device='npu:0') len(cache_position) if cache_position is not None else  0=1286
    positions=tensor([[   0,    1,    2,  ..., 6384, 6388, 6412]], device='npu:0') len(positions) if positions is not None else 0=1
    positions_in_cache=tensor([   0,    1,    2,  ..., 6427, 6428, 6429], device='npu:0') len(positions_in_cache) if positions_in_cache is not None else 0=5144
    sequence_length=1286 target_length=6430 additional_ids=None
    
    layer_id=3 past_seen_tokens=5144
    last_positions_in_cache=tensor([   0,    1,    2,  ..., 6427, 6428, 6429], device='npu:0') len(last_positions_in_cache) if last_positions_in_cache is not None else 0=5144
    cache_position=tensor([  42,   43,   44,  ..., 6384, 6388, 6412], device='npu:0') len(cache_position) if cache_position is not None else  0=1286
    positions=tensor([[   0,    1,    2,  ..., 6384, 6388, 6412]], device='npu:0') len(positions) if positions is not None else 0=1
    positions_in_cache=tensor([   0,    1,    2,  ..., 6427, 6428, 6429], device='npu:0') len(positions_in_cache) if positions_in_cache is not None else 0=5144
    sequence_length=1286 target_length=6430 additional_ids=None
    '''    
        
        
    positions = torch.cat((positions_in_cache, cache_position), dim=0).reshape(1, -1)
    if isdbg():logger.debug(f"{layer_id=} {past_seen_tokens=}\n{last_positions_in_cache=} {len(last_positions_in_cache) if last_positions_in_cache is not None else 0=}\n{cache_position=} {len(cache_position) if cache_position is not None else  0=}\n{positions=} {len(positions) if positions is not None else 0=}\n{positions_in_cache=} {len(positions_in_cache) if positions_in_cache is not None else 0=}\n{sequence_length=} {target_length=}")
    diagonal_attend_mask = positions > cache_position.reshape(-1, 1)
    
    
    
    causal_mask *= diagonal_attend_mask
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    return causal_mask

def generate(llm: AutoModelForCausalLM, past_key_values: DynamicCache, tokenizer: PreTrainedTokenizer, prompt: str|torch.Tensor, max_length: int=20, device: str="npu"):
    """
    past_key_value will be modified
    """
    output = []
    with torch.no_grad():
        if isinstance(prompt, str):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
        else:
            input_ids = prompt

        for _ in range(max_length):
            outputs = llm(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
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
