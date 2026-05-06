import hashlib
import inspect
import torch
from typing import List, Dict, Tuple
from transformers import (
    AutoTokenizer,
    DynamicCache,
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedTokenizer,
)
import os


def resolve_torch_dtype(dtype_name: str):
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(dtype_name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype '{dtype_name}'. Expected one of {sorted(mapping.keys())}.")
    return mapping[key]


def hash_tensors(tensors, algo: str = "sha256") -> str:
    """Compute a reproducible hash for a list of torch.Tensors."""
    h = hashlib.new(algo)

    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("All elements must be torch.Tensor")

        arr = t.detach().cpu().contiguous().numpy()

        # include metadata (shape + dtype)
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())

    return h.hexdigest()[:10]

def hash_strings(strs:List[str],algo:str ='sha256')->str:
    h = hashlib.new(algo)

    for s in strs:
        if not all(c in "0123456789abcdefABCDEF" for c in s):
            raise ValueError(f"Invalid hex string: {s}")
        h.update(s.encode())

    return h.hexdigest()[:10]


def chash(tokens: List[torch.Tensor]):
    return hash_tensors(tokens)


def dynamic_cache_num_layers(cache: DynamicCache) -> int:
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return len(cache)


def dynamic_cache_layer_kv(cache: DynamicCache, layer_idx: int):
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

def chunks_from_tokenss(
        tokens: torch.Tensor, sep: torch.Tensor
) -> List:  #of poschunk
    """
    Split `tokens` on subsequence `sep`, drop `sep` from output,
    and return chunks with positions relative to the cleaned sequence.
    """
    assert tokens.dim() == 1

    n, m = tokens.size(0), sep.size(0)
    result: List[PosChunk] = []

    if n < m:
        return [PosChunk(tokens.clone(), 0, n)]

    # Step 1: find all sep matches
    windows = tokens.unfold(0, m, 1)  # (n-m+1, m)
    matches = (windows == sep).all(dim=1)  # (n-m+1,)
    split_points = torch.nonzero(matches, as_tuple=False).squeeze(1).tolist()

    # Step 2: iterate and track offset in CLEANED sequence
    clean_offset = 0
    end = 0
    for idx in split_points:
        if end < idx:
            chunk = tokens[end:idx].clone()
            result.append(PosChunk(chunk, clean_offset, clean_offset + len(chunk)))
            clean_offset += len(chunk)
        end = idx + m
    if end < n:
        chunk = tokens[end:n].clone()
        result.append(PosChunk(chunk, clean_offset, clean_offset + len(chunk)))

    return result


def _find_token_subsequence(full_ids: List[int], sub_ids: List[int]) -> int:
    n = len(full_ids)
    m = len(sub_ids)
    if m == 0 or m > n:
        return -1
    for i in range(0, n - m + 1):
        if full_ids[i : i + m] == sub_ids:
            return i
    return -1


def _chat_user_content_start(tokenizer) -> int:
    cached = getattr(tokenizer, "_cachebend_user_content_start", None)
    if isinstance(cached, int) and cached >= 0:
        return cached

    marker = "CB_SYSTEM_SPLIT_SENTINEL_9f7e2c41"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": marker},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    full_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    marker_ids = tokenizer(marker, add_special_tokens=False).input_ids
    start = _find_token_subsequence(full_ids, marker_ids)
    if start < 0:
        start = 0
    setattr(tokenizer, "_cachebend_user_content_start", start)
    return start


def split_prompt_for_warm_chunks(
    tokens: torch.Tensor,
    sep: torch.Tensor,
    tokenizer,
) -> List["PosChunk"]:
    """
    Split warm-method prompts into:
    1) one leading prompt chunk containing the chat/system wrapper plus the
       fixed instruction prefix
    2) dataset chunks separated by <DSEP>
    3) the trailing query chunk

    This matches the old-src behavior where the prompt side is a single
    non-dataset chunk that can miss on query 1.
    """
    content_start = _chat_user_content_start(tokenizer)
    if content_start <= 0 or content_start >= int(tokens.numel()):
        return chunks_from_tokenss(tokens, sep)

    system_tokens = tokens[:content_start].clone()
    content_tokens = tokens[content_start:].clone()
    content_chunks = chunks_from_tokenss(content_tokens, sep)
    if not content_chunks:
        return [PosChunk(system_tokens, 0, len(system_tokens))]

    first_content = content_chunks[0]
    merged_prefix = torch.cat((system_tokens, first_content.tokens.clone()))
    out = [PosChunk(merged_prefix, 0, len(merged_prefix))]
    offset = len(system_tokens)
    for chunk in content_chunks[1:]:
        out.append(PosChunk(chunk.tokens.clone(), offset + chunk.start, offset + chunk.end))
    return out


class Chunk:
    def __init__(self, tokens: torch.Tensor):
        self.cid = chash(tokens)
        self.tokens = tokens

    def __eq__(self, other):
        return type(other) == type(self) and self.cid == other.cid

    def __repr__(self):
        return str(self.cid)

    def __str__(self):
        return str(self.cid)

class PosChunk(Chunk):
    def __init__(self, tokens: torch.Tensor, start, end):
        super().__init__(tokens)
        self.start = start
        self.end = end
        # chunk_log.debug(f"New chunk. {tokens=} {tokenizer.decode(tokens)=} {self.cid=} {start=} {end=}")

    def tokens_(self)->torch.Tensor:
        return self.tokens #[self.start:self.end]
    def __repr__(self):
        return str(f"{self.cid=} [{self.start}:{self.end})")

    def __str__(self):
        return str(f"{self.cid} [{self.start}:{self.end})")

    def __hash__(self):
        return self.cid

    def __len__(self):
        # this cannot be len(self.tokens): for Link0 we move start past the removed
        return self.end - self.start
class CachedChunk(PosChunk):
    def __init__(self,tokens:torch.Tensor,start:int,end:int,states:List[Tuple[torch.tensor,torch.tensor]]):
        super().__init__(tokens,start,end)
        self.states = states
        self._pinned_states = None
        
    ##################################################
    #new addition from Sam
    
    def get_layer(self, layer_idx: int, kv_idx: int, device=None):
        """
        Returns the CPU tensor for a specific layer.
        layer_idx: Layer number (0 to N-1)
        kv_idx: 0 for Key, 1 for Value
        device: Ignored here  want the CPU tensor so the generator handles the copy
        """
        # self.states[layer_idx] is tuple (K, V)
        tensor = self.states[layer_idx][kv_idx]
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
    
    
    """
    def get_layer(self, layer_idx: int, kv_idx: int, device: torch.device):
        
   
        #IMPOrtant!!!----> self.states structure is  [layer][0=k, 1=v]
    
        tensor = self.states[layer_idx][kv_idx]
        
        #for debug purposes
        print(f"DEBUG GET: L{layer_idx} TensorPtr: {tensor.data_ptr()}")
        #return tensor.to(device, non_blocking=True)
        return tensor
        """
    #######################################

    def _state(self, i, device):
        return [state[i].to(device, non_blocking=True) for state in self.states]

    def __state(self, i, device):
        # stack into one tensor on CPU first
        cpu_block = torch.stack([s[i] for s in self.states])  # shape [N, ...]
        cpu_block = cpu_block.pin_memory()  # pin once

        # single async transfer
        dev_block = torch.empty_like(cpu_block, device=device)
        dev_block.copy_(cpu_block, non_blocking=True)
        return dev_block
    
    def state(self, i, device):
        # assume self.states[j][i] is a pinned CPU tensor
        states_i = [s[i] for s in self.states]

        # Preallocate contiguous destination tensor on device
        dev_block = torch.empty(len(states_i), *states_i[0].shape,
                                dtype=states_i[0].dtype,
                                device=device)

        # Perform async copy directly for each slice
        for j, src in enumerate(states_i):
            dev_block[j].copy_(src, non_blocking=True)

        return dev_block


    def ks(self,device=None):
        return self.state(0,device)

    def vs(self,device):
        return self.state(1,device)


mcid_base_hash = chash([torch.tensor(12344321)])
ccid_base_hash = chash([torch.tensor(56788765)])

import random
import string

def generate_triplets(n):
    # NOTE: use unique chars so that there are no chances of the same triplet appearing across different doc boundarie
    # this is __required__ since the llmdoc will scan ids to detect chunks
    # we want to avoid that  A B C F is seen as "A B C" but  also ad "B C F"
    # F G A A B C
    letters = list(string.ascii_letters)
    n_atoms = 10
    E =  ["{} {} {}".format(letters[i], letters[i+1], letters[i+2]) for i in range(0,n_atoms * 3,3)]
    print("Generated E:", E)


    lst = [random.sample(E, 5) for _ in range(n)]
    print(f"{len(lst)=}")

    return lst

def parse_npu_process_info(npu_id, chip_id):
    import subprocess,re
    try:
        output = subprocess.check_output(['npu-smi', 'info'], text=True)
        pattern = re.compile( r'\|\s*(\d+)\s+(\d+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|')

        # Loop through each line and apply regex
        for line in output.splitlines():
            match = pattern.match(line)
            if match:
                npu, chip = int(match.group(1)), int(match.group(2))
                if npu == npu_id and chip == chip_id:
                    process_id = match.group(3).strip()
                    process_name = match.group(4).strip()
                    mem_mb = match.group(5).strip()
                    return {
                            "npu": npu,
                            "chip": chip,
                            "pid": process_id,
                            "name": process_name,
                            "memory_mb": int(mem_mb)
                            }

        return None  # Not found
    except subprocess.CalledProcessError as e:
        print("Error running npu-smi:", e)
        return None

def get_mem(npu_id, chip_id=0):
    try:
        return parse_npu_process_info(npu_id,chip_id)['memory_mb']
    except:
        return -1
def get_tensors():
    import gc
    import torch

    counter = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == 'npu':
                key = (tuple(obj.shape), str(obj.dtype))
                counter[key] = counter.get(key, 0) + 1
        except Exception:
            pass

    print("NPU tensor summary:")
    total_bytes = 0
    for (shape, dtype), count in counter.items():
        try:
            # Get element size
            element_size = torch.empty((), dtype=getattr(torch, dtype.split('.')[1])).element_size()
            # Total number of elements in one tensor
            numel = torch.tensor(shape).prod().item()
            # Total size in bytes
            size_bytes = count * numel * element_size
            size_mb = size_bytes / 1e6
            total_bytes += size_bytes
            print(f"{count:5d} tensor(s) of shape={shape}, dtype={dtype} → {size_mb:.2f} MB total")
        except Exception:
            print(f"{count:5d} tensor(s) of shape={shape}, dtype={dtype} → [Size calc failed]")

    print(f"~{total_bytes / 1e6:.2f} MB total on NPU")

def to_str_prompt(tokenizer,ascii_sep, lis) -> str:
    full_str = ascii_sep.join([x for x in lis])

    # if "llama" in model_path and False:
    if True:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_str},
        ]
        try:
            full_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except:   # in case disable thinking throws a tantrum but it should not
            full_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    else:
        messages=[{
            "role": "system",
            "content": "You must answer questions using only the provided documents."
            "Give the briefest possible answer. Do not repeat the question, add commentary, or explain."
            "If the documents lack the answer, reply only with: 'Not found.'"
        },
        {
            "role": "user",
            "content": "Here are the documents:"+full_str
        }
        ]
        full_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return full_str

def build_model(model_path):
    llm = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ) 
    llm.eval() 
    print(f"cutils: model built from {model_path=} with NORMAL attention (might be a problem for cachecraft)")
    return llm
    config = AutoConfig.from_pretrained(model_path)
    config.sliding_window = None
    config.attn_implementation = "eager"  # to have attns  and KV caches
    config.use_flash_attn = False        # <- this is the right flag for Qwen3
    config.use_sdpa = False              # optional, depending on HF version
    llm = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.bfloat16
    )  
    # , attn_implementation="eager")
    #llm=llm.to(torch.bfloat16)
    print(f"cutils: model built from {model_path=} with eager attention")
    return llm

def build_model_eager(model_path):
    config = AutoConfig.from_pretrained(model_path)
    config.sliding_window = None
    config._attn_implementation = "eager"  # to have attns
    llm = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.bfloat16,
    )  # , attn_implementation="eager")
    llm.eval()
    print(f"cutils: model built from {model_path=} with EAGER attention (might be a lil' slower)")
    return llm


def build_model_sdpa(model_path, torch_dtype: str = "bfloat16"):
    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "sdpa"
    dtype = resolve_torch_dtype(torch_dtype)
    llm = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=dtype
    )
    llm.eval()
    print(f"cutils: model built from {model_path=} with SDPA attention and dtype={dtype}")
    return llm
            
    

def build_tokenizer(model_path):
    
    special_tokens_dict = {"sep_token": '<DSEP>'} #this adds sep_token ascii and sep_token_id tokenized
    tokenizer_kwargs = {}
    model_path_l = str(model_path).lower()
    # Newer Mistral tokenizers expose a regex fix flag. When available, use it so
    # standalone tokenization and in-prompt tokenization remain consistent.
    if "mistral" in model_path_l:
        tokenizer_kwargs["fix_mistral_regex"] = True
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    assert num_added
    if tokenizer.pad_token is None:
        # use eos so we do not need to resize
        num_added = tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        assert not num_added
    # don't resize: we are **not** using this separator in the prompt that is passed to the LLM
    # this is just used in the str prompt
    # self.model.resize_token_embeddings(len(self.tokenizer))
    if tokenizer.bos_token is None:
        tokenizer.bos_token = ' '
        tokenizer.bos_token_id = tokenizer.encode(tokenizer.bos_token)[0]
        # don't resize: we are **not** using this separator in the prompt that is passed to the LLM
    #model.config.bos_token_id = tok.bos_token_id  # not needed
    print(f"cutils: tokenizer built from {model_path=}. Added DSEP")
    return tokenizer

@torch.no_grad()
def full_prefill_cache(chunks, llm,output_attentions):
    input_ids = torch.tensor(
        [t for c in chunks for t in c.tokens],
        dtype=torch.int64,
        device=llm.device,
    )
    forward_kwargs = {}
    try:
        if "logits_to_keep" in inspect.signature(llm.forward).parameters:
            # Qwen3 uses logits_to_keep=0 to mean "full sequence logits".
            # Cache prefills only need past_key_values, so keep just one step.
            forward_kwargs["logits_to_keep"] = 1
    except Exception:
        pass
    out = llm(
        input_ids.unsqueeze(0),  # add the batch dimension
        use_cache=True,
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
        **forward_kwargs,
    )
    return out.past_key_values, out

@torch.no_grad()
def do_query_with_state(
    llm, tokenizer,state: DynamicCache, query: torch.Tensor, output_attentions: bool,stats:Dict, max_length = 50
) -> str:
    # Stop conditions: EOS OR a "\n\n" sequence appears in the
    # accumulating decoded text — the latter mirrors vLLM's
    # `stop=["\n\n"]` (used by the v8 gate's T1a/T1b), so this path's
    # predictions are directly comparable to the gate's.
    output = []
    accum  = ""
    input_ids = query.to(llm.device).unsqueeze(0)


    for i in range(max_length):
        outputs = llm(
            input_ids=input_ids,
            past_key_values=state,
            use_cache=True,
            output_attentions=output_attentions,
        )
        logits = outputs.logits
        state = outputs.past_key_values

        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        decoded_token = tokenizer.decode(
            next_token_id, skip_special_tokens=True
        )
        output.append(decoded_token)
        accum += decoded_token

        input_ids = next_token_id.unsqueeze(0)

        # Capture first-step per-layer attention vectors for telemetry (JSON-serializable).
        if output_attentions and i == 0 and stats is not None:
            attn_vecs = []
            atts = getattr(outputs, "attentions", None)
            if atts:
                for layer_attn in atts:
                    if layer_attn is None:
                        continue
                    # expected shape: [bsz, heads, q_len, kv_len]
                    a = layer_attn.detach().float().cpu()
                    if a.dim() >= 4:
                        # average batch/head/query -> per-key vector
                        v = a.mean(dim=(0, 1, 2))
                    elif a.dim() == 3:
                        v = a.mean(dim=(0, 1))
                    elif a.dim() == 2:
                        v = a.mean(dim=0)
                    else:
                        v = a.reshape(-1)
                    attn_vecs.append(v.tolist())
            if attn_vecs:
                stats["attention_tensors"] = attn_vecs

        # Optionally stop on end-of-text token
        if next_token_id.item() == tokenizer.eos_token_id:
            if i == 0:
                print("EOS at token 0 usually means you have not added BOS")
            break
        # Stop on "\n\n" (vLLM-equivalent stop=["\n\n"])
        if "\n\n" in accum:
            break
    return "".join(output)

def lcs_length(A, B):
    """
    Compute length of Longest Common Subsequence (LCS) between A and B.
    Dynamic programming O(len(A)*len(B)).
    """
    m, n = len(A), len(B)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m):
        for j in range(n):
            if A[i] == B[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

def best_order_match(A, B_lists):
    """
    Given A and list of lists B_lists, return the B with the best order-preserving score.
    """
    best_score = -1
    best_B = None
    results = []

    for idx, B in enumerate(B_lists):
        overlap = set(A) & set(B)
        if not overlap:
            score = 0
        else:
            lcs = lcs_length(A, B)
            score = lcs / len(overlap)
        results.append((idx, score))
        if score > best_score:
            best_score = score
            best_B = B

    return best_B, results


# longest PREFIX of MC in SEQ. also returns # elem and length (useful tokens and total to compute waste)

def longest_prefix(SEQ, MC, respect_order=True):
    SEQ_pos = {x: i for i, x in enumerate(SEQ)}
    overlap_found = False
    last_overlap_idx = -1
    last_seq_pos = -1  # for order checking

    for i, x in enumerate(MC):
        if x in SEQ_pos:
            pos = SEQ_pos[x]
            # Check order constraint
            if respect_order and pos < last_seq_pos:
                break  # violates SEQ order → stop prefix here
            overlap_found = True
            last_overlap_idx = i
            last_seq_pos = pos
        elif overlap_found:
            last_overlap_idx = i  # still extend prefix if overlap seen before

    if last_overlap_idx == -1:
        return 0, 0  # no overlap found

    prefix = MC[:last_overlap_idx + 1]
    num_in_seq = sum(x in SEQ_pos for x in prefix)
    return num_in_seq, len(prefix)


def prefix_info_order_free(S, Q,Q_set):
    """Return best prefix info ignoring order of Q."""
    best_covered = set()
    covered = set()
    best_len = 0

    for i, x in enumerate(S):
        if x in Q_set:
            covered.add(x)
        if covered and len(covered) > len(best_covered):
            best_covered = covered.copy()
            best_len = i + 1

    if not best_covered:
        return set(), 0, 0
    waste = best_len - len(best_covered)
    return best_covered, waste, best_len


def prefix_info_order_sensitive(S, Q, Q_set):
    """Return best prefix info requiring order to match Q."""
    Q_index = {x: i for i, x in enumerate(Q)}
    best_covered = set()
    covered = set()
    last_q_pos = -1
    best_len = 0

    for i, x in enumerate(S):
        if x in Q_set:
            pos = Q_index[x]
            if pos > last_q_pos:
                covered.add(x)
                last_q_pos = pos
        if covered and len(covered) > len(best_covered):
            best_covered = covered.copy()
            best_len = i + 1

    if not best_covered:
        return set(), 0, 0
    waste = best_len - len(best_covered)
    return best_covered, waste, best_len


def stub_manual_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # ===== 1. Load model =====
    model_name = "meta-llama/Llama-3-8B"  # or your checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.eval()

    #  Prepare input =====
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    #Prefill manually, layer by layer =====
    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids)
        # LLaMA rotary embeddings are applied internally in each layer

        # Containers for inspection
        all_kv = []             # list of (key, value) per layer
        selected_attn = {}      # store attention maps from specific layers
        inspect_layers = [0, 5, 10]  # choose which layers to inspect

        for i, layer in enumerate(model.model.layers):
            out = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_value=None,
                output_attentions=(i in inspect_layers),
                use_cache=True,
            )

            hidden_states = out[0]
            past_kv = out[1]  # (key, value)
            all_kv.append(past_kv)

            if i in inspect_layers:
                attn = out[2]  # attention matrix
                selected_attn[i] = attn.detach().cpu()
                print(f"Layer {i} attention shape: {attn.shape}")

            print(f"Layer {i} done | hidden shape: {hidden_states.shape}")

        # Final normalization + LM head
        hidden_states = model.model.norm(hidden_states)
        logits = model.lm_head(hidden_states)

        print("Prefill complete.")



def save_snapshot(path_prefix, method_name, cache, query_ids, chunks_metadata):
    """
    Saves the KV cache and Query to disk for offline attention analysis.
    Handles NPU->CPU transfer, detaching, and fp16 conversion to save space.
    """
    try:
        
        save_dir = os.path.dirname(path_prefix)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        filename = f"{path_prefix}_{method_name}.pt"
        # print(f"[{method_name}] SNAPSHOT: Preparing to save to {filename}...")

        #Normalize Cache to a Standard List of CPU Tensors
        #Target Structure: List[Tuple(K_tensor, V_tensor)] per layer
        layers_data = []
        
        #Handle DynamicCache 
        if hasattr(cache, 'key_cache'):
            num_layers = len(cache.key_cache)
            for i in range(num_layers):
                ####
                k = cache.key_cache[i].detach().cpu().half()
                v = cache.value_cache[i].detach().cpu().half()
                layers_data.append((k, v))
        
       
        elif isinstance(cache, (list, tuple)):
            for layer in cache:
                k = layer[0].detach().cpu().half()
                v = layer[1].detach().cpu().half()
                layers_data.append((k, v))
                
        #Prep Payload
        payload = {
            "method": method_name,
            "query_ids": query_ids.detach().cpu(),  # The tokens of the question in the example
            "cache_layers": layers_data,            # The KV context
            "metadata": chunks_metadata             # Info about CIDs/Text for visualization
        }
        
        torch.save(payload, filename)
        # print(f"[{method_name}] SNAPSHOT: Saved to {filename}")
        
    except Exception as e:
        print(f"[{method_name}] SNAPSHOT ERROR: {e}")
