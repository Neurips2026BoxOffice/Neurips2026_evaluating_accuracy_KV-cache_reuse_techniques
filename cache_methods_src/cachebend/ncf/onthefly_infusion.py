"""
On-the-fly infusion variants for CacheBlend and FusionRAG.

Instead of loading pre-computed warm caches from .pt files, these managers
build infused KV caches at query time using top-K neighbors from a retrieval
index.  The infused caches then feed into the standard blending pipeline
with recompute_ratio > 0.

Usage:
    manager = CacheBlendOnTheFlyManager(device, R=0.1, top_k=10, llm=llm, tkn=tkn)
    manager.configure_query_neighbors(cid_to_neighbor_texts)
    output, used_cache, stats = manager.new_query(prompt_tensor)
"""

import gc
import inspect
import logging
from typing import Dict, List, Tuple

import torch
from functools import partial

from cachebend.utils import Timer
from cachebend.ncf.cutils import (
    CachedChunk,
    PosChunk,
    do_query_with_state,
    full_prefill_cache,
    split_prompt_for_warm_chunks,
)
from cachebend.ncf.cacheblend_warm import CacheBlendCache, CacheBlendCacheManager
from cachebend.ncf.fusionrag import (
    FusionRAGCachedChunk,
    FusionRAGCache,
    FusionRAGCacheManager,
    _is_zero_recompute_ratio,
)
from cachebend.llm_docs import positional_encoder, reverse_positional_encoder

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
cache_log = logging.getLogger("onthefly")
cache_log.setLevel(logging.INFO)


def _empty_npu_cache():
    if hasattr(torch, "npu"):
        try:
            torch.npu.empty_cache()
        except Exception:
            pass


def _npu_synchronize():
    if hasattr(torch, "npu"):
        try:
            torch.npu.synchronize()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Shared infusion helper
# ---------------------------------------------------------------------------


def _record_fallback(stats, kind: str, cid: str,
                     requested_k: int, reduced_k: int) -> None:
    """Append a fallback record to stats["infusion_fallback"].

    Keys:
        requested_k  — top_k the manager was configured with
        half_k       — reduced K used when retry-at-half succeeded
        half_cids    — list of CIDs that fell back to K=half
        isolated_cids — list of CIDs that fell back to K=0 (isolated)
    """
    if stats is None:
        return
    fb = stats.setdefault("infusion_fallback", {
        "requested_k": requested_k,
        "half_cids": [],
        "isolated_cids": [],
    })
    if kind == "half":
        fb["half_cids"].append(cid)
        fb["half_k"] = reduced_k
    else:
        fb["isolated_cids"].append(cid)


def _infuse_with_fallback(
    infuser: "OnTheFlyInfuser",
    chunk_tokens: torch.Tensor,
    neighbor_texts: List[str],
    top_k: int,
    cid: str,
    stats: dict = None,
    fallback: bool = True,
):
    """Infuse one chunk.

    If ``fallback`` is True (default): escalate on OOM through
        1. ``infuse_chunk(top_k)``                     — full K
        2. ``infuse_chunk(max(1, top_k // 2))``        — half K
        3. ``infuse_chunk_isolated()``                 — K=0
    Every successful fallback records its CID in
    ``stats["infusion_fallback"]``.

    If ``fallback`` is False: attempt only the full-K infusion; on OOM
    the RuntimeError is re-raised so the caller can mark the entire
    query as failed (used by exp/offline when running with replay
    support).
    """
    want_infuse = bool(neighbor_texts) and top_k > 0
    try:
        if want_infuse:
            infuser.top_k = top_k
            return infuser.infuse_chunk(chunk_tokens, neighbor_texts)
        return infuser.infuse_chunk_isolated(chunk_tokens)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        _empty_npu_cache()
        if not fallback:
            # Let the caller decide what to do — typically: mark the
            # whole query as failed and move on.
            raise

    half_k = max(1, top_k // 2)
    if want_infuse and half_k < top_k:
        cache_log.warning(
            f"OOM on infusion for CID {cid} at K={top_k}; "
            f"retrying with K={half_k}."
        )
        try:
            infuser.top_k = half_k
            result = infuser.infuse_chunk(chunk_tokens, neighbor_texts[:half_k])
            _record_fallback(stats, "half", cid, top_k, half_k)
            return result
        except RuntimeError as exc2:
            if "out of memory" not in str(exc2).lower():
                raise
            _empty_npu_cache()
            cache_log.warning(
                f"OOM again for CID {cid} at K={half_k}; "
                "falling back to isolated."
            )
    else:
        cache_log.warning(
            f"OOM on infusion for CID {cid}, falling back to isolated."
        )

    result = infuser.infuse_chunk_isolated(chunk_tokens)
    _record_fallback(stats, "isolated", cid, top_k, 0)
    return result


class OnTheFlyInfuser:
    """Build infused KV caches on-the-fly for individual chunks.

    For each chunk, builds an input sequence::

        [prefix <sep> neighbor_1 <sep> ... <sep> neighbor_K <sep> target_chunk <sep> dummy_query]

    forwards it through the model, and extracts the KV slice that
    corresponds to the target chunk.  The returned (states, start, end)
    can be wrapped in a ``CachedChunk`` whose RoPE positions match
    ``(start, end)`` inside the infusion context — exactly the format
    expected by the blending kernel.
    """

    def __init__(self, llm, tokenizer, device, top_k: int):
        self.llm = llm
        self.tokenizer = tokenizer
        self.device = device
        self.top_k = top_k

        # Pre-compute separator, prefix, and dummy query token IDs.
        sep_token = tokenizer.sep_token or tokenizer.eos_token
        self.sep_ids: List[int] = tokenizer(
            sep_token, add_special_tokens=False
        ).input_ids
        if not self.sep_ids:
            self.sep_ids = [tokenizer.eos_token_id]

        prefix_text = "Answer the question based on the following text."
        self.prefix_ids: List[int] = tokenizer(
            prefix_text, add_special_tokens=False
        ).input_ids

        dummy_query_text = "X"
        self.query_ids: List[int] = tokenizer(
            dummy_query_text, add_special_tokens=False
        ).input_ids

    # ------------------------------------------------------------------ #

    def infuse_chunk(
        self,
        chunk_tokens: torch.Tensor,
        neighbor_texts: List[str],
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int, int]:
        """Forward ``[prefix + neighbors + chunk + query]`` and extract
        the chunk's KV states.

        Returns
        -------
        states : list[(K_cpu, V_cpu)]  per layer
        chunk_start, chunk_end : int
            Positions of the target chunk inside the infusion sequence
            (used as RoPE origin for the blending kernel).
        """
        neighbors = neighbor_texts[: self.top_k]

        # Build token-ID sequence.
        input_seq: List[int] = list(self.prefix_ids)
        for text in neighbors:
            nbr_ids = self.tokenizer(text, add_special_tokens=False).input_ids
            input_seq.extend(self.sep_ids)
            input_seq.extend(nbr_ids)
        input_seq.extend(self.sep_ids)

        chunk_ids = chunk_tokens.tolist()
        chunk_start = len(input_seq)
        input_seq.extend(chunk_ids)
        chunk_end = len(input_seq)

        input_seq.extend(self.sep_ids)
        input_seq.extend(self.query_ids)

        return self._forward_and_slice(input_seq, chunk_start, chunk_end)

    def infuse_chunk_isolated(
        self,
        chunk_tokens: torch.Tensor,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int, int]:
        """Encode chunk in isolation (K=0 case)."""
        input_seq = chunk_tokens.tolist()
        return self._forward_and_slice(input_seq, 0, len(input_seq))

    # ------------------------------------------------------------------ #

    def _forward_and_slice(
        self,
        input_seq: List[int],
        start: int,
        end: int,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int, int]:
        input_tensor = torch.tensor(
            [input_seq], dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            outputs = self.llm.model(
                input_ids=input_tensor,
                use_cache=True,
            )

        past = outputs.past_key_values
        num_layers = len(past)

        states: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            k_slice = past[layer_idx][0][0, :, start:end, :]
            v_slice = past[layer_idx][1][0, :, start:end, :]
            k_cpu = torch.empty_like(k_slice, device="cpu", pin_memory=True)
            v_cpu = torch.empty_like(v_slice, device="cpu", pin_memory=True)
            k_cpu.copy_(k_slice, non_blocking=True)
            v_cpu.copy_(v_slice, non_blocking=True)
            states.append((k_cpu, v_cpu))

        _npu_synchronize()
        del outputs, past, input_tensor
        _empty_npu_cache()

        return states, start, end


# ---------------------------------------------------------------------------
# CacheBlend on-the-fly
# ---------------------------------------------------------------------------

class CacheBlendOnTheFlyManager(CacheBlendCacheManager):
    """CacheBlend with on-the-fly infusion.

    Call ``configure_query_neighbors`` before each ``new_query`` to
    provide the per-chunk neighbour texts.  The manager will:

    1. Build infused ``CachedChunk`` objects (one forward per chunk).
    2. Feed them into the standard ``reuse_cache`` → blending pipeline
       with the configured ``recompute_ratio``.
    """

    def __init__(
        self,
        device: str,
        R: float,
        top_k: int,
        llm=None,
        tkn=None,
        stub_mode: bool = False,
        loading_mode: str = "generator",
    ):
        cache = CacheBlendCache(R=R)
        super().__init__(
            device=device,
            cache=cache,
            llm=llm,
            tkn=tkn,
            stub_mode=stub_mode,
            loading_mode=loading_mode,
        )
        self.top_k = top_k
        self._chunk_neighbors: Dict[str, List[str]] = {}
        self._infuser = OnTheFlyInfuser(llm, tkn, device, top_k)
        cache_log.info(
            f"CacheBlendOnTheFlyManager ready (R={R}, top_k={top_k})"
        )

    # -- public API --------------------------------------------------------

    def configure_query_neighbors(
        self,
        chunk_cid_to_neighbor_texts: Dict[str, List[str]],
    ):
        """Set per-chunk neighbour info for the *next* ``new_query`` call.

        Parameters
        ----------
        chunk_cid_to_neighbor_texts : dict
            Maps chunk CID (str) → list of neighbour text strings.
        """
        self._chunk_neighbors = chunk_cid_to_neighbor_texts

    def build_infused_caches(
        self,
        chunks: List[PosChunk],
        stats=None,
    ) -> Dict[str, CachedChunk]:
        """Build infused ``CachedChunk`` objects for *chunks*.

        Skips chunks that are already present in ``self.cache.chunk_to_cchunk``.

        Returns a dict ``{cid: CachedChunk}`` that can be injected into
        the cache via :meth:`inject_prebuilt_caches`.
        """
        infused: Dict[str, CachedChunk] = {}
        for chunk in chunks:
            cid = str(chunk.cid)
            if cid in self.cache.chunk_to_cchunk:
                continue  # already populated (e.g. pre-injected)
            neighbor_texts = self._chunk_neighbors.get(cid, [])

            states, start, end = _infuse_with_fallback(
                self._infuser, chunk.tokens_(), neighbor_texts,
                top_k=self.top_k, cid=cid, stats=stats,
                fallback=getattr(self, "oom_fallback", True),
            )

            cchunk = CachedChunk(chunk.tokens_(), start, end, states)
            infused[str(cchunk.cid)] = cchunk

        return infused

    def inject_prebuilt_caches(self, cchunks: Dict[str, CachedChunk]):
        """Load pre-built ``CachedChunk`` objects into the live cache."""
        self.cache.chunk_to_cchunk.update(cchunks)

    # -- overrides ---------------------------------------------------------

    @torch.no_grad()
    def new_query(self, prompt, force_full=False, snapshot_path=None):
        self.n_q += 1
        stats: dict = {}

        chunks = self.split_prompt_chunks(prompt)
        chunks, query = chunks[:-1], chunks[-1].tokens
        context_length = sum(len(c) for c in chunks)

        if force_full:
            if self.stub_mode:
                return "STUB_FULL", False, stats
            cache, _ = full_prefill_cache(
                chunks, self.llm, output_attentions=False
            )
            return (
                do_query_with_state(
                    self.llm,
                    self.tokenizer,
                    cache,
                    query,
                    output_attentions=False,
                    stats=stats,
                ),
                False,
                stats,
            )

        # 1. Build infused caches on-the-fly
        with Timer("onthefly_infusion", stats):
            infused = self.build_infused_caches(chunks, stats)
            self.inject_prebuilt_caches(infused)

        stats["mode"] = "cacheblend_onthefly"
        stats["top_k_neighbors"] = self.top_k
        stats["infused_chunks"] = len(infused)
        stats["token_composition"] = {
            "hit_tokens": context_length,
            "miss_tokens": 0,
            "hit_ratio": 1.0,
            "total_tokens": context_length,
            "hit_chunks": len(chunks),
            "miss_chunks": 0,
            "total_chunks": len(chunks),
        }

        # 2. Blend (re-compute a fraction of tokens from the infused cache)
        with Timer("reuse_cache", stats):
            cache = self.cache.reuse_cache(
                chunks,
                self.llm,
                self.tokenizer,
                stats,
                stub_mode=self.stub_mode,
                loading_mode=self.loading_mode,
            )

        # 3. Generate
        with Timer("query", stats):
            if cache:
                try:
                    return (
                        do_query_with_state(
                            self.llm,
                            self.tokenizer,
                            cache,
                            query,
                            output_attentions=False,
                            stats=stats,
                        ),
                        True,
                        stats,
                    )
                finally:
                    _empty_npu_cache()
                    gc.collect()
            else:
                cache, _ = full_prefill_cache(
                    chunks, self.llm, output_attentions=False
                )
                return (
                    do_query_with_state(
                        self.llm,
                        self.tokenizer,
                        cache,
                        query,
                        output_attentions=False,
                        stats=stats,
                    ),
                    False,
                    stats,
                )

    def begin_fresh_query(self):
        self.cache = CacheBlendCache(R=self.cache.recomp_ratio)
        self._chunk_neighbors = {}


# ---------------------------------------------------------------------------
# Prompt-conditioned variant: a fixed deterministic prompt as the K=1
# neighbour for every chunk.  Useful as a baseline that decouples "having
# something to attend to" from "having a query-relevant retrieved chunk".
# ---------------------------------------------------------------------------

# Generic context-grounded QA instructions, ~480 tokens under the
# llama / mistral / qwen tokenizers we use in this project. Caller can
# override via the manager's ``prompt_text`` constructor arg if a
# different policy text is desired.
PROMPT_NEIGHBOR_TEXT = (
    "You are an expert reading-comprehension assistant. Your job is to "
    "answer questions using only the information that appears in the "
    "context passages provided alongside each question. The passages may "
    "come from many domains — encyclopedia articles, technical documents, "
    "news stories, conversational dialogues, scientific papers — and may "
    "be of varying length and style.\n\n"
    "Follow these rules when producing every answer:\n\n"
    "1. Ground each answer in the context. Do not draw on memorised facts "
    "or assumptions not directly supported by the provided passages. If "
    "the passages do not contain enough information, say so rather than "
    "guess.\n\n"
    "2. Output the shortest faithful answer. For most questions the right "
    "answer is a single word, a short noun phrase, a name, a date, a "
    "quantity, or a brief excerpt from the context. Do not restate the "
    "question, do not begin with phrases such as \"According to the "
    "context\", and do not add explanatory commentary unless explicitly "
    "asked.\n\n"
    "3. Match the surface form of the question. If asked for a year, give "
    "a year; if asked for a role, give the role; if yes/no, answer "
    "\"Yes\" or \"No\"; if \"How many\", give a number. Do not translate "
    "proper nouns or paraphrase quoted strings unless the question "
    "demands it.\n\n"
    "4. Treat distractor passages with care. Multi-hop questions "
    "deliberately include passages that mention the same entities for "
    "unrelated reasons. Verify that the entity you cite actually "
    "satisfies the predicate stated in the question; do not pick the "
    "first matching name you see.\n\n"
    "5. Combine evidence cleanly when the question is multi-hop. Identify "
    "the bridge entity or shared fact that links one passage to another, "
    "chain the inferences, and state only the final answer. Do not "
    "narrate the reasoning.\n\n"
    "6. Be deterministic. Given the same context and question, produce "
    "the same answer. Do not introduce stylistic variation, do not "
    "reformat numbers or dates, do not paraphrase exact quotations from "
    "the passages.\n\n"
    "Correctness is scored with strict token-level metrics against a "
    "reference answer. Padded or verbose answers are penalised. The right "
    "entity surrounded by extra text scores lower than the bare entity. "
    "Aim for the smallest substring of the context that fully and "
    "unambiguously answers the question."
)


def build_prompt_neighbor_dict(
    chunk_cids,
    prompt_text: str = PROMPT_NEIGHBOR_TEXT,
) -> Dict[str, List[str]]:
    """Return ``{str(cid): [prompt_text]}`` for each cid — the
    neighbour-dict shape that ``CacheBlendOnTheFlyManager.configure_query_neighbors``
    expects when you want the same deterministic prompt as the K=1
    neighbour for every chunk.

    Use this directly with the regular CacheBlend / FusionRAG on-the-fly
    managers, OR use :class:`CacheBlendPromptOnTheFlyManager` which wires
    it up automatically.
    """
    return {str(cid): [prompt_text] for cid in chunk_cids}


class CacheBlendPromptOnTheFlyManager(CacheBlendOnTheFlyManager):
    """CacheBlend with a fixed deterministic prompt as the K=1 neighbour
    for every chunk.

    Functionally equivalent to constructing :class:`CacheBlendOnTheFlyManager`
    with ``top_k=1`` and calling
    ``configure_query_neighbors(build_prompt_neighbor_dict(cids))``
    before every query — but the wiring is automatic, so the call site
    matches the simpler ``CacheBlendCacheManager``-style flow:

        mgr = CacheBlendPromptOnTheFlyManager(device, R=0.15, llm=..., tkn=...)
        out, _, _ = mgr.new_query(prompt_tensor)

    The prompt defaults to :data:`PROMPT_NEIGHBOR_TEXT` (generic
    context-grounded QA instructions, ~480 tokens). Override at
    construction time to use a different policy.
    """

    def __init__(
        self,
        device: str,
        R: float,
        llm=None,
        tkn=None,
        stub_mode: bool = False,
        loading_mode: str = "generator",
        prompt_text: str = PROMPT_NEIGHBOR_TEXT,
    ):
        super().__init__(
            device=device, R=R, top_k=1,
            llm=llm, tkn=tkn,
            stub_mode=stub_mode, loading_mode=loading_mode,
        )
        self._prompt_text = prompt_text
        cache_log.info(
            f"CacheBlendPromptOnTheFlyManager ready (R={R}, top_k=1, "
            f"prompt={len(prompt_text)} chars)"
        )

    def configure_query_neighbors(
        self,
        chunk_cid_to_neighbor_texts: Dict[str, List[str]] = None,
    ):
        """No-op — this manager always uses the configured prompt as the
        K=1 neighbour for every chunk, regardless of any user-supplied
        dict. Accepts the argument for call-site compatibility with the
        parent class, but logs a debug warning if a non-empty dict is
        passed (helps catch wiring mistakes)."""
        if chunk_cid_to_neighbor_texts:
            cache_log.debug(
                "CacheBlendPromptOnTheFlyManager: ignoring user-supplied "
                f"neighbour dict ({len(chunk_cid_to_neighbor_texts)} entries); "
                "the manager uses its configured prompt for every chunk."
            )

    def build_infused_caches(self, chunks, stats=None):
        # Auto-inject the prompt as the K=1 neighbour for every chunk
        # the parent is about to infuse.
        self._chunk_neighbors = build_prompt_neighbor_dict(
            (c.cid for c in chunks), prompt_text=self._prompt_text,
        )
        return super().build_infused_caches(chunks, stats=stats)


# ---------------------------------------------------------------------------
# FusionRAG on-the-fly
# ---------------------------------------------------------------------------

class FusionRAGOnTheFlyManager(FusionRAGCacheManager):
    """FusionRAG with on-the-fly infusion + query-guided selection + blending.

    Same contract as :class:`CacheBlendOnTheFlyManager`: call
    ``configure_query_neighbors`` before each ``new_query``.
    """

    def __init__(
        self,
        device: str,
        R: float,
        top_k: int,
        llm=None,
        tkn=None,
        stub_mode: bool = False,
        loading_mode: str = "generator",
    ):
        cache = FusionRAGCache(R=R)
        super().__init__(
            device=device,
            cache=cache,
            llm=llm,
            tkn=tkn,
            stub_mode=stub_mode,
            loading_mode=loading_mode,
        )
        self.top_k = top_k
        self._chunk_neighbors: Dict[str, List[str]] = {}
        self._infuser = OnTheFlyInfuser(llm, tkn, device, top_k)
        cache_log.info(
            f"FusionRAGOnTheFlyManager ready (R={R}, top_k={top_k})"
        )

    # -- public API --------------------------------------------------------

    def configure_query_neighbors(
        self,
        chunk_cid_to_neighbor_texts: Dict[str, List[str]],
    ):
        self._chunk_neighbors = chunk_cid_to_neighbor_texts

    def build_infused_caches(
        self,
        chunks: List[PosChunk],
        stats=None,
    ) -> Dict[str, FusionRAGCachedChunk]:
        """Build infused ``FusionRAGCachedChunk`` objects (with final-layer key).

        Skips chunks already present in ``self.cache.chunk_to_cchunk``.
        """
        infused: Dict[str, FusionRAGCachedChunk] = {}
        for chunk in chunks:
            cid = str(chunk.cid)
            if cid in self.cache.chunk_to_cchunk:
                continue  # already populated (e.g. pre-injected)
            neighbor_texts = self._chunk_neighbors.get(cid, [])

            states, start, end = _infuse_with_fallback(
                self._infuser, chunk.tokens_(), neighbor_texts,
                top_k=self.top_k, cid=cid, stats=stats,
                fallback=getattr(self, "oom_fallback", True),
            )

            final_layer_key = states[-1][0] if states else None
            cchunk = FusionRAGCachedChunk(
                tokens=chunk.tokens_(),
                start=start,
                end=end,
                states=states,
                final_layer_key=final_layer_key,
            )
            infused[str(cchunk.cid)] = cchunk

        return infused

    def inject_prebuilt_caches(
        self,
        cchunks: Dict[str, FusionRAGCachedChunk],
    ):
        self.cache.chunk_to_cchunk.update(cchunks)

    # -- overrides ---------------------------------------------------------

    @torch.no_grad()
    def new_query(self, prompt, force_full=False, snapshot_path=None):
        self.n_q += 1
        stats: dict = {}

        chunks = self.split_prompt_chunks(prompt)
        chunks, query = chunks[:-1], chunks[-1].tokens
        context_length = sum(len(c) for c in chunks)

        if force_full:
            if self.stub_mode:
                return "STUB_FULL", False, stats
            cache, _ = full_prefill_cache(
                chunks, self.llm, output_attentions=False
            )
            return (
                do_query_with_state(
                    self.llm,
                    self.tokenizer,
                    cache,
                    query,
                    output_attentions=False,
                    stats=stats,
                ),
                False,
                stats,
            )

        # 1. Build infused caches on-the-fly
        with Timer("onthefly_infusion", stats):
            infused = self.build_infused_caches(chunks, stats)
            self.inject_prebuilt_caches(infused)

        stats["mode"] = "fusionrag_onthefly"
        stats["top_k_neighbors"] = self.top_k
        stats["infused_chunks"] = len(infused)
        stats["token_composition"] = {
            "hit_tokens": context_length,
            "miss_tokens": 0,
            "hit_ratio": 1.0,
            "total_tokens": context_length,
            "hit_chunks": len(chunks),
            "miss_chunks": 0,
            "total_chunks": len(chunks),
        }

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
            # 2. Query-guided token selection (FusionRAG)
            query_final_q = self._compute_query_final_layer_q(
                query, context_length, stats
            )
            (
                chunk_final_keys,
                chunk_lengths,
                chunk_original_position_ranges,
            ) = self.cache.get_final_layer_keys_and_lengths(chunks)

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

        # 3. Blend
        with Timer("reuse_cache", stats):
            cache = self.cache.reuse_cache(
                chunks,
                self.llm,
                self.tokenizer,
                stats,
                recompute_arg,
                stub_mode=self.stub_mode,
                loading_mode=self.loading_mode,
            )

        # 4. Generate
        with Timer("query_generation", stats):
            if cache:
                try:
                    return (
                        do_query_with_state(
                            self.llm,
                            self.tokenizer,
                            cache,
                            query,
                            output_attentions=False,
                            stats=stats,
                        ),
                        True,
                        stats,
                    )
                finally:
                    _empty_npu_cache()
                    gc.collect()
            else:
                cache, _ = full_prefill_cache(
                    chunks, self.llm, output_attentions=False
                )
                return (
                    do_query_with_state(
                        self.llm,
                        self.tokenizer,
                        cache,
                        query,
                        output_attentions=False,
                        stats=stats,
                    ),
                    False,
                    stats,
                )

    def begin_fresh_query(self):
        self.cache = FusionRAGCache(R=self.cache.recomp_ratio)
        self._chunk_neighbors = {}
