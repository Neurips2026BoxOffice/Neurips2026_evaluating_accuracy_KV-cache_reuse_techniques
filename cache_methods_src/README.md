# Cache Methods Bundle

This bundle contains the shared cache-reuse method implementations and the
generic evaluation runners used by the BoxOffice and LongBench experiments.

## Contents

- `cachebend/cacheblend.py`
  - low-level cache blending utilities
- `cachebend/llm_docs.py`
  - blending kernels and cache composition helpers
- `cachebend/utils.py`
  - shared timers and evaluation utilities
- `cachebend/ncf/`
  - method managers and runtime helpers:
    - `cblend_cache_faster.py`
    - `cacheblend_warm.py`
    - `cc_cache.py`
    - `fusionrag.py`
    - `method_factory.py`
    - `onthefly_infusion.py`
    - `pseudo_LMCache_online_v4.py`
    - `warm_cache_builder.py`
    - `zcf_v10.py`
- `pilot_eval.py`
  - evaluation runner used by the BoxOffice regeneration pipeline
- `transformers_eval.py`
  - generic JSONL evaluation runner
- `launchers/run_transformers_eval.sh`
  - convenience launcher for `transformers_eval.py`

## Methods covered by this bundle

The surrounding bundles use this package to run:

- baseline / no-reuse paths
- CacheBlend variants
- FusionRAG-style query-guided variants
- CacheCraft / warm-cache variants
- LMCache-style online variants

## Basic usage

```bash
PYTHONPATH=. python transformers_eval.py \
  --method lmcache_online \
  --model-path /data/weights/llama3.1-8BI \
  --traces data/eval_traces.jsonl \
  --out results/lmcache_online_eval.json \
  --max-traces 50 \
  --recompute-ratio 0.15
```

Or with the launcher:

```bash
bash launchers/run_transformers_eval.sh \
  --method lmcache_online \
  --model-path /data/weights/llama3.1-8BI \
  --traces data/eval_traces.jsonl \
  --out results/lmcache_online_eval.json
```
