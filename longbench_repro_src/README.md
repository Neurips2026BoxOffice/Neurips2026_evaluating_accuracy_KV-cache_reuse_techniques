# LongBench Reproduction Bundle

This bundle contains the LongBench preparation and runtime pieces used by the
paper for the three QA datasets:

- `musique_lb`
- `2wiki_lb`
- `hotpotqa_lb`

It is meant to be used together with the methods bundle:

- sibling default: `../cache_methods_src`
- override: `CACHE_METHODS_SRC=/abs/path/to/cache_methods_src`

## Included data

The bundle ships the enriched datasets used for the paper under:

- `data/enriched/musique_lb.json`
- `data/enriched/2wikimultihopqa_lb.json`
- `data/enriched/hotpotqa_lb.json`

These `_lb.json` files contain the LongBench queries plus the golden chunk
metadata added during enrichment.

## Regenerating the enriched files

If you want to rebuild the enriched files from public sources:

```bash
bash scripts/regenerate_enriched_longbench.sh
```

This downloads:

- the LongBench QA subsets
- the corresponding original datasets with supporting evidence metadata

and then runs `longbench_scripts/enrich.py`.

## Running the methods

The main runtime entrypoints are:

- `runtime/run.sh`
- `runtime/run_intro.py`
- `scripts/run_longbench_pipeline.sh`

Example:

```bash
cd runtime
MODELS="/data/weights/llama3.1-8BI /data/weights/mistral-7B /data/weights/qwen3-8B" \
DATASETS="musique_lb 2wiki_lb hotpotqa_lb" \
GPUS="0 1" \
bash run.sh
```

There is also a top-level wrapper that wires in the methods bundle and the
enriched datasets automatically:

```bash
GPUS="0 1" bash scripts/run_longbench_pipeline.sh
```

To restrict the method grid, use the `METHODS` environment variable. For
example:

```bash
GPUS="0 1" METHODS="cb_k0 cb_k1 cb_k1q cb_kprompt" \
bash scripts/run_longbench_pipeline.sh
```

By default outputs are written to:

- `runtime/output/intro_<dataset>_<model>.json`

## Environment

This bundle assumes the environment used for the methods bundle. Install the
methods bundle requirements first, then also install:

```bash
pip install -r requirements.txt
```
