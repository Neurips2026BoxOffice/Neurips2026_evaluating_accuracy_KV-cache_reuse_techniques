# Anonymous Submission Code

This repository contains the code and data bundles used for the paper's
cache-reuse experiments on BoxOffice and LongBench-QA.

The release is organized as three sibling bundles:

- `cache_methods_src/`
  - shared method implementations and evaluation runners
- `boxoffice_repro_src/`
  - BoxOffice dataset release, regeneration code, and BoxOffice runtime
- `longbench_repro_src/`
  - LongBench enrichment assets and LongBench runtime

Figure-generation code is intentionally not included in this release yet.

## What is included

- the shared implementations for the cache-reuse methods used in the paper
- the released BoxOffice seeds for seeds `7, 11, 13, 17, 19, 23, 29, 31, 47, 73`
- the code to reproduce the verified BoxOffice seeds
- the code to regenerate BoxOffice warm similarity metadata from scratch
- the code to build the BoxOffice semantic pool and run the BoxOffice experiments
- the enriched LongBench QA files used in the paper
- the code to rebuild the enriched LongBench files from public sources
- the code to run the LongBench experiments on the enriched files

## Models used

Generation / evaluation models used by the default launchers:

- `llama3.1-8BI`
- `mistral`
- `qwen3-8B`

BoxOffice filtering defaults:

- `qwen3_8B`
- `llama3.1-8BI`
- `mistral`

Warm-retrieval / semantic-pool embedding model:

- `e5-base-v2`

The scripts use local model paths by default, for example:

```bash
/data/weights/llama3.1-8BI
/data/weights/mistral
/data/weights/qwen3-8B
/data/weights/e5-base-v2
```

Override those defaults with the environment variables shown in the examples
below.

## Repository layout

```text
repo/
  README.md
  .gitignore
  scripts/
    reproduce_boxoffice_seeds.sh
    regenerate_boxoffice_seed_from_scratch.sh
    run_boxoffice.sh
    regenerate_longbench.sh
    run_longbench.sh
  cache_methods_src/
  boxoffice_repro_src/
  longbench_repro_src/
```

## Environment setup

Create a Python environment and install the bundle requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r cache_methods_src/requirements.txt
pip install -r boxoffice_repro_src/requirements.txt
pip install -r longbench_repro_src/requirements.txt
```

The code expects an accelerator-backed runtime with the required model weights
available locally.

For this anonymous release, the hardware-vendor-specific accelerator backend is
not pinned in the top-level requirements files. Install the backend package
that matches your platform separately if needed.

## Quick start

Reproduce the released BoxOffice seeds using the bundled warm metadata:

```bash
bash scripts/reproduce_boxoffice_seeds.sh
```

Regenerate one BoxOffice seed from scratch, including warm similarity rebuild:

```bash
bash scripts/regenerate_boxoffice_seed_from_scratch.sh 7
```

Run the BoxOffice experiments:

```bash
GPUS="0 1" \
MODELS="/data/weights/llama3.1-8BI /data/weights/mistral /data/weights/qwen3-8B" \
bash scripts/run_boxoffice.sh
```

Rebuild the enriched LongBench files from public sources:

```bash
bash scripts/regenerate_longbench.sh
```

Run the LongBench experiments:

```bash
GPUS="0 1" \
MODELS="/data/weights/llama3.1-8BI /data/weights/mistral /data/weights/qwen3-8B" \
DATASETS="musique_lb 2wiki_lb hotpotqa_lb" \
bash scripts/run_longbench.sh
```

## BoxOffice notes

The BoxOffice bundle contains both:

- the released seed files under `boxoffice_repro_src/seeds/`
- the regeneration pipeline used to reproduce them

The default verified path reuses the bundled warm similarity metadata. The
from-scratch path rebuilds warm similarity metadata first and may produce
different downstream seeds.

The main BoxOffice entrypoints are:

- `boxoffice_repro_src/scripts/reproduce_verified_ten_seeds.sh`
- `boxoffice_repro_src/scripts/regenerate_seed_from_scratch.sh`
- `boxoffice_repro_src/scripts/run_boxoffice_pipeline.sh`
- `boxoffice_repro_src/runtime/bo-resume.sh`

## LongBench notes

The LongBench bundle ships the enriched files used for the paper:

- `musique_lb.json`
- `2wikimultihopqa_lb.json`
- `hotpotqa_lb.json`

These include the golden chunk metadata added during enrichment. The main
LongBench entrypoints are:

- `longbench_repro_src/scripts/regenerate_enriched_longbench.sh`
- `longbench_repro_src/scripts/run_longbench_pipeline.sh`
- `longbench_repro_src/runtime/run.sh`

## Bundle-specific documentation

- [Methods bundle](cache_methods_src/README.md)
- [BoxOffice bundle](boxoffice_repro_src/README.md)
- [LongBench bundle](longbench_repro_src/README.md)
