# BoxOffice Reproduction Bundle

Self-contained bundle for reproducing the BoxOffice dataset seeds used in the
paper workflow.

This bundle includes:

- the canonical corpus and source data needed for seed generation
- bundled warm-retrieval metadata used by the verified seed workflow
- the code to regenerate warm metadata from scratch
- the code to regenerate the verified seeds
- the runtime/orchestration code used to run the BoxOffice experiments
- the already-generated verified seed outputs for `7`, `11`, `13`, `17`,
  `19`, `23`, `29`, `31`, `47`, `73`

## Scope

The primary artifact in this bundle is the released seed dataset:

- `seeds/boxoffice_filtered_v4_balanced_m4_s<seed>_full.jsonl`
- `seeds/boxoffice_filtered_v4_balanced_m4_s<seed>_eval.jsonl`

These are the seed files used by the downstream BoxOffice experiments.

The bundle can also optionally build the downstream semantic-pool artifacts:

- `pool_v4_balanced_m4/boxoffice_s<seed>_{chunks,embeddings,text_md5,infusion_k5,infusion_k10}`

Those pool artifacts are secondary to the seed dataset and are not required to
regenerate the seed JSONL files themselves.

## Default Behavior

By default, this bundle reproduces the verified seeds using the bundled warm
metadata already present under:

- `artifacts/rebuilt_warm_k5/fusion_docs_topk5.jsonl`

This is the default and recommended path if the goal is to reproduce the
verified published seeds as closely as possible.

The bundle can also rebuild the warm metadata from scratch from the bundled
corpus and source inputs. Rebuilding warm metadata may produce different
downstream seeds, which is expected.

## Requirements

This workflow expects an accelerator-backed runtime with the required model
weights available locally.

Validated defaults:

- visible accelerator slots: `0,1`
- filter models: `qwen3_8B,llama3.1-8BI,mistral-7B`
- warm-retrieval embedding model: `e5-base-v2`
- max concurrent eval jobs: `6`

Pinned Python package versions from the validated runtime are listed in:

- `requirements.txt`

## Bundle Layout

```text
boxoffice_repro_src/
  README.md
  requirements.txt

  corpus/
    source_corpus_100_films.jsonl
    canonical_corpus_100_chunks.jsonl

  data/
    phase2_boxoffice_balanced_matrix_s7.jsonl
    phase2_boxoffice_balanced_matrix_s11.jsonl
    supplemental_boxoffice_corpus_v1_50.jsonl

  prompts/
    INSTRUCTION_CHUNK.txt
    PROMPT_SPEC.md
    QUESTION_TEMPLATE.txt

  artifacts/
    rebuilt_warm_k5/
      fusion_docs_topk5.jsonl
      canonical_retrieval_topk5.jsonl
      soup_topk5.jsonl
      stats_topk5.json

  scripts/
    01_rebuild_warm_fusion_docs.sh
    02_regenerate_filtered_v2_from_corpus.sh
    03_regenerate_balanced_v4_from_filtered_v2.sh
    04_build_boxoffice_pool.sh
    regenerate_end_to_end.sh
    regenerate_seed_from_scratch.sh
    reproduce_verified_seed.sh
    reproduce_verified_five_seeds.sh
    reproduce_verified_ten_seeds.sh
    run_boxoffice_pipeline.sh

  runtime/
    run_boxoffice.py
    build_pool.py
    run.sh
    aggregate_groups.py
    aggregate_shards.py
    bo-resume.sh

  experiments/
    v1_helpers/
    v2_filtered/
    v4_balanced/
    pool/

  src/
    pilot_eval.py
    cachebend/

  seeds/
  manifests/
  validation/
  pool_v4_balanced_m4/
```

## Included Verified Seeds

This bundle already contains the verified generated seed outputs for:

- `7`
- `11`
- `13`
- `17`
- `19`
- `23`
- `29`
- `31`
- `47`
- `73`

Included files:

- `seeds/boxoffice_filtered_v4_balanced_m4_s7_{full,eval}.jsonl`
- `seeds/boxoffice_filtered_v4_balanced_m4_s11_{full,eval}.jsonl`
- `seeds/boxoffice_filtered_v4_balanced_m4_s13_{full,eval}.jsonl`
- `seeds/boxoffice_filtered_v4_balanced_m4_s17_{full,eval}.jsonl`
- `seeds/boxoffice_filtered_v4_balanced_m4_s19_{full,eval}.jsonl`
- `seeds/boxoffice_filtered_v4_balanced_m4_s23_{full,eval}.jsonl`
- `seeds/boxoffice_filtered_v4_balanced_m4_s29_{full,eval}.jsonl`
- `seeds/boxoffice_filtered_v4_balanced_m4_s31_{full,eval}.jsonl`
- `seeds/boxoffice_filtered_v4_balanced_m4_s47_{full,eval}.jsonl`
- `seeds/boxoffice_filtered_v4_balanced_m4_s73_{full,eval}.jsonl`
- matching `manifests/` files
- matching `validation/` files

## Quick Start

To reproduce the full verified ten-seed release using the bundled warm
metadata:

```bash
cd generate_boxoffice
bash scripts/reproduce_verified_ten_seeds.sh
```

To force a warm-metadata rebuild first:

```bash
cd generate_boxoffice
REBUILD_WARM=1 bash scripts/reproduce_verified_ten_seeds.sh
```

To also emit downstream pool artifacts while reproducing the seeds:

```bash
cd generate_boxoffice
BUILD_POOL=1 bash scripts/reproduce_verified_ten_seeds.sh
```

The original five-seed subset can still be reproduced with:

```bash
bash scripts/reproduce_verified_five_seeds.sh
```

## Running the BoxOffice experiments

The bundle also includes the runtime layer used to run the BoxOffice
methods on the released seeds. It expects the methods bundle to be available as
either:

- sibling default: `../cache_methods_src`
- override: `METHODS_BUNDLE=/abs/path/to/cache_methods_src`

The main pipeline wrapper is:

```bash
GPUS="0 1 2 3" bash scripts/run_boxoffice_pipeline.sh
```

This wrapper can:

- optionally regenerate the requested seeds first,
- build the corpus-wide semantic pool used by the warm methods,
- and then launch the BoxOffice evaluation runtime.

Useful knobs:

- `REGENERATE_SEEDS=1` regenerates the requested seeds before evaluation
- `FROM_SCRATCH=1` forces warm similarity rebuild before regenerating each seed
- `SEEDS="7 11"` narrows the evaluation seed set

The lower-level runtime entrypoints live under `runtime/`:

- `runtime/build_pool.py`
- `runtime/run_boxoffice.py`
- `runtime/run.sh`
- `runtime/bo-resume.sh`

## Regenerating one verified seed

Use:

```bash
cd generate_boxoffice
bash scripts/reproduce_verified_seed.sh <seed>
```

Supported verified seeds:

- `7`
- `11`
- `13`
- `17`
- `19`
- `23`
- `29`
- `31`
- `47`
- `73`

Examples:

```bash
bash scripts/reproduce_verified_seed.sh 7
bash scripts/reproduce_verified_seed.sh 11
bash scripts/reproduce_verified_seed.sh 13
bash scripts/reproduce_verified_seed.sh 17
bash scripts/reproduce_verified_seed.sh 19
bash scripts/reproduce_verified_seed.sh 23
bash scripts/reproduce_verified_seed.sh 29
bash scripts/reproduce_verified_seed.sh 31
bash scripts/reproduce_verified_seed.sh 47
bash scripts/reproduce_verified_seed.sh 73
```

Seed-specific verified settings encoded by the wrapper:

- `7` uses `source_tag=regenerated_v4_e2e_20260501_135032` and `EVAL_PER_CELL_OVERSAMPLED=100`
- `11` uses `source_tag=regenerated_v4_e2e_20260501_135032` and `EVAL_PER_CELL_OVERSAMPLED=100`
- `13` uses `source_tag=regenerated_filtered_v2_joint_probe` and `EVAL_PER_CELL_OVERSAMPLED=150`
- `17` uses `source_tag=regenerated_filtered_v2_joint_probe` and `EVAL_PER_CELL_OVERSAMPLED=150`
- `19` uses `source_tag=regenerated_filtered_v2_joint_probe` and `EVAL_PER_CELL_OVERSAMPLED=150`
- `23` uses `source_tag=regenerated_filtered_v2_joint_probe` and `EVAL_PER_CELL_OVERSAMPLED=100`
- `29` uses `source_tag=regenerated_filtered_v2_joint_probe` and `EVAL_PER_CELL_OVERSAMPLED=150`
- `31` uses `source_tag=regenerated_filtered_v2_joint_probe` and `EVAL_PER_CELL_OVERSAMPLED=150`
- `47` uses `source_tag=regenerated_filtered_v2_joint_probe` and `EVAL_PER_CELL_OVERSAMPLED=150`
- `73` uses `source_tag=regenerated_filtered_v2_joint_probe` and `EVAL_PER_CELL_OVERSAMPLED=150`

To rebuild warm metadata before a single-seed run:

```bash
REBUILD_WARM=1 bash scripts/reproduce_verified_seed.sh 23
```

To also emit the downstream pool for a single seed:

```bash
BUILD_POOL=1 bash scripts/reproduce_verified_seed.sh 23
```

## Generating One Seed From Scratch

If the goal is to make the "from zero" path explicit, use:

```bash
cd generate_boxoffice
bash scripts/regenerate_seed_from_scratch.sh 23
```

This wrapper:

- forces a warm similarity metadata rebuild,
- runs the filtered candidate-generation and model-filtering phase,
- emits the final balanced seed files for the requested seed,
- and optionally emits the downstream pool artifacts with `BUILD_POOL=1`.

Example:

```bash
BUILD_POOL=1 bash scripts/regenerate_seed_from_scratch.sh 23
```

## Manual Phase-by-Phase Execution

Warm metadata rebuild:

```bash
bash scripts/01_rebuild_warm_fusion_docs.sh
```

Phase 2 filtered-v2 generation:

```bash
DEVICE_POOL_CSV=0,1 MAX_PARALLEL_JOBS=6 \
MODELS_CSV=qwen3_8B,llama3.1-8BI,mistral-7B \
bash scripts/02_regenerate_filtered_v2_from_corpus.sh
```

Phase 3 balanced seed emission:

```bash
PROBE_ROOT=regenerated/filtered_v2_joint_probe_<TS> \
SEEDS_CSV=23 \
SOURCE_TAG=regenerated_filtered_v2_joint_probe \
bash scripts/03_regenerate_balanced_v4_from_filtered_v2.sh
```

Optional pool build:

```bash
SEEDS_CSV=23 DEVICES_CSV=0,1 bash scripts/04_build_boxoffice_pool.sh
```

## Changing the Filter Models

The default filter model set is:

- `qwen3_8B`
- `llama3.1-8BI`
- `mistral-7B`

To use a different subset:

```bash
DEVICE_POOL_CSV=0,1 MAX_PARALLEL_JOBS=6 \
MODELS_CSV=qwen3_8B,llama3.1-8BI \
bash scripts/02_regenerate_filtered_v2_from_corpus.sh
```

If the filter model set changes, the resulting seeds should be treated as new
outputs rather than reproductions of the verified release set.

## Notes

- The verified reproduction path reuses the bundled warm metadata by default.
- Rebuilding warm metadata is supported, but can change the resulting seeds.
- The wrapper scripts encode `EVAL_PER_CELL_OVERSAMPLED=100` for seeds `7`,
  `11`, and `23`, and `150` for seeds `13`, `17`, `19`, `29`, `31`, `47`, and
  `73`.
- The filter runtime uses the bundled `BaselineNosep` wiring in the included
  `src/cachebend` snapshot.

## Provenance

This bundle was assembled from the `generate_boxoffice` workflow and then
refined through direct verification against the authoritative `s7`, `s11`,
`s23`, `s47`, and `s73` seed outputs.
