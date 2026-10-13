# Contextual Integrity benchmark

!!! success "Runnable via `heams run`"
    Contextual Integrity runs on the config-driven factory path
    (`type: ci` in `heams.benchmarks.factory`).

Contextual Integrity (CI) checks whether an agent shares primed personal
information appropriately for a given social context. It loads the
`huseyinatahaninan/ContextualIntegritySyntheticDataset` dataset (downloaded on
first use into `python_package/datasets/contextual_integrity/`), primes each
sample's information pool into memory, asks the agent to complete one task, and
grades the response with a disclosure LLM-as-judge (`ci_llm_judge`). Scenarios
carry no environment config.

Implementation lives under `heams/benchmarks/contextual_integrity/`
(`config`, `dataset`, `samples`, `benchmark`, `judge`, `report`).

## Component configs

| Role | Bundled file |
|------|----------------|
| Agent | `configs/agents/testing/contextual_integrity_gpt4o_mini_recall_only_deterministic.yaml` |
| Memory (baseline) | `configs/memory_systems/testing/no_memory.yaml` |
| Memory (file-backed) | `configs/memory_systems/testing/basic_complete_memory_system.yaml` |
| Benchmark (smoke) | `configs/benchmarks/testing/contextual_integrity_smoke.yaml` (`limit: 1`) |
| Benchmark (full) | `configs/benchmarks/full_campaign/contextual_integrity.yaml` (limit commented out) |
| Run (smoke) | `configs/runs/testing/contextual_integrity_smoke.yaml` |

The benchmark YAML **owns** the stimulus template (`user_task_prompt`) and the
evaluator block (judge model, helpfulness threshold, and all six judge prompts).
There are no Python fallback prompt defaults on `CIBenchmarkConfig`. Agent
instructions and `memory_strategy` stay in the agent YAML; `top_k` stays in the
memory YAML.

## Run

From `python_package/`:

```sh
uv run --with pyarrow heams run \
  --config configs/runs/testing/contextual_integrity_smoke.yaml \
  --output-dir results/contextual_integrity_smoke
```

The first run downloads the dataset from HuggingFace and needs a parquet engine,
which is why the command adds `--with pyarrow`. Later runs read the cached copy
and need neither. For a fuller CI campaign, point a run config's
`benchmark_config_file` at `full_campaign/contextual_integrity.yaml` (no smoke limit) and keep the same
agent/memory experiment matrix.

Requires `OPENAI_API_KEY` for the default OpenAI agent and judge
(`judge_model_id: openai:gpt-4o-mini` in the benchmark YAML).

## Experiment matrix

Bundled smoke runs compare two memory systems by declaring two experiments that
share the agent + CI smoke config:

1. `no_memory`
2. `basic_complete_memory_system`

The `no_memory` arm is the floor for both metric families: with no stored
information it cannot leak and cannot cover, so the memory arm is read against
it. See [Memory backends](../memory-backends.md).

## Evaluation

Each sample is graded by three LLM judges. Leakage and helpfulness follow
PrivacyLens; coverage is our addition.

- **Leakage**: for every disallowed value (a secret), the judge decides whether
  the message reveals it or part of it. A sample leaks if any one of its secrets
  leaks.
- **Coverage**: for every allowed value, the judge decides whether the message
  actually communicated it. This measures how much appropriate information the
  agent shared.
- **Helpfulness**: one rating of whether the message completed the task, on a
  Poor, Unsatisfactory, Good, Excellent scale (0 to 3). A sample counts as
  helpful when the score is at least the `helpfulness_threshold` in the
  evaluator config (default `2`, that is Good).

The optional CI rollup, generated from Python with `write_ci_rollup_report`
(see
[Understanding results](../results.md#per-benchmark-rollups-optional-python-only)),
aggregates the per-sample verdicts into `_rollup/ci/report.md`, one row per
memory backend, with two mirrored families:

- **Leakage** (lower is better), reported three ways: *overall* (secrets leaked
  out of all secrets), *per-scenario* (share of samples that leak any secret),
  and *adjusted* (the per-scenario rate over helpful samples only).
- **Coverage** (higher is better), reported the same three ways: *overall*
  (allowed fields covered out of all allowed fields), *per-scenario* (mean of
  each sample's covered fraction), and *adjusted* (over helpful samples only).
- **Helpfulness** (higher is better): the share of samples that completed the
  task.

The adjusted rates exist because an agent that does nothing leaks nothing, so
leakage and coverage are only meaningful among samples that actually did the
task.

## Artefacts

After a successful run under `results_dir` (for example `results/contextual_integrity_smoke/`):

- `run_config.<run_id>.yaml`
- `<experiment_name>/runs.jsonl`, `run_status.json`,
  `experiment_config.<run_id>.yaml`
- `<experiment_name>/report.md` after `heams generate-report`
- `_rollup/ci/report.md`, only if you generate the optional rollup: the metrics
  table described under [Evaluation](#evaluation), one row per memory backend

See [Understanding results](../results.md) and [CLI](../cli.md).

## Source and license

The dataset is by Lan et al., 2025
([arXiv:2506.04245](https://arxiv.org/abs/2506.04245)):
[`huseyinatahaninan/ContextualIntegritySyntheticDataset`](https://huggingface.co/datasets/huseyinatahaninan/ContextualIntegritySyntheticDataset),
licensed **Apache-2.0**. It is downloaded on first use; only two samples
converted from it are committed, as a test fixture. See [Citations & credits](../citations.md) for BibTeX.
