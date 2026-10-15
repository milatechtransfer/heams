# LoCoMo benchmark

!!! success "Runnable via `heams run`"
    LoCoMo is the reference suite on the config-driven factory path
    (`type: locomo` in `heams.benchmarks.factory`).

LoCoMo loads the upstream `locomo10.json` dataset from `snap-research/locomo`
(downloaded on first use into `python_package/datasets/locomo/`), primes the full conversation
into memory, and grades each per-question agent response with an LLM-as-judge
evaluator (`locomo_llm_judge`). Scenarios carry no environment config.

Each conversation is one scenario with every scored QA item in
`execution_sequence`. The intended evaluation semantic is **independent
questions**: `agent_conversation: independent` (the LoCoMo default) tells the
executor to call `reset_state()` before each QA so later questions do not see
prior benchmark prompts or answers. Primed conversation memory is unchanged.
`stateful` keeps agent messages across QA items; most other benchmarks use
that (BEAM is also independent). The policy is authored on the benchmark YAML,
copied onto the scenario, and serialized in `experiment_config.<run_id>.yaml`. A `limit_stimuli: 1` smoke
cannot validate isolation — use at least two QA items.

Do not branch the executor on the locomo benchmark name; honor
`scenario.agent_conversation`.

Implementation lives under `heams/benchmarks/locomo/`
(`config`, `dataset`, `samples`, `benchmark`, `judge`, `report`).

## Component configs

| Role | Bundled file |
|------|----------------|
| Agent | `configs/agents/testing/gpt4o_mini_recall_only_tool_use.yaml` |
| Memory (baseline) | `configs/memory_systems/testing/no_memory.yaml` |
| Memory (file-backed) | `configs/memory_systems/testing/basic_complete_memory_system.yaml` |
| Benchmark (smoke) | `configs/benchmarks/testing/locomo_smoke.yaml` (`limit: 1`, `limit_stimuli: 1`) |
| Benchmark (full) | `configs/benchmarks/full_campaign/locomo.yaml` (limits commented out) |
| Run (smoke) | `configs/runs/testing/locomo_smoke.yaml` |

The benchmark YAML **owns** answer prompts and the evaluator block. There are no
Python fallback prompt defaults on `LocomoBenchmarkConfig`. Agent instructions
and `memory_strategy` stay in the agent YAML; `top_k` stays in the memory YAML.

## Run

From `python_package/`:

```sh
make run CONFIG=smoke_test_memory_system OUTPUT_DIR=results/smoke_test_memory_system
# or
make run CONFIG=testing/locomo_smoke OUTPUT_DIR=results/locomo_smoke
# or
uv run heams run \
  --config configs/runs/testing/locomo_smoke.yaml \
  --output-dir results/locomo_smoke
```

`smoke_test_memory_system.yaml` compares six memory configurations (no-memory, in-context,
MilaMem, Mem0, HippoRAG, and the example in-memory system) on one smoke LoCoMo scenario. `smoke_test_benchmark.yaml` includes the same one-scenario
LoCoMo check with the in-context memory system alongside the other smoke benchmarks.

For a fuller LoCoMo campaign, point a run config's `benchmark_config_file` at
`full_campaign/locomo.yaml` (no smoke limits) and keep the same agent/memory
experiment matrix.

Requires `OPENAI_API_KEY` for the default OpenAI agent and judge
(`judge_model_id: openai:gpt-4o-mini` in the benchmark YAML).

## Experiment matrix

Bundled smoke runs compare two memory systems by declaring two experiments that
share agent + LoCoMo smoke config:

1. `no_memory`
2. `basic_complete_memory_system`

See [Memory backends](../memory-backends.md).

## Artefacts

After a successful run under the exact `output_dir` (for example
`results/locomo_smoke/`):

- `run_config.<run_id>.yaml`
- `<experiment_name>/runs.jsonl`, `run_status.json`,
  `experiment_config.<run_id>.yaml`
- `<experiment_name>/report.md` after `heams generate-report`
- `_rollup/locomo/report.md`, only if you generate the optional rollup from
  Python: category accuracy tables and per-run detail (question / answer /
  target answer)

See [Understanding results](../results.md) and [CLI](../cli.md).

## Source and license

LoCoMo is by Maharana et al., 2024
([arXiv:2402.17753](https://arxiv.org/abs/2402.17753)). Its dataset,
`locomo10.json` from [`snap-research/locomo`](https://github.com/snap-research/locomo),
is licensed **CC BY-NC 4.0** (non-commercial use only). It is downloaded on
first use and not redistributed in this repository. See [Citations & credits](../citations.md) for BibTeX.
