# Benchmark CLI

The `heams` console script (defined in `pyproject.toml` as `[project.scripts]`;
`mapb`, the command's former name, still works as an alias) runs **declarative
campaigns** from YAML run configs. Each experiment in a run config pairs one
agent config, one memory system config, and one benchmark config. Experiments
execute serially.

Implementation lives in `heams.cli.cli`; `uv run python -m heams.cli …` is equivalent
to `uv run heams …` if you prefer the module form.

!!! info "Current benchmark support"
    The config-driven factory path supports **LoCoMo** (`type: locomo`),
    **BEAM** (`type: beam`), **MPBench** (`type: mpbench`), **LongMemEval**
    (`type: long_mem_eval`), **Evo-Memory single-turn**
    (`type: evomemory_single_turn`), **Transparency** (`type: transparency`),
    **Contextual Integrity** (`type: ci`), **STALE** (`type: stale`),
    **Erasability & Editability** (`type: ee`), and **Retention policy**
    (`type: retention_policy`).

For the full command surface see [CLI reference](cli-reference.md). For memory
component files see [Memory backends](memory-backends.md). For reading artefacts
see [Understanding results](results.md), or browse them with `heams browse`
([Results browser](results-browser.md)).

## Run a campaign

From `python_package/`:

```sh
make run-smoke
```

This runs both bundled smoke configs in separate exact output directories:
`results/smoke_test_benchmark/` and `results/smoke_test_memory_system/`.
For one run, provide both `CONFIG` and `OUTPUT_DIR`:

```sh
make run CONFIG=smoke_test_benchmark OUTPUT_DIR=results/smoke_test_benchmark
```

Equivalent direct invocation:

```sh
uv run heams run \
  --config configs/runs/smoke_test_benchmark.yaml \
  --output-dir results/smoke_test_benchmark
```

`--output-dir` names one exact run directory. A missing or empty directory starts
a run there; an existing recognized directory resumes from its saved snapshots
and durable result ledgers. The source `--config` is not re-resolved during
resume, so edits to it cannot change an initialized run. Non-HEAMS and malformed
targets are rejected without clearing them. Use `--overwrite` only when
intentionally starting over a recognized run:

```sh
uv run heams run --config configs/runs/smoke_test_benchmark.yaml --output-dir results/smoke_test_benchmark
uv run heams run --config configs/runs/smoke_test_benchmark.yaml --output-dir results/smoke_test_benchmark --overwrite
```

Or with Make:

```sh
make run CONFIG=smoke_test_benchmark OUTPUT_DIR=results/smoke_test_benchmark
make run CONFIG=smoke_test_memory_system OUTPUT_DIR=results/smoke_test_memory_system
make run CONFIG=testing/locomo_smoke OUTPUT_DIR=results/locomo_smoke
make run CONFIG=testing/long_mem_eval_smoke OUTPUT_DIR=results/long_mem_eval_smoke
make run CONFIG=testing/transparency_smoke OUTPUT_DIR=results/transparency_smoke
```

`CONFIG` is the relative path of a file under `configs/runs/` (without `.yaml`).

## Config tree

```text
configs/
  agents/            # model, instructions, memory_strategy, …
  memory_systems/    # backend type + retrieval knobs
  benchmarks/        # workload type, dataset, limits, prompts, evaluator
  runs/              # campaign: list of named experiments
```

Run configs **must** live under `configs/runs/`. Component references are
relative YAML paths under the matching subdirectory; nested paths such as
`campaigns/mem0.yaml` are supported.

The default `smoke_test_benchmark.yaml` contains one in-context experiment per
smoke-scale benchmark, except Retention policy, which needs a memory config
with a `privacy_config` (see [Retention policy](benchmarks/retention_policy.md)).
The complementary
`smoke_test_memory_system.yaml` contains six one-scenario LoCoMo experiments:
no-memory, in-context, MilaMem, Mem0, HippoRAG, and the example in-memory system. A representative benchmark smoke entry is:

```yaml
run_name: smoke_test_benchmark
experiments:
  - name: gpt56_luna_in_context_locomo
    agent_config_file: campaigns/locomo.yaml
    memory_system_config_file: campaigns/in_context.yaml
    benchmark_config_file: smoke_campaign/locomo.yaml
```

There is **no** CLI override for model, benchmark selection, memory backend,
limits, or prompts. Edit the component YAML (or add another experiment) instead.

## Inspect memory plugins

List all built-in and installed external memory plugins by stable type name:

```sh
uv run heams memory list
```

Run a small isolated diagnostic for each discovered plugin:

```sh
uv run heams memory test
```

The diagnostic validates default configuration, constructs the plugin in a
temporary runtime, and calls retain and recall with a sentinel value. Each
plugin reports `pass`, `failure`, or `setup-required`. A skipped operation is a
passing result with a note, so the `no_memory` baseline remains diagnosable.
The command does not execute a benchmark or require live agent services.

## Results layout

The output directory is the run directory itself. HEAMS does not create a
timestamped child below it:

- `output_dir/run_config.<run_id>.yaml` — snapshot of the resolved run
- `output_dir/<experiment_name>/` — one stable directory per experiment
  - `experiment_config.<run_id>.yaml` — resolved agent / memory / benchmark
  - `runs.jsonl` — one `RunResult` per completed scenario
  - `execution_traces.jsonl` — one ordered framework + agent trace per result
  - `run_status.json` — success / failure and counts
  - `workspace/` — per-run working directory for memory files

`heams run` writes no Markdown reports; run `heams generate-report` afterwards (see
below). Optional per-benchmark rollups are generated from Python; see
[Understanding results](results.md#per-benchmark-rollups-optional-python-only).

`run_id` looks like `YYYYMMDD-HHMMSS-<8 hex chars>`.

## Environment and models

The CLI auto-loads `.env`. Set `OPENAI_API_KEY` for the bundled OpenAI agents and
the LLM judges. Model choice belongs in the **agent**
component YAML (`model_id`), not on the command line.

Example agent preset (`configs/agents/testing/gpt4o_mini_tool_use.yaml`):

```yaml
type: agent
memory_strategy: tool_use
model_id: openai:gpt-4o-mini
instructions: >-
  …
model_args:
  temperature: 0.0
```

`memory_strategy` controls how the agent can access memory:

| Strategy | Behavior |
|----------|----------|
| `no_memory` | No memory access. |
| `tool_use` | The agent receives recall, retain, and delete tools. Backend maintenance is not agent-exposed. |
| `recall_only_tool_use` | The agent receives only the recall tool; the run cannot write memory through agent tools. |
| `deterministic` | The framework recalls and injects memory, then retains the completed turn. |
| `deterministic_with_delete` | As `deterministic`, but explicit memory-deletion requests invoke the backend's `delete()` method, bypass recall, and are not retained as Q/A memory. Backend delete capabilities still apply. |
| `recall_only_deterministic` | The framework recalls and injects memory without retaining the completed turn. |

Limits and dataset shaping belong in the **benchmark** component YAML (for
example `limit` / `limit_stimuli` on `configs/benchmarks/testing/locomo_smoke.yaml`).

## Generate-report command

```sh
uv run heams generate-report --results-dir results/smoke_test
```

Writes one `report.md` per experiment directory found under `results_dir`
(each experiment is identified by having its own `runs.jsonl`). Each report is
built entirely from that experiment's saved `runs.jsonl` - no re-run or live
`Scenario`/`Benchmark` objects required - and includes top-level metrics, a
per-scenario summary table, and full per-run detail (transcript, LLM judge
output, target vs. predicted answers) for up to `--max-examples` runs (10 by
default, split between failures and successes). When an
experiment also has `execution_traces.jsonl`, each detail includes the joined
framework/agent event sequence plus counts by event type and tool name. Older
result directories without trace sidecars render unchanged.

Pass `--detailed` to also write `submetrics.csv` directly under
`results_dir` - one row per tagged sub-metric score and per efficiency
metric (tokens, latency, memory operations), across every benchmark in the
run:

```sh
uv run heams generate-report --results-dir results/smoke_test --detailed
```

Each row has columns `campaign, experiment, run_id, memory_system, benchmark,
metric_name, metric_value, agent_model_name, timestamp`. `campaign` and
`run_id` are the run's `run_name`/`run_id`; `metric_name` is either
`<tag_key>=<tag_value>` for one of a benchmark's tagged evaluation scores
(each benchmark's judge tags its per-criterion scores with dimensions such as
category, dataset, or attack class - the same tags the "Mean sub-metrics"
tables in `report.md` are built from) or a named efficiency metric
(optionally prefixed with a memory-system component name, e.g.
`vector_store:Total operations`). Built from the same saved
`run_config.*.yaml` / `experiment_config.*.yaml` / `runs.jsonl` artifacts as
the Markdown reports above, so it can also be regenerated after the fact.

## Browse command

```sh
uv run heams browse --results-dir results/smoke_test
```

Serves a local, read-only web GUI over a run, a campaign, or the whole
`results/` folder: a criterion table that takes one run per memory system
and benchmark (the largest, or the latest finished, with its source shown),
each experiment's samples, and each sample's transcript and judge output.
See [Results browser](results-browser.md).
