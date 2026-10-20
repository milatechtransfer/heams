# CLI reference

The `heams` command is defined in `heams.cli.cli`. Run `uv run heams --help` for the
live subcommand list. `mapb`, the command's former name, still works as an alias.

## Commands

### `heams run`

Execute a campaign from a declarative run config.

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | yes | Path to a run config under `python_package/configs/runs/` |
| `--output-dir` | yes | Exact directory for this run and its resumable artifacts |
| `--overwrite` | no | Clear and restart a recognized HEAMS run in the exact output directory |

```sh
cd python_package
uv run heams run --config configs/runs/smoke_test_benchmark.yaml --output-dir results/smoke_test_benchmark
```

Make wrappers (from `python_package/`):

```sh
make run-smoke
make run CONFIG=smoke_test_benchmark OUTPUT_DIR=results/smoke_test_benchmark
make run CONFIG=smoke_test_memory_system OUTPUT_DIR=results/smoke_test_memory_system
make run CONFIG=testing/locomo_smoke OUTPUT_DIR=results/locomo_smoke
make run CONFIG=testing/long_mem_eval_smoke OUTPUT_DIR=results/long_mem_eval_smoke
make run CONFIG=testing/evomemory_single_turn_smoke OUTPUT_DIR=results/evomemory_single_turn_smoke
make run CONFIG=testing/transparency_smoke OUTPUT_DIR=results/transparency_smoke
```

There are **no** CLI overrides for benchmark selection, model, memory system,
limits, concurrency, or examples directory. Those settings live in component YAML
files referenced by the run config.

### `heams memory list`

List stable type names for all built-in and installed external memory plugins.

```sh
uv run heams memory list
```

### `heams memory test`

Run an isolated retain/recall smoke diagnostic for every discovered plugin.

```sh
uv run heams memory test
```

Each plugin reports `pass`, `failure`, or `setup-required`. A skipped operation
is reported as a passing diagnostic with a note. The command does not run a
benchmark or require live agent services.

### `heams generate-report`

Generate a per-experiment Markdown report from saved artefacts, without
re-running benchmarks.

| Flag | Required | Description |
|------|----------|-------------|
| `--results-dir` | yes | Run directory containing per-experiment artefacts |
| `--max-examples` | no | Number of runs per experiment that get a full detail section, split between failures and successes (default 10) |
| `--detailed` | no | Also write `submetrics.csv` under `--results-dir` |

```sh
uv run heams generate-report --results-dir results/smoke_test_benchmark
```

For every subdirectory of `--results-dir` that has its own `runs.jsonl`
(i.e. every experiment), writes `report.md` alongside it containing top-level
metrics, a per-scenario summary table, and full detail (transcript, LLM judge
output, target vs. predicted answers) for up to `--max-examples` runs. An
experiment with a
missing or empty `runs.jsonl` is skipped with a warning rather than aborting
the rest.

With `--detailed`, also writes `submetrics.csv` directly under `--results-dir`: one
row per tagged sub-metric score and per efficiency metric across every
benchmark in the run, with columns `campaign, experiment, run_id,
memory_system, benchmark, metric_name, metric_value, agent_model_name,
timestamp` (see [cli.md](cli.md#generate-report-command) for the full column
semantics).

No `heams` command writes per-benchmark rollups (`_rollup/<type>/report.md`).
Generate them from saved artifacts by calling
`write_benchmark_rollup_reports(results_dir)` from Python; see
[Understanding results](results.md#per-benchmark-rollups-optional-python-only).

### `heams browse`

Open a local web GUI for navigating saved results, from the campaign-wide
criterion table down to individual benchmark samples.

| Flag | Required | Description |
|------|----------|-------------|
| `--results-dir` | yes | Run, campaign or whole `results/` directory; searched recursively |
| `--cache-dir` | no | Where to keep `runs.jsonl` indexes (default `<results-dir>/_results_browser_cache`) |
| `--host` | no | Interface to bind (default `127.0.0.1`) |
| `--port` | no | Port to listen on (default `8765`; `0` picks a free port) |
| `--no-open` | no | Do not open a browser tab automatically |

```sh
uv run heams browse --results-dir ../results
```

See [Results browser](results-browser.md) for what each view shows and how
`runs.jsonl` indexes are cached.

## Run config fields

Top-level keys in `configs/runs/*.yaml`:

| Key | Type | Description |
|-----|------|-------------|
| `run_name` | str | Human-readable campaign name |
| `experiments` | list | One or more named experiments (names must be unique) |

Each experiment:

| Key | Type | Description |
|-----|------|-------------|
| `name` | str | Unique experiment id (used in artefact directory names) |
| `agent_config_file` | relative YAML path | File under `configs/agents/` |
| `memory_system_config_file` | relative YAML path | File under `configs/memory_systems/` |
| `benchmark_config_file` | relative YAML path | File under `configs/benchmarks/` |

References may include nested directories but cannot be absolute, use `..`, or
use globs. Missing files fail validation before execution.

## Component config ownership

| Component | Owns |
|-----------|------|
| Agent (`configs/agents/`) | `model_id`, instructions, `memory_strategy`, tool limits, model args |
| Memory system (`configs/memory_systems/`) | Backend type, `top_k`, storage knobs, optional privacy wrapper |
| Benchmark (`configs/benchmarks/`) | Workload `type`, `examples_dir`, `limit` / `limit_stimuli`, prompts, evaluator |

For runnable benchmark configurations, see [LoCoMo](benchmarks/locomo.md),
[Contextual Integrity](benchmarks/contextual_integrity.md), and
[LongMemEval](benchmarks/long_mem_eval.md). For authoring a new factory-backed
benchmark, see [Adding a benchmark](adding-benchmark.md).

## Environment variables

The CLI calls `load_dotenv()` on startup. Common variables:

| Variable | Used for |
|----------|----------|
| `OPENAI_API_KEY` | The bundled OpenAI agents and every LLM judge (default `gpt-4o-mini`) |
| `HF_TOKEN` | Gated Hugging Face datasets (GPQA, used by Evo-Memory) |
| `LETTA_PG_URI` | Optional: an existing Postgres for the `letta` memory plugin; unset, it starts a private one (see its README) |

Place a `.env` file at the project root or `python_package/` (pytest also loads
`../.env`).

## Examples

### Bundled smoke runs

```sh
cd python_package
make run CONFIG=smoke_test_benchmark OUTPUT_DIR=results/smoke_test_benchmark
make run CONFIG=smoke_test_memory_system OUTPUT_DIR=results/smoke_test_memory_system
```

### Custom run config

```sh
cd python_package
uv run heams run \
	--config configs/runs/testing/locomo_smoke.yaml \
	--output-dir results/locomo_smoke
uv run heams run \
	--config configs/runs/testing/long_mem_eval_smoke.yaml \
	--output-dir results/long_mem_eval_smoke
```

Narrative walkthrough: [CLI](cli.md).
