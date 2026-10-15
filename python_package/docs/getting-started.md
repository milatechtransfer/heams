# Getting started

## Setup

This package uses [uv](https://docs.astral.sh/uv/) for dependency management. From
`python_package/`, install the dependencies and all bundled memory plugins with:

```sh
uv sync --dev --extra all-memory-plugins --python 3.12
```

The `make setup` target will also run this command for you. From the project root:

```sh
make setup
```

### Environment variables

Create a `.env` file at the project root. The repository includes a template you
can copy and fill in:

```sh
cp .env_template .env
```

| Variable | Needed for |
|----------|------------|
| `OPENAI_API_KEY` | The bundled agents and LLM-as-judge evaluators; required for real runs |
| `HF_TOKEN` | Accessing datasets on Hugging Face |

!!! warning "API keys"
    `HF_TOKEN` must come from a Hugging Face account that has accepted the terms
    of the gated [GPQA dataset](https://huggingface.co/datasets/Idavidrein/gpqa). You can leave it empty if you are already logged
    in to that account with the Hugging Face CLI (`hf auth login`).

## Running experiments

### Run the benchmark smoke tests

The benchmark smoke tests run one small scenario of every benchmark (except
Retention policy) with in-context memory. They check that your setup, keys and
datasets work, and give you a first set of results to look at:

```sh
cd python_package
make run-smoke-benchmark
```

The target runs two `heams` commands:

```sh
uv run heams run --config configs/runs/smoke_test_benchmark.yaml --output-dir results/smoke_test_benchmark
uv run heams generate-report --results-dir results/smoke_test_benchmark
```

`heams run` executes every experiment in the run config and writes the raw
results; `heams generate-report` turns them into a readable `report.md` per
experiment. Datasets are downloaded on first use. Running the target again
resumes the existing run instead of starting over; to start fresh, use
`make run-overwrite CONFIG=smoke_test_benchmark OUTPUT_DIR=results/smoke_test_benchmark`.

### How the configs fit together

A **run config** under `configs/runs/` lists experiments. Each experiment pairs
an agent, a memory system and a benchmark, each described by its own YAML file:

```yaml
# configs/runs/smoke_test_benchmark.yaml (excerpt)
run_name: smoke_test_benchmark
experiments:
  - name: gpt56_luna_in_context_locomo
    agent_config_file: campaigns/locomo.yaml
    memory_system_config_file: campaigns/in_context.yaml
    benchmark_config_file: smoke_campaign/locomo.yaml
```

The file names are relative to one directory per component:

| Directory | Describes | Example fields |
|-----------|-----------|----------------|
| `configs/agents/` | The agent under test | `model_id`, `memory_strategy`, `instructions`, `model_args` |
| `configs/memory_systems/` | The memory system and its settings | `backend_config.type` (one of the names from `make memory-list`), `top_k`, optional `privacy_config` |
| `configs/benchmarks/` | The benchmark, its dataset slice and its evaluator | `type`, `limit`, `evaluator` (for example an LLM judge and its model) |

The smoke benchmark configs in `configs/benchmarks/smoke_campaign/` set
`limit: 1`, which keeps the run small. To run your own experiments, write a run
config that points at the component files you want and run it with
`make run CONFIG=<name> OUTPUT_DIR=<dir>`. See [CLI](cli.md) for the run config
format, [Memory backends](memory-backends.md) for memory-system configs, and the
[benchmark pages](index.md#benchmarks) for each benchmark's options.

### Running campaigns on a Slurm cluster

If you have access to a Slurm cluster, you can launch **campaigns**: sets of
runs that execute in parallel as separate Slurm jobs. See [Cluster](cluster.md)
for the campaign tooling and how to set it up.

## Exploring results

### Results browser

Open the results in a local, read-only web dashboard:

```sh
make browse RESULTS_DIR=results/smoke_test_benchmark
```

It prints a local URL (`http://127.0.0.1:8765/` by default) and opens it in your
browser; stop it with `Ctrl+C`. The dashboard has three levels:

- **Criterion table**: memory systems as columns, and rows grouped by
  dimension (Performance, Privacy, Efficiency), benchmark and criterion. Each
  cell shows the score and the number of samples behind it.
- **Experiment**: status, agent model, accuracy, a pass-rate breakdown per tag
  (question type, attack class, ...) and a searchable list of samples.
- **Sample**: the judge's records and reasoning, the full transcript with every
  tool call, efficiency metrics and the memory ledger.

`RESULTS_DIR` can be one run, a campaign, or the whole `results/` folder; the
browser searches it recursively and caches what it reads in
`_results_browser_cache/`. See [Results browser](results-browser.md) for how it
picks runs and computes each cell.

### Artefacts on disk

Each run writes one directory, with one subdirectory per experiment:

```text
results/smoke_test_benchmark/
├── run_config.<run_id>.yaml          # snapshot of the resolved run
├── gpt56_luna_in_context_locomo/
│   ├── experiment_config.<run_id>.yaml  # resolved agent, memory and benchmark configs
│   ├── runs.jsonl                     # one result per completed sample
│   ├── execution_traces.jsonl         # ordered trace of memory ops, LLM and tool calls
│   ├── run_status.json                # success flag and counts
│   ├── report.md                      # written by heams generate-report
│   └── workspace/                     # memory files used during the run
└── gpt56_luna_in_context_beam_100k/
    └── ...
```

`report.md` is the quickest read: metrics, a summary, and the detail of up to
10 samples (change that with `--max-examples`). `runs.jsonl` is the
machine-readable source for everything else. See
[Understanding results](results.md) for every file and field.

To evaluate your own memory system, see
[Adding a memory system](adding-memory-system.md).

## Makefile entry points

You can do everything with the `heams` CLI directly (`uv run heams ...` from
`python_package/`; see the [CLI reference](cli-reference.md)). For common tasks
we have also defined Makefile targets that wrap those commands. Run `make help`
in either directory to list them.

### Runs and results

Run these from `python_package/`:

| Target | What it does |
|--------|--------------|
| `make run CONFIG=<name> OUTPUT_DIR=<dir>` | Runs `configs/runs/<name>.yaml` into `<dir>`; running it again resumes the run |
| `make run-overwrite CONFIG=<name> OUTPUT_DIR=<dir>` | Same as `run`, but clears `<dir>` and starts over |
| `make run-smoke-benchmark` | Every benchmark except Retention policy at smoke scale with in-context memory, then generates reports; writes to `results/smoke_test_benchmark/` |
| `make run-smoke-memory-system` | Every memory system except Letta on one LoCoMo scenario, then generates reports; writes to `results/smoke_test_memory_system/` |
| `make run-smoke` | Both smoke targets above |
| `make browse RESULTS_DIR=<dir>` | Opens `<dir>` in the [results browser](results-browser.md) |
| `make memory-list` | Lists the memory systems the framework has discovered |
| `make memory-test` | Runs an isolated smoke diagnostic on the memory plugins |
| `make launch-campaign CAMPAIGN=<name>` | Submits a Slurm campaign; see [Cluster](cluster.md) |

### Development

Run these from the project root or `python_package/`:

| Target | What it does |
|--------|--------------|
| `make setup` | `uv sync --dev --extra all-memory-plugins --python 3.12` |
| `make ollama-install` | Installs Ollama (Homebrew on macOS when available) |
| `make test` | Full test suite, including integration tests; from the project root it also runs the Mem0 and HippoRAG plugin tests |
| `make test-ci` | Tests without integration and Ollama tests, as CI runs them; from the project root it also runs each plugin's CI tests |
| `make check` | Pre-commit lint and format checks across the repository |
| `make docs` / `make docs-serve` / `make docs-strict` | Builds or serves this site with [Zensical](https://zensical.org/) |
| `make setup-pre-commit` | Installs the pre-commit git hook (project root only) |

## Documentation (local)

Preview the docs site (works from project root or `python_package/`):

```sh
make docs-serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The site is built with
[Zensical](https://zensical.org/) using the config at the project root (`zensical.yml`).
Use `make docs` to build static output into `site/`, or
`make docs-strict` to fail on warnings (same as CI).

## Tests

`make test` runs the full suite, including integration tests. In
`python_package/` it runs `uv run --python "${PYTHON_VERSION:-3.12}" pytest -s tests/`;
the root target runs that and also the Mem0 and HippoRAG plugin tests.
`make test-ci` skips integration and Ollama tests, as CI does.

| Where you run | What runs |
|---------------|-----------|
| **Project root** | `python_package/` suite plus the Mem0 and HippoRAG plugin tests |
| **`python_package/`** | Full pytest suite, including integration tests |

From the project root (recommended before sharing changes):

```sh
make test
```

For the full suite, including integration tests:

```sh
cd python_package && make test
```

Run the repository-wide pre-commit checks from either directory:

```sh
make check
```

Run TinyAgent integration tests only (requires `ollama serve` and at least one local
model):

```sh
cd python_package
uv run --python "${PYTHON_VERSION:-3.12}" pytest -m "integration and requires_ollama" tests/agents/
```

The tests use `ollama:qwen2.5:7b`; to use another model, edit `MODEL_IDS` in
`tests/agents/conftest.py`.

## Next steps

- [Architecture](architecture.md) — config-driven runs, experiments, and factories
- [CLI](cli.md) — `heams run` and declarative YAML configs
- [Memory backends](memory-backends.md) — compare backends via experiment matrices
- [Benchmarks](index.md#benchmarks) — per-benchmark behavior and migration status
