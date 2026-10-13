# Getting started

## Makefile entry points

Use the **project root** `Makefile` for setup, tests, and docs. Benchmark runs use
targets in `python_package/` (config-driven `heams run`).

| Target | From project root | From `python_package/` | Notes |
|--------|-------------------|------------------------|-------|
| `setup` | delegates | `uv sync --dev --extra all-memory-plugins --python $(PYTHON_VERSION)` | Install dependencies and all bundled memory plugins |
| `run` | — | `uv run --python $(PYTHON_VERSION) heams run --config configs/runs/$(CONFIG).yaml --output-dir $(OUTPUT_DIR)` | Requires `CONFIG` and `OUTPUT_DIR` |
| `run-smoke` | — | runs both bundled smoke configs | Writes to `results/smoke_test_benchmark/` and `results/smoke_test_memory_system/` |
| `run CONFIG=smoke_test_benchmark OUTPUT_DIR=results/smoke_test_benchmark` | — | runs `configs/runs/smoke_test_benchmark.yaml` | In-context memory system across every smoke-scale benchmark |
| `run CONFIG=smoke_test_memory_system OUTPUT_DIR=results/smoke_test_memory_system` | — | runs `configs/runs/smoke_test_memory_system.yaml` | Four campaign memory systems against one smoke LoCoMo scenario |
| `run CONFIG=testing/long_mem_eval_smoke OUTPUT_DIR=results/long_mem_eval_smoke` | — | runs `configs/runs/testing/long_mem_eval_smoke.yaml` | LongMemEval smoke with two memory systems |
| `ollama-install` | delegates | installs Ollama | Homebrew on macOS when available |
| `test` | delegates | `uv run --python $(PYTHON_VERSION) pytest -s tests/` | Run the full Python test suite |
| `docs` / `docs-serve` / `docs-strict` | delegates | runs [Zensical](https://zensical.org/) | Config at project root `zensical.yml` → `site/` |
| `check` | `uvx pre-commit run --all-files --show-diff-on-failure` | delegates | Lint and format checks across the project |

## Setup

This package uses [uv](https://docs.astral.sh/uv/) for dependency management.

!!! tip "Smoke run first"
    Verify wiring before a full campaign:

    ```sh
    make setup
    cd python_package && make run-smoke
    ```

Ensure Ollama is installed (for local model runs):

```sh
make ollama-install
```

### Environment variables

Create a `.env` file at the project root (or use `python_package/../.env`). The
repository includes a template you can copy and fill in:

```sh
cp .env_template .env
```

Add keys such as:

```env
OPENAI_API_KEY=sk-...
```

!!! warning "API keys"
    The bundled agents and LLM-as-judge evaluators use OpenAI models, so
    `OPENAI_API_KEY` must be set for real runs. Evo-Memory also needs an
    `HF_TOKEN` from a Hugging Face account that has accepted the terms of the
    gated GPQA dataset; `make run-smoke` includes Evo-Memory.

The CLI loads `.env` automatically via `load_dotenv`. Provider secrets are also
injected into agent configs from global settings when missing.

## Your first run

With `OPENAI_API_KEY` set, run one LoCoMo question end to end and turn the
result into a report:

```sh
cd python_package
uv run heams run --config configs/runs/examples/quickstart.yaml --output-dir results/quickstart
uv run heams generate-report --results-dir results/quickstart
```

`results/quickstart/` then holds one folder per experiment, with the raw
results in `runs.jsonl` and a readable `report.md` (metrics, a summary, and the
detail of up to 10 questions; change that with `--max-examples`).

To compare memory systems, run `configs/runs/examples/compare_memory_systems.yaml`
the same way: it asks the same question with no memory, with the whole
conversation in context, and with the example plugin. The example plugin only
finds memories that contain the whole query word for word, so it usually
scores like no memory; it shows how a plugin plugs in, not good retrieval. To
evaluate your own memory system, see
[Adding a memory system](adding-memory-system.md).

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
