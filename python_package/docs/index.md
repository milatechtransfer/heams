# HEAMS: Holistic Evaluation of Agent Memory Systems

This package contains the HEAMS benchmarking framework.

## Overview

The goal of this project is to build an extensible benchmarking library for the holistic evaluation of agent memory systems.
The benchmarks evaluate agent memory systems across dimensions of performance, privacy, user sovereignty and security.

!!! info "What you can do here"
    - Install the package and run experiments with `heams run`
    - Compare memory systems by declaring multiple experiments in a run config
    - Read results and per-experiment reports; add new benchmarks and memory systems

## Quick start

From the project root:

1. Install dependencies:

    ```sh
    make setup
    ```

2. Create your environment file from the template:

    ```sh
    cp .env_template .env
    ```

3. Open `.env` and set the keys the smoke run needs:

    ```env
    OPENAI_API_KEY=sk-...
    HF_TOKEN=hf_...
    ```

    `OPENAI_API_KEY` is required because the bundled agents and LLM-as-judge
    evaluators use OpenAI models. The Evo-Memory benchmark in the smoke run needs
    Hugging Face access to the gated GPQA dataset: either set `HF_TOKEN`, or leave
    it empty if you are already logged in with `hf auth login`. Either way, the
    account must have accepted the GPQA terms.

4. Move into `python_package/`. The remaining steps run from there:

    ```sh
    cd python_package
    ```

5. List the memory systems the framework has discovered, both built-in ones and
    installed plugins. Each name is a `type` you can use in a memory-system
    config:

    ```sh
    make memory-list
    ```

6. Run the benchmark smoke tests (one small scenario per benchmark, with
    in-context memory). The target also writes a `report.md` for each
    experiment under `python_package/results/smoke_test_benchmark/`:

    ```sh
    make run-smoke-benchmark
    ```

7. Explore the results in the local [results browser](results-browser.md):

    ```sh
    make browse RESULTS_DIR=results/smoke_test_benchmark
    ```

    It opens `http://127.0.0.1:8765/` in your browser, where you can drill down
    from per-benchmark scores to each sample's transcript and judge output. Stop
    it with `Ctrl+C`.

See [Getting started](getting-started.md) for more detailed setup instructions, [CLI](cli.md) for more information about the CLI commands that the framework supports, and [Understanding results](results.md)
for details about the result artefacts that are created by experiments.

## Guides

| Topic | Page |
|-------|------|
| System overview | [Architecture](architecture.md) |
| Memory systems as component YAML | [Memory backends](memory-backends.md) |
| `heams run` / `heams generate-report` / `heams browse` and config fields | [CLI reference](cli-reference.md) |
| Reading experiment dirs and reports | [Understanding results](results.md) |
| Reviewing results in a local web GUI | [Results browser](results-browser.md) |
| Slurm campaign tooling | [Cluster](cluster.md) |
| Papers, datasets, licenses, and BibTeX | [Citations & credits](citations.md) |

## Benchmarks

| Benchmark | Status | Description |
|-----------|--------|-------------|
| [LoCoMo](benchmarks/locomo.md) | Runnable via `heams run` | Long-context conversational QA with LLM-as-judge grading |
| [BEAM](benchmarks/beam.md) | Runnable via `heams run` | Long-context memory abilities at 100K–10M tokens with nugget scoring |
| [MPBench](benchmarks/mpbench.md) | Runnable via `heams run` | Memory-poisoning RSR via primed payload + LLM judge |
| [LongMemEval](benchmarks/long_mem_eval.md) | Runnable via `heams run` | Timestamped chat-history recall with LLM-as-judge grading |
| [Evo-Memory (single-turn)](benchmarks/evomemory_single_turn.md) | Runnable via `heams run` | Ordered memory stream over upstream single-turn datasets |
| [Transparency](benchmarks/transparency.md) | Runnable via `heams run` | Memory audit scenarios (BEAM fixtures) |
| [Contextual Integrity](benchmarks/contextual_integrity.md) | Runnable via `heams run` | Disclosure / leakage scenarios with CI LLM judge |
| [STALE](benchmarks/stale.md) | Runnable via `heams run` | Memory-staleness detection across three probing dimensions with LLM-as-judge grading |
| [Erasability & Editability](benchmarks/erasability_editability.md) | Runnable via `heams run` | Explicit-deletion and cascade-update tasks (MEME corpus) with LLM-as-judge grading |
| [Retention policy](benchmarks/retention_policy.md) | Runnable via `heams run` | Whether a memory system's governance blocks prohibited facts at ingest and keeps permitted ones recallable |

## Contributing

To add a new factory-backed benchmark, see [Adding a benchmark](adding-benchmark.md).
Agents (Cursor, Claude Code, Copilot) should also load
[`.agents/skills/implement-benchmark/`](../../.agents/skills/implement-benchmark/).

To add a memory system as a plugin, see
[Adding a memory system](adding-memory-system.md).
