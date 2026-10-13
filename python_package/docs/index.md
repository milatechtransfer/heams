# HEAMS: Holistic Evaluation of Agent Memory Systems

This package contains the Python implementation of HEAMS. The repository `README.md` links to these pages; edit the Markdown
files in this folder as the source of truth.

## Overview

The broader project aims to build an extensible benchmarking library for agentic
private memory. The benchmark evaluates memory architectures across privacy and
performance dimensions.

!!! info "What you can do here"
    - Install the package and run campaigns with `heams run`
    - Compare memory systems by declaring multiple experiments in a run config
    - Read results and per-experiment reports; add new benchmarks and memory systems

!!! info "Config-driven benchmarks"
    Supported benchmark families are selected in typed experiment configs and
    materialized by `create_benchmark` as ordered `BenchmarkSample` workloads.
    Use `heams run --config ... --output-dir ...` to execute them.

## Quick start

From the project root:

```sh
make setup
cd python_package && make run-smoke
```

See [Getting started](getting-started.md) for setup (uv, Ollama, Makefile targets,
tests), [CLI](cli.md) for declarative runs, and [Understanding results](results.md)
for artefact layout. Preview docs locally with `make docs-serve` (project root or
`python_package/`).

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
| [Retention policy](benchmarks/retention_policy.md) | Runnable via `heams run` | Whether a privacy layer blocks prohibited facts at ingest and keeps permitted ones recallable |

## Contributing

To add a new factory-backed benchmark, see [Adding a benchmark](adding-benchmark.md).
Agents (Cursor, Claude Code, Copilot) should also load
[`.agents/skills/implement-benchmark/`](../../.agents/skills/implement-benchmark/).

To add a memory system as a plugin, see
[Adding a memory system](adding-memory-system.md).
