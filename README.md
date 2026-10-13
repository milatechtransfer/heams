# HEAMS: Holistic Evaluation of Agent Memory Systems

[![Tests](https://github.com/milatechtransfer/heams/actions/workflows/test.yml/badge.svg)](https://github.com/milatechtransfer/heams/actions/workflows/test.yml)
[![Check](https://github.com/milatechtransfer/heams/actions/workflows/check.yml/badge.svg)](https://github.com/milatechtransfer/heams/actions/workflows/check.yml)
[![Docs](https://github.com/milatechtransfer/heams/actions/workflows/docs.yml/badge.svg)](https://milatechtransfer.github.io/heams/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

HEAMS is an open framework for evaluating memory systems for
AI agents. It measures both what memory adds to an agent (recalling, updating,
and reusing information over long interactions) and what it risks for the user
(leaking, keeping, or failing to forget personal information). Agents, memory
systems, and benchmarks are declared in YAML, so comparing memory systems is a
matter of listing more experiments in a run config.

Documentation: <https://milatechtransfer.github.io/heams/>

## What's included

**Benchmarks**

| Benchmark | What it measures |
|---|---|
| [LoCoMo](python_package/docs/benchmarks/locomo.md) | Answering questions about very long, multi-session conversations |
| [BEAM](python_package/docs/benchmarks/beam.md) | Long-term memory abilities in conversations of 100K to 10M tokens |
| [MPBench](python_package/docs/benchmarks/mpbench.md) | Resistance to poisoned memories planted in the store |
| [LongMemEval](python_package/docs/benchmarks/long_mem_eval.md) | Recall over long, timestamped chat histories |
| [Evo-Memory (single-turn)](python_package/docs/benchmarks/evomemory_single_turn.md) | Reusing experience across ordered streams of tasks |
| [Transparency](python_package/docs/benchmarks/transparency.md) | Reporting accurately what is stored, without leaking unrelated memories |
| [Contextual Integrity](python_package/docs/benchmarks/contextual_integrity.md) | Sharing personal information only when the context makes it appropriate |
| [STALE](python_package/docs/benchmarks/stale.md) | Recognizing when a stored memory is out of date |
| [Erasability & Editability](python_package/docs/benchmarks/erasability_editability.md) | Deleting facts on request and propagating updates to dependent facts |
| [Retention policy](python_package/docs/benchmarks/retention_policy.md) | Blocking prohibited facts at ingest while keeping permitted ones recallable |

**Memory systems**

| Memory system | Kind | What it is |
|---|---|---|
| `in_context` | Built in | Recalls the most recent history that fits a token budget |
| `no_memory` | Built in | Stores nothing; the baseline |
| `milamem` | Plugin | Two-store memory with deduplication, conflict resolution, consolidation, and LLM-judged deletion |
| `basic_complete_memory_system` | Plugin | File-backed memory with configurable recall (BM25 by default), soft delete, and consolidation |
| `mem0` | Plugin | Wraps [Mem0](https://github.com/mem0ai/mem0) |
| `hipporag` | Plugin | Wraps [HippoRAG 2](https://github.com/OSU-NLP-Group/HippoRAG) |
| `letta` | Plugin | Wraps a [Letta](https://github.com/letta-ai/letta) (MemGPT) agent; needs Postgres, see its [README](memory_plugins/heams_memory_letta/README.md) |
| `example.in_memory` | Plugin | Minimal example to copy when writing your own |

`make setup` installs all the plugins. Run `uv run heams memory list` (from
`python_package/`) to see what is installed, and see
[Memory backends](python_package/docs/memory-backends.md) for configuration.

## Requirements

- git, make, and [uv](https://docs.astral.sh/uv/); `make setup` installs
  Python 3.12 through uv if needed (the package requires Python ≥ 3.12.6)
- An `OPENAI_API_KEY`: the bundled agents and LLM judges use OpenAI models
- For Evo-Memory (including `make run-smoke`), an `HF_TOKEN` from a Hugging Face
  account that has accepted the terms of the gated
  [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) dataset

## Install

```sh
git clone https://github.com/milatechtransfer/heams.git
cd heams
make setup
cp .env_template .env   # then fill in OPENAI_API_KEY (and HF_TOKEN)
```

Upstream datasets are downloaded on first use.

## Quickstart

Run one LoCoMo question end to end and turn the result into a report:

```sh
cd python_package
uv run heams run --config configs/runs/examples/quickstart.yaml --output-dir results/quickstart
uv run heams generate-report --results-dir results/quickstart
```

Each experiment's folder under `results/quickstart/` holds the raw results
(`runs.jsonl`) and a readable `report.md`. To compare memory systems, run
`configs/runs/examples/compare_memory_systems.yaml` the same way. `make run-smoke`
runs a small smoke check of every benchmark except Retention policy (which has
its own `retention_policy_smoke` config), plus four memory systems on LoCoMo.

To explore results across runs in a local web GUI, run
`uv run heams browse --results-dir results` (see
[Results browser](python_package/docs/results-browser.md)).

Next: [Getting started](python_package/docs/getting-started.md),
[CLI](python_package/docs/cli.md), and
[Understanding results](python_package/docs/results.md).

## Extending

- [Adding a memory system](python_package/docs/adding-memory-system.md): wrap
  your memory system as a plugin package, without changing the framework.
- [Adding a benchmark](python_package/docs/adding-benchmark.md).

## Citing

A paper describing the framework is in preparation. Until it is published,
please cite this repository if you use the framework in your research: use the
"Cite this repository" button on GitHub, or the metadata in
[`CITATION.cff`](CITATION.cff).

<!-- Replace the paragraph above with the paper's BibTeX once it is published. -->

Please also cite the original paper of each benchmark you report. Several
benchmarks are adapted to this framework's shared setup, so report numbers as
obtained with this framework rather than comparing them directly with the
original papers. [Citations & credits](python_package/docs/citations.md) lists
every benchmark, dataset, and library used, with its paper, license, and BibTeX.

## License

This repository is released under the [MIT License](LICENSE), with two
exceptions. The datasets created for this project
(`python_package/datasets/retention_policy/` and
`python_package/datasets/transparency/seed.json`) are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party files
included in the repository, such as the small upstream excerpts used as test
fixtures, keep their original licenses.

Upstream benchmark datasets are downloaded from their original sources and keep
their own licenses; some restrict use (for example, LoCoMo is non-commercial and
BEAM is share-alike). Check
[Citations & credits](python_package/docs/citations.md) before using a dataset,
especially commercially.

## Funding

This framework was developed by the Applied Machine Learning Research Team at
[Mila – Quebec AI Institute](https://mila.quebec) in a 2026 research
collaboration funded by [Mozilla](https://www.mozilla.org).
The collaboration studies privacy and user control in memory for AI agents, and
the framework is released openly so others can run the same evaluations on
their own memory systems. The views expressed are those of the authors and do
not necessarily reflect those of Mozilla.

## Team

**Core team:** Jacob Lavoie, Chirag Goel, Jordan Bannister, Amirhossein
Naghshzan, and Arsène Fansi Tchango.

**Contributors:** Marzia Nouri (Contextual Integrity and Retention policy
benchmarks) and Mehar Bhatia (STALE and Erasability & Editability benchmarks).
See the
[contributors page](https://github.com/milatechtransfer/heams/graphs/contributors)
for everyone who has contributed code.

## Acknowledgements

We thank Matt Harris and the team at Mozilla for their guidance
and feedback throughout the collaboration, and our colleagues at Mila for their
supervision, project management, and feedback on the design and results of this
work.
<!-- Name other people to thank here once the team confirms the list. -->

## Contact

For questions, bug reports, and research collaborations, please open an
[issue on GitHub](https://github.com/milatechtransfer/heams/issues).
To contribute code, a benchmark, or a memory system, see
[CONTRIBUTING.md](CONTRIBUTING.md).
