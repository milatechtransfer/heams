# Changelog

## 1.0.0 (2026-10-16)

First public release of HEAMS: Holistic Evaluation of Agent Memory Systems, an
open framework for evaluating memory systems for AI agents on both what memory
adds to an agent and what it risks for the user.

### Benchmarks

Ten benchmarks run through the same `heams run` path:

- **Performance:** LoCoMo, BEAM (100K to 10M tokens), LongMemEval, Evo-Memory
  (single-turn), STALE.
- **Privacy:** Contextual Integrity, Transparency, Erasability & Editability,
  Retention policy, MPBench (memory poisoning).

Upstream datasets are downloaded on first use and keep their own licenses; see
[Citations & credits](python_package/docs/citations.md).

### Memory systems

- Built in: `in_context` and `no_memory` (the baseline).
- Bundled plugins: `milamem`, `basic_complete_memory_system`, `mem0`,
  `hipporag` (HippoRAG 2), `letta` (Letta / MemGPT), and `example.in_memory`, a
  minimal plugin to copy.
- New memory systems plug in as separate packages through the
  `heams.memory_systems` entry point, without changes to the framework; see
  [Adding a memory system](python_package/docs/adding-memory-system.md).

### Running and reviewing results

- `heams run` executes a campaign declared in YAML: each experiment pairs one
  agent, one memory system and one benchmark. Runs are resumable.
- `heams generate-report` writes a Markdown report per experiment, and
  `heams browse` opens a local web GUI over saved results.
- `heams memory list` and `heams memory test` list and check installed memory
  systems.
- Slurm scripts submit whole campaigns on a cluster.

### Requirements

Python 3.12.6 or newer, [uv](https://docs.astral.sh/uv/), and an
`OPENAI_API_KEY` for the bundled agents and LLM judges. Install by cloning the
repository and running `make setup`; see the [README](README.md).

### Notes

- `mapb`, the command's former name, still works as an alias for `heams`.
- The code is released under the MIT License; the datasets created for this
  project are released under CC BY 4.0.
