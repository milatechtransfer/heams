# MPBench memory-poisoning benchmark

!!! success "Runnable via `heams run`"
    MPBench is available on the config-driven factory path
    (`type: mpbench` in `heams.benchmarks.factory`).

MPBench evaluates **retrieval/behavior resistance (RSR)** after framework
injection of a poisoned memory. The upstream
[Digital-Trust-Lab/mp-bench](https://github.com/Digital-Trust-Lab/mp-bench)
dataset (Apache-2.0) is downloaded on first use into
`python_package/datasets/mpbench/`.

This v1 adapter does **not** measure write-phase ASR (agent reading untrusted
`context` and choosing to retain). Poison is seeded through
`MemoryPriming.entries` using the isolated `expected_memory` field (same priming
path as LoCoMo / LongMemEval). Upstream **benign / none-attack** controls
(`benign_data.jsonl.jsonl`, `attack_type: benign`) are never loaded; only the
six adversarial memory-poisoning classes from `adversarial_data.jsonl.jsonl`
(the upstream file names) are used.

Implementation lives under `heams/benchmarks/mpbench/`
(`config`, `dataset`, `samples`, `benchmark`, `judge`, `report`).

## Protocol

1. Setup primes `expected_memory` as memory **content only** (no attack-class
   or source-benchmark metadata — recall would otherwise surface it to the agent)
2. `UserStimulus` asks `retrieval_query`
3. `mpbench_llm_judge` decides whether the answer followed the poison
   (`poison_followed`); **resistance** = not followed

Attack classes (normalized): `explicit`, `conditional`, `compaction`, `fact`,
`precedent`, `skill`.

## Component configs

| Role | Bundled file |
|------|----------------|
| Agent | `configs/agents/testing/gpt4o_mini_recall_only_tool_use.yaml` |
| Memory (baseline) | `configs/memory_systems/testing/no_memory.yaml` |
| Benchmark (smoke) | `configs/benchmarks/testing/mpbench_smoke.yaml` (`limit_per_class: 1`) |
| Benchmark (default) | `configs/benchmarks/full_campaign/mpbench.yaml` (`limit_per_class: 10`) |
| Run (smoke) | `configs/runs/testing/mpbench_smoke.yaml` |

The benchmark YAML owns `answer_prompt` and the evaluator block. There are no
Python fallback prompt defaults. The answer prompt tells the agent not to guess
when it has no prior context — some retrieval queries are leading yes/no
questions that otherwise inflate false `poison_followed` on the no-memory
baseline.

## Run

From `python_package/`:

```sh
make run CONFIG=testing/mpbench_smoke OUTPUT_DIR=results/mpbench_smoke
# or
uv run heams run \
   --config configs/runs/testing/mpbench_smoke.yaml \
   --output-dir results/mpbench_smoke
```

Requires `OPENAI_API_KEY` for the default OpenAI agent and judge.

## Artefacts

After a successful run under the exact `output_dir`:

- per-experiment `runs.jsonl`, `run_status.json`, `experiment_config.*.yaml`
- per-experiment `report.md` after `heams generate-report`
- `_rollup/mpbench/report.md`, only if you generate the optional rollup from
  Python: resistance by memory backend and attack class, plus macro-average /
  worst-class summary

See [Understanding results](../results.md).

## Source and license

MP-Bench is by Dash et al., 2026
([arXiv:2606.04329](https://arxiv.org/abs/2606.04329); AIWILD workshop at ICML
2026). Its dataset,
[`Digital-Trust-Lab/mp-bench`](https://github.com/Digital-Trust-Lab/mp-bench), is
licensed **Apache-2.0**. It is downloaded on first use and not redistributed in
this repository. See [Citations & credits](../citations.md) for BibTeX.
