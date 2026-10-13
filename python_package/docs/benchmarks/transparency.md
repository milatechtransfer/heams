# Transparency benchmark

The transparency benchmark is a typed `BaseBenchmark` on the config-driven
`heams run` path. Each sample primes target and background memories, asks an
audit question, and evaluates whether the agent discloses stored contents,
provenance, and relevant metadata without leaking unrelated memories.

For BEAM as a **memory-ability** benchmark (full chats, nugget scoring, 100K–10M
token scales), see [BEAM](beam.md). Transparency only reuses BEAM as an audit
fixture source.

Transparency v0 stays memory-only. Scenarios use the
`transparency_llm_judge` evaluator. Ground truth is attached on
`BenchmarkSample` at materialization time.

## Purpose

Agent memory is only useful for privacy-sensitive assistants if users can
**audit** what was stored. This benchmark measures whether the conversational
interface makes persistent state **legible** under user-initiated audit
questions. See
`src/heams/evaluation/scenarios/transparency_criterion_mapping.md` for the
criterion list.

## Scope and eligibility

Transparency v0 treats the memory system as a black box. Its fixture-backed
ground truth is the inventory that is expected to be stored and accessible at
audit time, not merely the entries offered to the backend during priming.
The benchmark has no backend-independent way to inspect the post-retention
store and construct ground truth from what actually survived ingestion.

Therefore, a backend is eligible for Transparency v0 only when it retains
**every primed target and background entry** and makes those entries available
to the agent at audit time. This is a precondition for interpreting a score as
memory transparency. Backends with retention gates, retention policies,
capacity eviction, expiry, ingestion filtering, or other behavior that can
discard or hide primed entries must be excluded unless that behavior is known
not to affect the fixture.

`no_memory` is not a valid Transparency v0 condition: its correct behavior is to
retain nothing and report no matching memory, which does not satisfy the
benchmark's assumed stored inventory. The bundled configs still run it as a
reference floor, which shows what an agent reports when nothing is stored, so
do not read its score as a transparency result.

Within this scope, the benchmark measures whether an agent faithfully and
completely reports relevant stored contents, provenance, temporal status, and
the absence of matching entries without disclosing unrelated stored
information. It does not measure retention quality, policy compliance during
ingestion, or whether a backend faithfully retains an arbitrary incoming
memory stream.

## Run

From `python_package/`:

```sh
# BEAM smoke (downloads and converts two samples)
uv run --with pyarrow heams run \
  --config configs/runs/testing/transparency_smoke.yaml \
  --output-dir results/transparency_smoke

# Combined benchmark smoke, including the same BEAM sample
uv run --with pyarrow heams run \
  --config configs/runs/smoke_test_benchmark.yaml \
  --output-dir results/smoke_test_benchmark
```

The shipped Transparency benchmark configs (`testing/`, `smoke_campaign/`,
`full_campaign/`) all use a BEAM source. The standalone smoke config converts two
samples and the combined smoke one; `full_campaign/transparency.yaml` converts
the full configured split. JSON remains available for custom datasets, but is
not used by the shipped run configs.

## Config ownership

| Config | Owns |
|--------|------|
| `configs/benchmarks/*/transparency*.yaml` | BEAM split, materialization and adapter limits, audit prompt, evaluator |
| `configs/agents/testing/transparency_gpt4o_mini_recall_only_tool_use.yaml` | Instructions + `memory_strategy: recall_only_tool_use` |
| `configs/memory_systems/testing/basic_complete_memory_system_transparency.yaml` | `top_k: 50` |
| `configs/runs/testing/transparency_smoke.yaml` | Experiment tuples (agent + memory + benchmark) |

Benchmark code does **not** overlay agent/memory after resolution. Policy that
used to live in `overlay_agent_and_memory` now lives in the YAML files above.
Choose only a backend that meets the [Scope and eligibility](#scope-and-eligibility)
precondition.

### BEAM adapter limits

On a BEAM source, `source.adapter` controls how many chat turns become primed
target memories after source-id dedupe:

```yaml
adapter:
  default_target_source_limit: 6
  extended_target_source_limit: 10
  extended_target_question_types:
    - contradiction_resolution
    - knowledge_update
    - event_ordering
```

`materialization_limit` caps how many downloaded BEAM questions are converted;
benchmark `limit` caps how many are executed.

## Package layout

```
src/heams/benchmarks/transparency/
  benchmark.py   # BaseBenchmark
  config.py      # TransparencyBenchmarkConfig
  dataset.py     # JSON loading / BEAM downloading
  beam.py        # BEAM → Transparency sample conversion
  samples.py     # TransparencySample → BenchmarkSample
  judge.py       # TransparencyLlmJudgeConfig
  report.py      # summary / detail / rollup sections
```

Factory dispatch: `create_benchmark` when `type: transparency`.

## Reporting

Run `heams generate-report` after a run for a `report.md` per experiment. The
optional Transparency rollup, `_rollup/transparency/report.md`, is not written by
any `heams` command; generate it from Python with `write_benchmark_rollup_reports`
(see
[Understanding results](../results.md#per-benchmark-rollups-optional-python-only)).
It reads saved `experiment_config.*.yaml` + `runs.jsonl` only and does not
download or reload benchmark datasets.

## Dataset notes

- BEAM source: `Mohammadta/BEAM` on Hugging Face
- Supported splits: `100K`, `500K`, `1M`
- Conversion lives in `heams.benchmarks.transparency.beam`

## Source and license

BEAM-sourced samples come from BEAM (Tavakoli et al., 2025,
[arXiv:2510.27246](https://arxiv.org/abs/2510.27246)), licensed **CC BY-SA 4.0**
and downloaded on first use. `datasets/transparency/seed.json` is our own,
released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
<!-- Cite the project whitepaper here once it is published. -->
See [Citations & credits](../citations.md).
