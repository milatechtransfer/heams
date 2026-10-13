# Exemplar map

All paths relative to `python_package/` unless noted.

## Shape → clone

| Workload shape | Clone package | Why | Mirror tests |
|----------------|---------------|-----|--------------|
| Single-turn QA over primed dialogue history | `src/heams/benchmarks/locomo/` | Reference layout; YAML prompts; `record_to_benchmark_sample` | `tests/benchmarks/locomo/test_benchmark.py` |
| Same shape; judge under `evaluation/` | `src/heams/benchmarks/long_mem_eval/` | Locomo twin; template validators | `tests/benchmarks/long_mem_eval/test_benchmark.py` |
| External corpus + adapter (download/materialize) | `src/heams/benchmarks/transparency/` (+ `beam.py`) | Adapter boundary; early `limit`; typed fixtures | `tests/benchmarks/transparency/test_benchmark.py` |
| Multi-stimulus task streams | `src/heams/benchmarks/evomemory/` | Task, feedback and reset stimuli per stream; per-dataset match modes | `tests/benchmarks/evomemory/single_turn/` |
| Disclosure / leakage / policy judges | `src/heams/benchmarks/contextual_integrity/` | Custom judge + rollup without orchestrator edits | `tests/benchmarks/contextual_integrity/test_benchmark.py` |
| Multi-probe **one** sample (N stimuli) + HF/local dataset | `src/heams/benchmarks/stale/` | All-in-one judge; `tests/fixtures/smoke_e2e/` pattern | `tests/benchmarks/stale/test_benchmark.py`, `tests/benchmarks/stale/test_report.py` |
| Memory-poisoning / adversarial priming | `src/heams/benchmarks/mpbench/` | Priming-based attack samples + custom judge/report | `tests/benchmarks/mpbench/test_benchmark.py` |
| Erasure / editability | `src/heams/benchmarks/erasability_editability/` | Lifecycle-oriented judge + rollup | `tests/benchmarks/erasability_editability/test_benchmark.py` |
| Very long conversations primed once, rubric scoring | `src/heams/benchmarks/beam/` | HF splits as variants; nugget judge; rollup by ability and scale | `tests/benchmarks/beam/test_benchmark.py` |
| Memory operations only (`MemoryManipulation`), no agent turn | `src/heams/benchmarks/retention_policy/` | One sample per policy; per-fact records; preconditions checked at config resolve | `tests/benchmarks/retention_policy/test_benchmark.py` |

Orchestration wiring smoke (any suite): `tests/orchestration/test_run_orchestration.py`.

## Package layout to copy

```
src/heams/benchmarks/<name>/
  __init__.py      # Locomo-style TYPE_CHECKING + lazy __getattr__ if needed
  config.py        # type: Literal["…"]; settings.datasets_dir default; extra="forbid"
  dataset.py       # materialize / resolve paths; soft vs hard failures
  samples.py       # record → BenchmarkSample (no Scenario round-trip)
  benchmark.py     # BaseBenchmark; limit before conversion; skipped: list[SkipReason]
  judge.py         # optional; or under evaluation/
  report.py        # optional; registered in reporting/benchmark_rollup.py
```

## Minimal PR file list

Required for a new `type:`:

- `src/heams/benchmarks/<name>/` (package above)
- `src/heams/benchmarks/factory.py` (union + `create_benchmark` branch + `__all__`)
- `src/heams/benchmarks/__init__.py` (public re-exports; keep Locomo pattern; no drive-by cleanup)
- `configs/benchmarks/testing/<name>_smoke.yaml` (`limit: 1`)
- `configs/runs/testing/<name>_smoke.yaml`
- `tests/benchmarks/<name>/test_benchmark.py` (+ one-record fixture; e.g. `tests/fixtures/smoke_e2e/<name>/`)
- `docs/benchmarks/<name>.md`
- `zensical.yml`, `docs/index.md`, root `README.md` (nav links and benchmark table)

Add only when needed:

- New evaluator: implementation + `evaluation/factory.py` import + `EvaluatorConfig` union + `evaluation/__init__.py` export
- Dedicated agent YAML under `configs/agents/` (prefer reusing existing agent configs)
- `report.py` + entry in `heams/reporting/benchmark_rollup._ROLLUP_RENDERERS`
- Full (non-smoke) `configs/benchmarks/<name>.yaml` + `configs/runs/<name>.yaml`

Do **not** include for a normal benchmark PR:

- `src/heams/orchestration/**`
- Shared package “cleanup” unrelated to this suite

## What good suites demonstrate

| Suite | Takeaway |
|-------|----------|
| Locomo | Reference layout; clone first when unsure |
| Transparency | Adapter + early limit + artifact rollup |
| LongMemEval | Template validators; Locomo-shaped |
| EvoMemory | Direct sample conversion; early limit on load |
| Contextual Integrity | Typed path; scoped change; offline rollup |
| STALE | Multi-stimulus sample; do not touch orchestrator for reporting |
| MPBench | Adversarial priming workload; custom judge/report |
| Erasability / editability | Lifecycle-oriented samples + rollup |
| BEAM | Config variants (splits) declared in `report_info` |
| Retention policy | Agent-free samples; precondition checks in config resolution |
| Dataset defaults (any suite) | Always `settings.datasets_dir` |
